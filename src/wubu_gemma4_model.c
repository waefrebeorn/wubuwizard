/* wubu_gemma4_model.c — Gemma 4 12B dual-head-dim ISWA forward pass.
 *
 * Port of llama.cpp gemma4 model, adapted for wubuwizard quantized matmul.
 * Architecture: 48 layers, 40 sliding-window (HEAD_DIM=256) + 8 full-attention (HEAD_DIM=512).
 * Full attention at indices 5, 11, 17, 23, 29, 35, 41, 47 (every 6th from 5).
 * KV sharing: layers 40-47 reuse KV from layers 38-39.
 *
 * Uses quantized_matmul() from quantized_matmul.c for ALL quantized matmuls —
 * SSE2/AVX2-accelerated Q4_K vec_dot, OpenMP, and Q8_K-quantized activations.
 *
 * C11 — no VLA, no compound literals.
 */

#include "wubu_gemma4.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

/* ============ Tensor name helper ============ */
static char tn_buf[256];
#define TN(fmt, i) (snprintf(tn_buf, sizeof(tn_buf), fmt, i), tn_buf)

/* ============ Layer initialization ============ */

static void g4_layer_init(g4_layer_t *l, int idx) {
    memset(l, 0, sizeof(*l));
    l->layer_idx = idx;
    l->is_full = g4_layer_is_full(idx);
    l->share_kv = g4_layer_shares_kv(idx);
    l->kv_src_idx = g4_layer_kv_src(idx);
    l->head_dim = g4_head_dim(idx);
    l->q_dim = g4_q_dim(idx);
    l->kv_dim = g4_kv_dim(idx);
    l->kv_heads = g4_kv_heads(idx);
    l->n_rot = g4_n_rot(idx);
    l->rope_base = g4_rope_base(idx);
    l->kv_eq = false;
}

/* ============ Quantized weight helpers ============ */

static void qw_load(g4_qweight_t *w, gguf_ctx *ctx, const char *name) {
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) { memset(w, 0, sizeof(*w)); return; }
    w->ggml_type = t->ggml_type;
    w->n_elems = 1;
    for (int d = 0; d < t->n_dims; d++) w->n_elems *= t->dims[d];
    w->raw_bytes = gguf_raw_size(t->ggml_type, w->n_elems);
    uint64_t offset = t->data_offset;
    w->data = (const uint8_t *)ctx->data_blob + offset;
    w->data_owned = 0;
    if (w->data < (const uint8_t *)ctx->data_blob ||
        w->data + w->raw_bytes > (const uint8_t *)ctx->data_blob + ctx->data_blob_size) {
        fprintf(stderr, "[G4] ERROR: tensor %s out of bounds\n", name);
    }
}

static void qw_load_aliased(g4_qweight_t *w, g4_qweight_t *src) {
    memcpy(w, src, sizeof(*w));
    w->data_owned = 0;  /* alias doesn't own the data */
}

static float *g4_load_norm_f32(gguf_ctx *ctx, const char *name, int expected) {
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) return NULL;
    int64_t n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= t->dims[d];
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, buf, n)) { free(buf); return NULL; }
    if (n != expected)
        printf("[G4] Warning: %s has %lld elems (expected %d)\n", name, (long long)n, expected);
    return buf;
}

/* ============ Fast batched quantized matmul ============
 *
 * Uses quantized_matmul() from quantized_matmul.c which:
 *   - Quantizes F32 input to Q8_K once per row
 *   - Uses SSE2/AVX2 vec_dot for K-quant weights (Q4_K, Q5_K, Q6_K, IQ types)
 *   - OpenMP parallelized over columns with prefetching
 *   - Falls back to dequant+SGEMM for rare types
 *
 * c[M,N] = x[M,K] @ dequant(W[K,N])
 *
 * For Q4_0 (token_embd / output head): uses dequant-to-F32 then SGEMM
 * (not critical path — LM head called once per forward)
 */
void g4_batched_qmatmul(const float *x, const g4_qweight_t *w,
                               int M, int N, int K, float *c) {
    if (!w->data || w->raw_bytes <= 0) {
        memset(c, 0, (size_t)M * N * sizeof(float));
        return;
    }
    if (w->ggml_type == GGML_TYPE_Q4_0) {
        /* Q4_0: on-the-fly dequant + dot — no F32 buffer (was 13s bottleneck).
         * Each column: 18 bytes/32 elems, dequant to register, dot with x.
         * OpenMP parallel over output columns. */
        int row_bytes = (int)(w->raw_bytes / N);
        #pragma omp parallel for if(N > 256)
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < M; m++) {
                const uint8_t *col_data = w->data + (size_t)n * row_bytes;
                int64_t n_blocks = (K + 31) / 32;
                float sum = 0.0f;
                for (int64_t b = 0; b < n_blocks; b++) {
                    uint16_t d_bits;
                    memcpy(&d_bits, col_data + b * 18, 2);
                    uint32_t sign = (d_bits >> 15) & 1;
                    uint32_t exp  = (d_bits >> 10) & 0x1F;
                    uint32_t mant = d_bits & 0x03FF;
                    uint32_t f32_bits;
                    if (exp == 0) {
                        /* subnormal fp16: normalize (mant<<shift until bit 10),
                         * drop implicit 1, exponent 113-shift. The old code
                         * assumed bit 10 set -> wrong scale for mant < 512. */
                        if (mant == 0) f32_bits = sign << 31;
                        else {
                            int shift = __builtin_clz(mant) - 21;
                            uint32_t m = ((uint32_t)mant << shift) & 0x3FFu;
                            f32_bits = (sign << 31) | ((113u - (uint32_t)shift) << 23) | (m << 13);
                        }
                    }
                    else if (exp == 31)
                        f32_bits = (sign << 31) | (0xFF << 23) | (mant << 13);
                    else
                        f32_bits = (sign << 31) | ((uint32_t)(127 - 15 + exp) << 23) | (mant << 13);
                    float d;
                    memcpy(&d, &f32_bits, 4);
                    const uint8_t *qs = col_data + b * 18 + 2;
                    int64_t base = b * 32;
                    for (int j = 0; j < 32 && base + j < K; j++) {
                        /* ggml Q4_0 layout: elems 0-15 = LOW nibbles of
                         * qs[0..15], elems 16-31 = HIGH nibbles. */
                        int val = (j < 16) ? (qs[j] & 0x0F) : (qs[j - 16] >> 4);
                        sum += x[(size_t)m * K + base + j] * d * (float)(val - 8);
                    }
                }
                c[(size_t)m * N + n] = sum;
            }
        }
        return;
    }

    /* K-quant types: use quantized_matmul which has SIMD vec_dot + OpenMP over columns */
    int64_t col_stride = w->raw_bytes / N;
    for (int m = 0; m < M; m++) {
        quantized_matmul(x + (size_t)m * K, w->data, w->ggml_type,
                        (int64_t)K, (int64_t)N, (int64_t)col_stride,
                        c + (size_t)m * N);
    }
}

/* ============ Math ============ */

void g4_rms_norm(const float *x, const float *weight, int n, float eps, float *out) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * inv * weight[i];
}

float g4_gelu_tanh(float x) {
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

void g4_softcap(float *logits, int n, float cap) {
    float inv_cap = 1.0f / cap;
    #pragma omp parallel for if(n > 65536)
    for (int i = 0; i < n; i++) {
        logits[i] = tanhf(logits[i] * inv_cap) * cap;
    }
}

/* ============ RoPE (dual-mode) ============ */

static void g4_rope_apply(float *q, float *k, int n_tokens, int d_head,
                          int q_stride, int k_stride,
                          const int *positions, int n_rot, float rope_base,
                          const float *freqs, int is_full) {
    for (int t = 0; t < n_tokens; t++) {
        int pos = positions[t];
        for (int h = 0; h < q_stride; h += d_head) {
            for (int i = 0; i < n_rot; i += 2) {
                float freq_val = (is_full && freqs && i / 2 < d_head / 8)
                    ? freqs[i / 2] : 1.0f;
                float theta = (float)pos / powf(rope_base * freq_val,
                                                (float)i / (float)d_head);
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);
                int base = t * q_stride + h;
                float q0 = q[base + i];
                float q1 = q[base + i + 1];
                q[base + i]     = q0 * cos_t - q1 * sin_t;
                q[base + i + 1] = q0 * sin_t + q1 * cos_t;
                if (k) {
                    int kbase = t * k_stride + h;
                    float k0 = k[kbase + i];
                    float k1 = k[kbase + i + 1];
                    k[kbase + i]     = k0 * cos_t - k1 * sin_t;
                    k[kbase + i + 1] = k0 * sin_t + k1 * cos_t;
                }
            }
        }
    }
}

/* ============ KV cache ============ */

static void g4_kv_init(g4_kv_cache_t *cache, int kv_heads, int head_dim) {
    int init = 2048;
    cache->max = init;
    cache->size = 0;
    cache->kv_heads = kv_heads;
    cache->head_dim = head_dim;
    int elem = kv_heads * head_dim;
    cache->k = (float *)calloc((size_t)init * elem, sizeof(float));
    cache->v = (float *)calloc((size_t)init * elem, sizeof(float));
}

static void g4_kv_grow(g4_kv_cache_t *cache, int needed) {
    if (needed <= cache->max) return;
    int new_max = cache->max * 2;
    if (new_max < needed) new_max = needed + 1024;
    int elem = cache->kv_heads * cache->head_dim;
    cache->k = (float *)realloc(cache->k, (size_t)new_max * elem * sizeof(float));
    cache->v = (float *)realloc(cache->v, (size_t)new_max * elem * sizeof(float));
    memset(cache->k + (size_t)cache->max * elem, 0,
           (size_t)(new_max - cache->max) * elem * sizeof(float));
    memset(cache->v + (size_t)cache->max * elem, 0,
           (size_t)(new_max - cache->max) * elem * sizeof(float));
    cache->max = new_max;
}

static void g4_kv_store(g4_kv_cache_t *cache, int pos,
                        const float *k, const float *v) {
    g4_kv_grow(cache, pos + 1);
    int elem = cache->kv_heads * cache->head_dim;
    memcpy(cache->k + (size_t)pos * elem, k, (size_t)elem * sizeof(float));
    memcpy(cache->v + (size_t)pos * elem, v, (size_t)elem * sizeof(float));
    if (pos >= cache->size) cache->size = pos + 1;
}

/* ============ Init ============ */

bool g4_model_init(g4_model_t *model, const char *gguf_path) {
    memset(model, 0, sizeof(*model));
    model->n_layers = G4_N_LAYERS;
    model->max_ctx = G4_MAX_CTX;
    model->current_pos = 0;

    gguf_ctx *ctx = gguf_open(gguf_path);
    if (!ctx) return false;
    if (!gguf_buffer_data(ctx)) { gguf_close(ctx); return false; }

    printf("[G4] Loading model (%lld tensors, %zu MB data blob)\n",
           (long long)ctx->n_tensors, ctx->data_blob_size / 1048576);

    model->layers = (g4_layer_t *)calloc(model->n_layers, sizeof(g4_layer_t));
    if (!model->layers) { gguf_close(ctx); return false; }

    /* ---- Global tensors ---- */

    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t) { fprintf(stderr, "[G4] Missing token_embd.weight\n"); goto fail; }
    qw_load(&model->token_embd, ctx, "token_embd.weight");
    printf("  token_embd: %lld x %lld (type=%d, %lld MB)\n",
           (long long)t->dims[0], (long long)t->dims[1],
           t->ggml_type, (long long)model->token_embd.raw_bytes / 1048576);

    t = gguf_find_tensor(ctx, "output.weight");
    if (t) {
        qw_load(&model->output, ctx, "output.weight");
        model->tied_output = false;
        printf("  output: separate (%lld MB)\n", (long long)model->output.raw_bytes / 1048576);
    } else {
        qw_load_aliased(&model->output, &model->token_embd);
        model->tied_output = true;
        printf("  output: tied to token_embd\n");
    }

    t = gguf_find_tensor(ctx, "output_norm.weight");
    if (!t) { fprintf(stderr, "[G4] Missing output_norm.weight\n"); goto fail; }
    model->output_norm_weight = (float *)malloc(G4_HIDDEN * sizeof(float));
    gguf_read_tensor_f32(ctx, t, model->output_norm_weight, G4_HIDDEN);

    /* ---- Per-layer ---- */
    for (int i = 0; i < model->n_layers; i++) {
        g4_layer_t *l = &model->layers[i];
        g4_layer_init(l, i);

        int hd = l->head_dim;

        /* Norms */
        l->attn_norm_weight = g4_load_norm_f32(ctx, TN("blk.%d.attn_norm.weight", i), G4_HIDDEN);
        if (!l->attn_norm_weight) { fprintf(stderr,"[G4] attn_norm[%d] missing\n", i); goto fail; }

        l->attn_q_norm_weight = g4_load_norm_f32(ctx, TN("blk.%d.attn_q_norm.weight", i), hd);
        if (!l->attn_q_norm_weight) { fprintf(stderr,"[G4] attn_q_norm[%d] missing\n", i); goto fail; }

        l->attn_k_norm_weight = g4_load_norm_f32(ctx, TN("blk.%d.attn_k_norm.weight", i), hd);
        if (!l->attn_k_norm_weight) l->attn_k_norm_weight = l->attn_q_norm_weight;

        l->post_attn_norm_weight = g4_load_norm_f32(ctx, TN("blk.%d.post_attention_norm.weight", i), G4_HIDDEN);
        if (!l->post_attn_norm_weight) { fprintf(stderr,"[G4] post_attn_norm[%d] missing\n", i); goto fail; }

        l->ffn_norm_weight = g4_load_norm_f32(ctx, TN("blk.%d.ffn_norm.weight", i), G4_HIDDEN);
        if (!l->ffn_norm_weight) { fprintf(stderr,"[G4] ffn_norm[%d] missing\n", i); goto fail; }

        l->post_ffn_norm_weight = g4_load_norm_f32(ctx, TN("blk.%d.post_ffw_norm.weight", i), G4_HIDDEN);
        if (!l->post_ffn_norm_weight) { fprintf(stderr,"[G4] post_ffw_norm[%d] missing\n", i); goto fail; }

        /* Quantized weights */
        qw_load(&l->attn_q, ctx, TN("blk.%d.attn_q.weight", i));
        qw_load(&l->attn_out, ctx, TN("blk.%d.attn_output.weight", i));

        if (!l->share_kv) {
            qw_load(&l->attn_k, ctx, TN("blk.%d.attn_k.weight", i));
            t = gguf_find_tensor(ctx, TN("blk.%d.attn_v.weight", i));
            if (t) {
                qw_load(&l->attn_v, ctx, TN("blk.%d.attn_v.weight", i));
            } else {
                qw_load_aliased(&l->attn_v, &l->attn_k);
                l->kv_eq = true;
            }
        }

        qw_load(&l->ffn_gate, ctx, TN("blk.%d.ffn_gate.weight", i));
        qw_load(&l->ffn_up,   ctx, TN("blk.%d.ffn_up.weight", i));
        qw_load(&l->ffn_down, ctx, TN("blk.%d.ffn_down.weight", i));

        t = gguf_find_tensor(ctx, TN("blk.%d.layer_output_scale.weight", i));
        if (t) {
            gguf_read_tensor_f32(ctx, t, &l->layer_out_scale, 1);
            l->has_out_scale = true;
        }

        if (l->is_full) {
            t = gguf_find_tensor(ctx, TN("blk.%d.rope_freqs.weight", i));
            if (t) {
                l->rope_freqs = (float *)malloc((size_t)(hd / 2) * sizeof(float));
                gguf_read_tensor_f32(ctx, t, l->rope_freqs, hd / 2);
                l->has_rope_freqs = true;
            }
        }
    }

    model->data_blob = ctx->data_blob;
    model->data_blob_size = ctx->data_blob_size;
    ctx->data_blob = NULL;

    /* Save GGUF file handle and tensor info for GPU weight streaming.
     * We keep the file open so we can read weights directly from disk
     * during inference, bypassing the mmap'd data blob which causes
     * cudaMemcpy failures on WSL2 when file > physical RAM. */
    model->gguf_file = ctx->file;
    model->gguf_data_offset = ctx->data_blob_offset;
    model->gguf_n_tensors = (int)ctx->n_tensors;
    model->gguf_tensor_info = ctx->tensors;
    ctx->file = NULL;       /* model now owns the file handle */
    ctx->tensors = NULL;    /* model now owns the tensor info */
    ctx->data_blob = NULL;  /* already transferred above */

    /* Close the gguf context but NOT the file (model owns it now) */
    free(ctx);

    printf("[G4] %d layers loaded (~%d MB quantized + %.1f MB norms)\n",
           model->n_layers, (int)(model->data_blob_size / 1048576),
           (double)(model->n_layers * 6 * G4_HIDDEN * 4) / 1048576.0);

    model->kv_cache = (g4_kv_cache_t *)calloc(model->n_layers, sizeof(g4_kv_cache_t));
    for (int i = 0; i < model->n_layers; i++) {
        g4_layer_t *l = &model->layers[i];
        g4_kv_init(&model->kv_cache[i], l->kv_heads, l->head_dim);
    }

    return true;

fail:
    gguf_close(ctx);
    g4_model_destroy(model);
    return false;
}

/* ============ KV-cache helpers ============ */

static g4_kv_cache_t *g4_kv_get(g4_model_t *model, g4_layer_t *layer) {
    return &model->kv_cache[layer->kv_src_idx];
}

/* ============ Sliding window attention (OpenMP-parallel) ============ */

static void g4_sliding_attn_cpu(g4_model_t *model, g4_layer_t *layer, g4_kv_cache_t *cache,
                                const float *q, float *attn_out,
                                int N, const int *positions,
                                float *scores_buf, float *row_buf) {
    (void)row_buf;
    (void)model;
    int hd = layer->head_dim;
    int qd = layer->q_dim;
    int kvd = cache->kv_heads * cache->head_dim;
    int n_heads = G4_HEADS;
    int kv_heads = cache->kv_heads;
    int window = G4_SLIDING_WINDOW;

    memset(attn_out, 0, (size_t)N * qd * sizeof(float));

    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++) {
        int pos = positions[t];
        int kv_start = (pos > window) ? (pos - window) : 0;
        int kv_len = pos - kv_start + 1;
        if (kv_len > cache->size) kv_len = cache->size;
        if (kv_len <= 0) continue;
        int kv_end = kv_start + kv_len - 1;

        for (int hq = 0; hq < n_heads; hq++) {
            int kv_h = hq % kv_heads;
            const float *q_vec = q + (size_t)t * qd + (size_t)hq * hd;

            /* Scores in thread-local part of scores_buf */
            float *s = scores_buf + (size_t)(t * n_heads + hq) * window;

            float max_s = -FLT_MAX;
            for (int kp = kv_start; kp <= kv_end; kp++) {
                const float *kv = cache->k + (size_t)kp * kvd + (size_t)kv_h * hd;
                float score = 0.0f;
                for (int d = 0; d < hd; d++)
                    score += q_vec[d] * kv[d];
                score *= 1.0f / sqrtf((float)hd);
                s[kp - kv_start] = score;
                if (score > max_s) max_s = score;
            }

            float sum_e = 0.0f;
            for (int kp = kv_start; kp <= kv_end; kp++) {
                float e = expf(s[kp - kv_start] - max_s);
                s[kp - kv_start] = e;
                sum_e += e;
            }
            float inv_sum = 1.0f / (sum_e + 1e-10f);

            for (int d = 0; d < hd; d++) {
                float val = 0.0f;
                for (int kp = kv_start; kp <= kv_end; kp++) {
                    val += s[kp - kv_start] * inv_sum *
                           (cache->v + (size_t)kp * kvd + (size_t)kv_h * hd)[d];
                }
                attn_out[(size_t)t * qd + (size_t)hq * hd + d] = val;
            }
        }
    }
}

/* ============ Full attention (OpenMP-parallel) ============ */

static void g4_full_attn_cpu(g4_model_t *model, g4_layer_t *layer, g4_kv_cache_t *cache,
                             const float *q, float *attn_out,
                             int N, const int *positions,
                             float *scores_buf, float *row_buf) {
    (void)row_buf;
    (void)model;
    int hd = layer->head_dim;
    int qd = layer->q_dim;
    int kvd = cache->kv_heads * cache->head_dim;
    int n_heads = G4_HEADS;
    int kv_heads = cache->kv_heads;
    int kv_size = cache->size;

    memset(attn_out, 0, (size_t)N * qd * sizeof(float));

    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++) {
        int pos = positions[t];
        int kv_end = kv_size - 1;
        if (kv_end > pos) kv_end = pos;
        int kv_len = kv_end + 1;
        if (kv_len <= 0) continue;

        for (int hq = 0; hq < n_heads; hq++) {
            int kv_h = hq % kv_heads;
            const float *q_vec = q + (size_t)t * qd + (size_t)hq * hd;
            float *s = scores_buf + (size_t)(t * n_heads + hq) * kv_size;

            float max_s = -FLT_MAX;
            for (int kp = 0; kp <= kv_end; kp++) {
                const float *kv = cache->k + (size_t)kp * kvd + (size_t)kv_h * hd;
                float score = 0.0f;
                for (int d = 0; d < hd; d++)
                    score += q_vec[d] * kv[d];
                score *= 1.0f / sqrtf((float)hd);
                s[kp] = score;
                if (score > max_s) max_s = score;
            }

            float sum_e = 0.0f;
            for (int kp = 0; kp <= kv_end; kp++) {
                float e = expf(s[kp] - max_s);
                s[kp] = e;
                sum_e += e;
            }
            float inv_sum = 1.0f / (sum_e + 1e-10f);

            for (int d = 0; d < hd; d++) {
                float val = 0.0f;
                for (int kp = 0; kp <= kv_end; kp++) {
                    val += s[kp] * inv_sum *
                           (cache->v + (size_t)kp * kvd + (size_t)kv_h * hd)[d];
                }
                attn_out[(size_t)t * qd + (size_t)hq * hd + d] = val;
            }
        }
    }
}

/* ============ Single layer forward (CPU, fused matmuls in one parallel region) ============ */

void g4_layer_forward_cpu(g4_model_t *model, int il, float *x,
                                 float *normed, float *attn_out,
                                 float *q, float *k, float *v,
                                 float *ffn_gate, float *ffn_up, float *ffn_down_out,
                                 float *scores, float *row, int N, const int *positions) {
    g4_layer_t *layer = &model->layers[il];
    int hd = layer->head_dim;
    int qd = layer->q_dim;
    int kvd = layer->kv_dim;
    int kvh = layer->kv_heads;
    (void)row;

    /* ===== Pre-attention norm ===== */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++)
        g4_rms_norm(x + (size_t)t * G4_HIDDEN, layer->attn_norm_weight,
                    G4_HIDDEN, G4_RMS_EPS, normed + (size_t)t * G4_HIDDEN);

    /* ===== Q projection (batched quantized matmul) ===== */
    g4_batched_qmatmul(normed, &layer->attn_q, N, qd, G4_HIDDEN, q);

    /* ===== K, V projections ===== */
    if (!layer->share_kv) {
        g4_batched_qmatmul(normed, &layer->attn_k, N, kvd, G4_HIDDEN, k);
        if (layer->kv_eq) {
            memcpy(v, k, (size_t)N * kvd * sizeof(float));
        } else {
            g4_batched_qmatmul(normed, &layer->attn_v, N, kvd, G4_HIDDEN, v);
        }
    }

    /* ===== Q/K head norms ===== */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++) {
        for (int h = 0; h < G4_HEADS; h++)
            g4_rms_norm(q + (size_t)t * qd + (size_t)h * hd,
                        layer->attn_q_norm_weight, hd, G4_RMS_EPS,
                        q + (size_t)t * qd + (size_t)h * hd);
        if (!layer->share_kv) {
            for (int h = 0; h < kvh; h++) {
                g4_rms_norm(k + (size_t)t * kvd + (size_t)h * hd,
                            layer->attn_k_norm_weight, hd, G4_RMS_EPS,
                            k + (size_t)t * kvd + (size_t)h * hd);
                float ss = 0.0f;
                for (int d = 0; d < hd; d++)
                    ss += v[(size_t)t * kvd + (size_t)h * hd + d]
                        * v[(size_t)t * kvd + (size_t)h * hd + d];
                float inv_rms = 1.0f / sqrtf(ss / (float)hd + G4_RMS_EPS);
                for (int d = 0; d < hd; d++)
                    v[(size_t)t * kvd + (size_t)h * hd + d] *= inv_rms;
            }
        }
    }

    /* ===== RoPE ===== */
    if (!layer->share_kv) {
        g4_rope_apply(q, k, N, hd, qd, kvd, positions,
                      layer->n_rot, layer->rope_base, layer->rope_freqs, layer->is_full);
    } else {
        g4_rope_apply(q, NULL, N, hd, qd, 0, positions,
                      layer->n_rot, layer->rope_base, layer->rope_freqs, layer->is_full);
    }

    /* ===== Store K,V to KV cache ===== */
    if (!layer->share_kv) {
        for (int t = 0; t < N; t++) {
            int pos = positions[t];
            g4_kv_store(&model->kv_cache[il], pos,
                        k + (size_t)t * kvd, v + (size_t)t * kvd);
        }
    }

    g4_kv_cache_t *cache = g4_kv_get(model, layer);

    /* ===== Attention ===== */
    if (layer->is_full) {
        g4_full_attn_cpu(model, layer, cache, q, attn_out, N, positions, scores, row);
    } else {
        g4_sliding_attn_cpu(model, layer, cache, q, attn_out, N, positions, scores, row);
    }

    /* ===== Attention output projection ===== */
    g4_batched_qmatmul(attn_out, &layer->attn_out, N, G4_HIDDEN, qd, normed);

    /* ===== Post-attention norm + residual ===== */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++) {
        g4_rms_norm(normed + (size_t)t * G4_HIDDEN, layer->post_attn_norm_weight,
                    G4_HIDDEN, G4_RMS_EPS, normed + (size_t)t * G4_HIDDEN);
        for (int d = 0; d < G4_HIDDEN; d++)
            x[(size_t)t * G4_HIDDEN + d] += normed[(size_t)t * G4_HIDDEN + d];
    }

    /* ===== FFN ===== */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++)
        g4_rms_norm(x + (size_t)t * G4_HIDDEN, layer->ffn_norm_weight,
                    G4_HIDDEN, G4_RMS_EPS, normed + (size_t)t * G4_HIDDEN);

    /* Quantize input to Q8_K once, reuse across gate/up/down */
    g4_batched_qmatmul(normed, &layer->ffn_gate, N, G4_FFN, G4_HIDDEN, ffn_gate);
    g4_batched_qmatmul(normed, &layer->ffn_up,   N, G4_FFN, G4_HIDDEN, ffn_up);

    /* GELU + element-wise multiply (OpenMP parallel) */
    int ffn_elems = N * G4_FFN;
    #pragma omp parallel for if(ffn_elems > 4096)
    for (int i = 0; i < ffn_elems; i++)
        ffn_gate[i] = g4_gelu_tanh(ffn_gate[i]);
    #pragma omp parallel for if(ffn_elems > 4096)
    for (int i = 0; i < ffn_elems; i++)
        ffn_up[i] *= ffn_gate[i];

    g4_batched_qmatmul(ffn_up, &layer->ffn_down, N, G4_HIDDEN, G4_FFN, ffn_down_out);

    /* Post-FFN norm + residual */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++) {
        g4_rms_norm(ffn_down_out + (size_t)t * G4_HIDDEN, layer->post_ffn_norm_weight,
                    G4_HIDDEN, G4_RMS_EPS, ffn_down_out + (size_t)t * G4_HIDDEN);
        for (int d = 0; d < G4_HIDDEN; d++)
            x[(size_t)t * G4_HIDDEN + d] += ffn_down_out[(size_t)t * G4_HIDDEN + d];
    }

    if (layer->has_out_scale) {
        float s = layer->layer_out_scale;
        #pragma omp parallel for if(N * G4_HIDDEN > 4096)
        for (int i = 0; i < N * G4_HIDDEN; i++) x[i] *= s;
    }
}

/* ============ Forward pass ============ */

extern int omp_get_max_threads(void);

void g4_model_forward(g4_model_t *model, const float *embeddings, int B, int T, float *logits) {
    const int N = B * T;

    const int max_q_dim = G4_MAX_Q_DIM;
    const int max_kv_dim = G4_MAX_KV_DIM;
    const int max_dim2 = (G4_FFN > G4_HIDDEN ? G4_FFN : G4_HIDDEN) > max_q_dim
                         ? (G4_FFN > G4_HIDDEN ? G4_FFN : G4_HIDDEN)
                         : max_q_dim;

    /* Grow buffers */
    if (N > model->buf_size) {
        model->buf_size = N + 1024;
        int bs = model->buf_size;
        model->buf_q = (float *)realloc(model->buf_q, (size_t)bs * max_q_dim * sizeof(float));
        model->buf_k = (float *)realloc(model->buf_k, (size_t)bs * max_kv_dim * sizeof(float));
        model->buf_v = (float *)realloc(model->buf_v, (size_t)bs * max_kv_dim * sizeof(float));
        model->buf_attn_out = (float *)realloc(model->buf_attn_out, (size_t)bs * max_q_dim * sizeof(float));
        model->buf_normed = (float *)realloc(model->buf_normed, (size_t)bs * G4_HIDDEN * sizeof(float));
        model->buf_embd = (float *)realloc(model->buf_embd, (size_t)bs * G4_HIDDEN * sizeof(float));
        model->buf_ffn_gate = (float *)realloc(model->buf_ffn_gate, (size_t)bs * G4_FFN * sizeof(float));
        model->buf_ffn_up = (float *)realloc(model->buf_ffn_up, (size_t)bs * G4_FFN * sizeof(float));
        model->buf_ffn_out = (float *)realloc(model->buf_ffn_out, (size_t)bs * G4_HIDDEN * sizeof(float));
        model->buf_scores = (float *)realloc(model->buf_scores,
                                             (size_t)bs * G4_HEADS * G4_SLIDING_WINDOW * sizeof(float));
        model->buf_row = (float *)realloc(model->buf_row, (size_t)max_dim2 * sizeof(float));
    }

    float *x = model->buf_embd;
    float *normed = model->buf_normed;
    float *attn_out = model->buf_attn_out;
    float *q = model->buf_q;
    float *k = model->buf_k;
    float *v = model->buf_v;
    float *ffn_gate = model->buf_ffn_gate;
    float *ffn_up = model->buf_ffn_up;
    float *ffn_down_out = model->buf_ffn_out;
    float *row = model->buf_row;
    float *scores = model->buf_scores;

    memcpy(x, embeddings, (size_t)N * G4_HIDDEN * sizeof(float));

    int *positions = (int *)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) positions[i] = model->current_pos + i;

    for (int il = 0; il < model->n_layers; il++) {
        g4_layer_forward_cpu(model, il, x, normed, attn_out, q, k, v,
                            ffn_gate, ffn_up, ffn_down_out, scores, row, N, positions);
    }

    /* ===== Final norm ===== */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++)
        g4_rms_norm(x + (size_t)t * G4_HIDDEN, model->output_norm_weight,
                    G4_HIDDEN, G4_RMS_EPS, normed + (size_t)t * G4_HIDDEN);

    /* ===== LM head (Q4_0) ===== */
    g4_batched_qmatmul(normed, &model->output, N, G4_VOCAB, G4_HIDDEN, logits);

    /* ===== Softcap ===== */
    g4_softcap(logits, N * G4_VOCAB, G4_SOFTCAP);

    model->current_pos += N;
    free(positions);
}

/* ============ Token-based forward ============ */

void g4_model_forward_from_tokens(g4_model_t *model, const int *tokens, int B, int T, float *logits) {
    const int N = B * T;
    float *embd = (float *)malloc((size_t)N * G4_HIDDEN * sizeof(float));

    int row_bytes = (int)(model->token_embd.raw_bytes / G4_VOCAB);
    for (int i = 0; i < N; i++) {
        int tok = tokens[i];
        if (tok < 0 || tok >= G4_VOCAB) tok = 0;
        gguf_dequantize(model->token_embd.data + (size_t)tok * row_bytes,
                       model->token_embd.ggml_type, G4_HIDDEN,
                       embd + (size_t)i * G4_HIDDEN);
    }

    float scale = sqrtf((float)G4_HIDDEN);
    for (int i = 0; i < N * G4_HIDDEN; i++) embd[i] *= scale;

    g4_model_forward(model, embd, B, T, logits);
    free(embd);
}

/* ============ Single-token decode ============ */

void g4_model_decode(g4_model_t *model, int token, float *logits) {
    g4_model_forward_from_tokens(model, &token, 1, 1, logits);
}

/* ============ Reset ============ */

void g4_model_reset(g4_model_t *model) {
    model->current_pos = 0;
    for (int i = 0; i < model->n_layers; i++)
        model->kv_cache[i].size = 0;
}

/* ============ Destroy ============ */

void g4_model_destroy(g4_model_t *model) {
    for (int i = 0; i < model->n_layers; i++) {
        g4_layer_t *l = &model->layers[i];
        free(l->attn_norm_weight);
        free(l->attn_q_norm_weight);
        if (l->attn_k_norm_weight && l->attn_k_norm_weight != l->attn_q_norm_weight)
            free(l->attn_k_norm_weight);
        free(l->post_attn_norm_weight);
        free(l->ffn_norm_weight);
        free(l->post_ffn_norm_weight);
        free(l->rope_freqs);
    }
    free(model->layers);
    free(model->output_norm_weight);
    free(model->data_blob);

    if (model->kv_cache) {
        for (int i = 0; i < model->n_layers; i++) {
            free(model->kv_cache[i].k);
            free(model->kv_cache[i].v);
        }
        free(model->kv_cache);
    }

    free(model->buf_q);
    free(model->buf_k);
    free(model->buf_v);
    free(model->buf_attn_out);
    free(model->buf_normed);
    free(model->buf_embd);
    free(model->buf_ffn_gate);
    free(model->buf_ffn_up);
    free(model->buf_ffn_out);
    free(model->buf_scores);
    free(model->buf_row);

    /* Close GGUF file and free tensor info */
    if (model->gguf_file) {
        fclose(model->gguf_file);
    }
    free(model->gguf_tensor_info);
}
