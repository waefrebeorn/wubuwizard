/* wubu_inkling2_fwd.c -- EG-118 step 3: numeric primitives and the embedding path.
 *
 * Deliberately unoptimised: every weight is dequantised to f32 and multiplied with a plain dot product.
 * That is far too slow to generate text with, but it makes the graph checkable, and each primitive here
 * is written to be validated independently before anything is built on top of it.
 *
 * Tensor layout follows ggml: dims[0] is the fastest-varying axis, so a linear weight with
 * dims (n_in, n_out) is row-major as W[o * n_in + i].
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_inkling2.h"

/* ---- prototypes (definitions follow in dependency order) ---- */
void in2_rmsnorm_unweighted(const float *x, int n, float eps, float *out);
void in2_rmsnorm(const float *x, const float *w, int n, float eps, float *out);
void in2_matvec(const float *W, const float *x, int n_in, int n_out, float *y);
void in2_shortconv(const float *x, int n_tok, int channels, const float *w, int k, float *out);
void in2_l2_clip(float *x, const float *scale, int n_head, int dim, float threshold);
void in2_rel_bias_table(const float *rel_states, const float *P, int n_head, int d_rel, int extent, float *bias);
void in2_expert_act(const float *gate, const float *up, int n, float *out);
int  in2_dequant_slice(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t elem_off, int64_t n_elems, float *dst);
int  in2_router(const float *x, const float *W, const float *corr_bias, int n_expert, int d_model,
                int top_k, float scale, int *idx, float *w);
int  in2_moe(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
             int il, const float *x, int n_tok, float *out);
int  in2_layer_sliding(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                       int il, float *h, int n_tok, int pos0);
float *in2_tensor_to_f32(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t *n_out);

/* Unweighted RMS norm: x * rsqrt(mean(x^2) + eps). Used by the loop message gates. */
void in2_rmsnorm_unweighted(const float *x, int n, float eps, float *out)
{
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale;
}

/* Weighted RMS norm, the form used by attn_norm / ffn_norm / the full-layer input norms. */
void in2_rmsnorm(const float *x, const float *w, int n, float eps, float *out)
{
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

/* y[o] = dot(W + o*n_in, x) for o in [0, n_out). */
void in2_matvec(const float *W, const float *x, int n_in, int n_out, float *y)
{
    for (int o = 0; o < n_out; ++o) {
        const float *row = W + (size_t)o * n_in;
        double acc = 0.0;
        for (int i = 0; i < n_in; ++i) acc += (double)row[i] * x[i];
        y[o] = (float)acc;
    }
}

/* Static depthwise causal short convolution over a sequence.
 *
 * Reference semantics (Inkling2LoopShortConvolution): nn.Conv1d(kernel=k, groups=channels,
 * padding=k-1, bias=False), i.e. output at step t sees x[t-k+1..t], followed by
 *     out = (conv(x) + x) / sqrt(k + 1)
 * The residual add and the 1/sqrt(k+1) scale are part of the block, not an optimisation detail:
 * dropping them changes the magnitude by ~2.2x.
 *
 * w is ggml (k, channels): ne0 = k is the FASTEST axis, so the weight for channel c, tap j lives at
 * w[c*k + j]. That matches nn.Conv1d's [out_channels, 1, kernel] layout. Indexing it as
 * w[j*channels + c] (tap-major) reads a plausible-looking but wrong filter: the per-tap magnitudes come
 * out flat, which is how the mistake was caught. Tap index == backward distance, so tap 0 is the
 * current token.
 */
void in2_shortconv(const float *x, int n_tok, int channels, const float *w, int k, float *out)
{
    const float scale = 1.0f / sqrtf((float)(k + 1));
    for (int t = 0; t < n_tok; ++t) {
        for (int c = 0; c < channels; ++c) {
            float acc = 0.0f;
            for (int tap = 0; tap < k; ++tap) {
                const int src = t - tap;             /* tap 0 = current token */
                if (src < 0) continue;               /* left padding with zeros */
                acc += w[(size_t)c * k + tap] * x[(size_t)src * channels + c];
            }
            const float res = x[(size_t)t * channels + c];
            out[(size_t)t * channels + c] = (acc + res) * scale;
        }
    }
}

/* L2 clip, optionally preceded by a learned per-(head,dim) scale.
 *
 *   x *= scale;  row = x[h*dim .. h*dim+dim];  x *= min(1, threshold / (|row| + 1e-6))
 *
 * q uses this with a learned scale (attn_q_clip), k uses it bare, and the relative-position states use it
 * with rel_clip. The scale is stored in ggml order (dim, heads), i.e. scale[h*dim + d].
 */
void in2_l2_clip(float *x, const float *scale, int n_head, int dim, float threshold)
{
    for (int h = 0; h < n_head; ++h) {
        float *row = x + (size_t)h * dim;
        if (scale) for (int d = 0; d < dim; ++d) row[d] *= scale[(size_t)h * dim + d];
        double ss = 0.0;
        for (int d = 0; d < dim; ++d) ss += (double)row[d] * row[d];
        const float norm = sqrtf((float)ss);
        float f = threshold / (norm + 1e-6f);
        if (f > 1.0f) f = 1.0f;
        for (int d = 0; d < dim; ++d) row[d] *= f;
    }
}

/* Relative-position bias table: bias[h][dist] = sum_r rel_states[h][r] * P[dist][r].
 *
 * P is attn_rel_bias, stored in ggml order (d_rel, extent) so that r is the fastest axis:
 * P[dist * d_rel + r]. Computing the whole table once and indexing it by distance is what the reference
 * does, and it is cheaper than a gemm per (query,key) pair. Distances outside [0, extent) get 0, which
 * the caller handles by skipping them.
 */
void in2_rel_bias_table(const float *rel_states, const float *P, int n_head, int d_rel, int extent, float *bias)
{
    for (int h = 0; h < n_head; ++h) {
        const float *rs = rel_states + (size_t)h * d_rel;
        float *out = bias + (size_t)h * extent;
        for (int d = 0; d < extent; ++d) {
            const float *pr = P + (size_t)d * d_rel;
            double acc = 0.0;
            for (int r = 0; r < d_rel; ++r) acc += (double)rs[r] * pr[r];
            out[d] = (float)acc;
        }
    }
}

/* Read a block-aligned element range of a tensor and dequantise it. The MoE expert blocks are
 * (2*d_ff, d_model, n_expert) with expert slices contiguous and 256-aligned, so one expert can be read
 * without touching the other 127 (2.7 GB if dequantised whole). */
int in2_dequant_slice(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t elem_off, int64_t n_elems, float *dst)
{
    if (!ctx || !t || !t->info || !dst) return WUBU_INKLING2_ERR_MISSING_TENSOR;
    if (elem_off % 256 != 0) return WUBU_INKLING2_ERR_SHAPE;
    const int ty = t->info->ggml_type;
    /* wubu_fp4_raw_size returns -1 for n_elems <= 0, so the byte offset of element 0 has to be
     * spelled out rather than asked for. Expert 0 of every tensor hit this. */
    const int64_t byte_off = (elem_off == 0) ? 0 : gguf_raw_size(ty, elem_off);
    const int64_t nbytes   = gguf_raw_size(ty, n_elems);
    if (byte_off < 0 || nbytes <= 0) return WUBU_INKLING2_ERR_UNKNOWN_TYPE;
    unsigned char *raw = (unsigned char *)malloc((size_t)nbytes);
    if (!raw) return WUBU_INKLING2_ERR_IO;
    wubu_in2_tensor_t shifted = *t;
    shifted.file_offset = t->file_offset + byte_off;
    int rc = wubu_inkling2_read(ctx, &shifted, raw, nbytes);
    if (rc == WUBU_INKLING2_OK) gguf_dequantize(raw, ty, n_elems, dst);
    free(raw);
    return rc;
}

/* Softcap: tanh(x / c) * c. Bounds a value into (-c, c) without a hard clip. */
static float in2_softcap(float x, float c) { return tanhf(x / c) * c; }

/* Softcapped SwiGLU: the sigmoid sees the RAW gate; only the linear factor and `up` are softcapped.
 *   act = softcap(gate, 4.0) * sigmoid(gate) * softcap(up, 25.0)
 * Applying the softcap inside the sigmoid as well is a common misreading and changes the gate
 * non-linearly, so it is spelled out here. */
void in2_expert_act(const float *gate, const float *up, int n, float *out)
{
    for (int i = 0; i < n; ++i) {
        const float g = gate[i];
        const float sig = 1.0f / (1.0f + expf(-g));
        out[i] = in2_softcap(g, 4.0f) * sig * in2_softcap(up[i], 25.0f);
    }
}

/* Router: sigmoid scores, selection by score + correction bias, weights from the UNBIASED scores
 * renormalised to sum 1 and then multiplied by the routed scaling factor (8.0).
 *
 * The distinction matters: the correction bias only changes which experts are picked, never how much
 * they contribute. Returns the number of experts selected. */
int in2_router(const float *x, const float *W, const float *corr_bias, int n_expert, int d_model,
               int top_k, float scale, int *idx, float *w)
{
    float *scores = (float *)malloc((size_t)n_expert * sizeof(float));
    float *adj    = (float *)malloc((size_t)n_expert * sizeof(float));
    if (!scores || !adj) { free(scores); free(adj); return 0; }
    for (int e = 0; e < n_expert; ++e) {
        const float *row = W + (size_t)e * d_model;
        double acc = 0.0;
        for (int i = 0; i < d_model; ++i) acc += (double)row[i] * x[i];
        const float s = 1.0f / (1.0f + expf(-(float)acc));
        scores[e] = s;
        adj[e] = s + (corr_bias ? corr_bias[e] : 0.0f);
    }
    /* partial selection sort for top_k */
    char *taken = (char *)calloc((size_t)n_expert, 1);
    for (int k = 0; k < top_k; ++k) {
        int best = -1; float bestv = -1e30f;
        for (int e = 0; e < n_expert; ++e) if (!taken[e] && adj[e] > bestv) { bestv = adj[e]; best = e; }
        if (best < 0) break;
        taken[best] = 1; idx[k] = best;
    }
    double sum = 0.0;
    for (int k = 0; k < top_k; ++k) sum += scores[idx[k]];
    const float inv = (float)(1.0 / (sum + 1e-20));
    for (int k = 0; k < top_k; ++k) w[k] = scores[idx[k]] * inv * scale;
    free(scores); free(adj); free(taken);
    return top_k;
}

/* ------------------------------------------------------------------------------------------- */
/* A complete sliding-layer forward over n_tok tokens starting at absolute position pos0.       */
/*                                                                                              */
/* Order is taken from Inkling2LoopSlidingDecoderLayer.forward, which is not the conventional    */
/* pre-norm block:                                                                              */
/*                                                                                              */
/*   block_residual = h                                                                         */
/*   h = attention_input(block_states, h)        # depth mix, layer>0 only; empty -> bare rmsnorm */
/*   h = attn(input_layernorm(h))                                                               */
/*   h = post_attention_conv(h)                  # (conv(x)+x)/sqrt(k+1)                        */
/*   boundary = (il % block_size == 0)                                                          */
/*   if boundary: block_states += block_residual      # residual banked, NOT added               */
/*   else:        residual = h = block_residual + h                                            */
/*   h = mlp_input(block_states, h)             # bare rmsnorm when block_states is empty        */
/*   h = moe(post_attention_layernorm(h))                                                       */
/*   h = post_mlp_conv(h)                                                                       */
/*   return residual + h                                                                        */
/*                                                                                              */
/* For a single token the attention reduces to one key plus the sink, so the output is exactly  */
/*   v * sigmoid(score - sink_logit)                                                            */
/* which is used as the instrument: the sink must dampen the output and the weights must not     */
/* silently renormalise to 1.                                                                   */
/* ------------------------------------------------------------------------------------------- */

void in2_matvec_row_major(const float *W, const float *x, int n_in, int n_out, float *y);

/* softmax weights over keys [0, n_key) plus a sink logit; writes alpha[n_key]. */
static void in2_softmax_sink(const float *logits, int n_key, float sink, float *alpha)
{
    float mx = sink;
    for (int i = 0; i < n_key; ++i) if (logits[i] > mx) mx = logits[i];
    double sum = exp((double)(sink - mx));
    for (int i = 0; i < n_key; ++i) { alpha[i] = expf(logits[i] - mx); sum += alpha[i]; }
    const float inv = (float)(1.0 / sum);
    for (int i = 0; i < n_key; ++i) alpha[i] *= inv;   /* note: sum(alpha) < 1 by the sink share */
}

int in2_layer_sliding(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                      int il, float *h, int n_tok, int pos0)
{
    const wubu_in2_layer_t *L = &m->layer[il];
    const int d = cfg->d_model, NH = cfg->n_head, NKV = cfg->n_kv_head, HD = cfg->head_dim;
    const float eps = cfg->rms_eps;

    const int DBG = getenv("IN2_DEBUG") != NULL;
    float *an = in2_tensor_to_f32(ctx, &L->attn_norm, NULL);
    if (!an) { if (DBG) fprintf(stderr, "[L%d] attn_norm failed\n", il); return WUBU_INKLING2_ERR_IO; }
    if (DBG) fprintf(stderr, "[L%d] step1 ok\n", il);

    /* 1. attn_norm (weighted). */
    float *xn = (float *)calloc((size_t)n_tok * d, sizeof(float));
    for (int t = 0; t < n_tok; ++t) in2_rmsnorm(h + (size_t)t * d, an, d, eps, xn + (size_t)t * d);

    /* 2. q: project, scale+clip per head. */
    const int qdim = NH * HD, kdim = NKV * HD;
    float *Wq = in2_tensor_to_f32(ctx, &L->wq, NULL);
    float *qcl = in2_tensor_to_f32(ctx, &L->q_clip, NULL);
    float *q = (float *)calloc((size_t)n_tok * qdim, sizeof(float));
    if (!Wq || !q) { if (DBG) fprintf(stderr, "[L%d] q alloc/proj failed Wq=%p q=%p\n", il, (void*)Wq, (void*)q);
        free(an); free(xn); free(Wq); free(qcl); free(q); return WUBU_INKLING2_ERR_IO; }
    if (DBG) fprintf(stderr, "[L%d] step2 ok (qcl=%p)\n", il, (void*)qcl);
    for (int t = 0; t < n_tok; ++t) {
        in2_matvec(Wq, xn + (size_t)t * d, d, qdim, q + (size_t)t * qdim);
        in2_l2_clip(q + (size_t)t * qdim, qcl, NH, HD, 22.6274166f);
    }

    /* 3. k, v: project, static short conv, then bare L2 clip (k) / unweighted rms (v). */
    float *Wk = in2_tensor_to_f32(ctx, &L->wk, NULL);
    float *Wv = in2_tensor_to_f32(ctx, &L->wv, NULL);
    float *kc = in2_tensor_to_f32(ctx, &L->k_conv, NULL);
    float *vc = in2_tensor_to_f32(ctx, &L->v_conv, NULL);
    float *kraw = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *vraw = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *kt = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *vt = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *k = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *v = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    int rc = WUBU_INKLING2_OK;
    if (!Wk || !Wv || !kc || !vc || !kraw || !vraw || !kt || !vt || !k || !v) rc = WUBU_INKLING2_ERR_IO;
    for (int t = 0; t < n_tok && rc == WUBU_INKLING2_OK; ++t) {
        in2_matvec(Wk, xn + (size_t)t * d, d, kdim, kraw + (size_t)t * kdim);
        in2_matvec(Wv, xn + (size_t)t * d, d, kdim, vraw + (size_t)t * kdim);
    }
    if (rc == WUBU_INKLING2_OK) {
        in2_shortconv(kraw, n_tok, kdim, kc, 4, kt);
        in2_shortconv(vraw, n_tok, kdim, vc, 4, vt);
        for (int t = 0; t < n_tok; ++t) {
            memcpy(k + (size_t)t * kdim, kt + (size_t)t * kdim, sizeof(float) * kdim);
            in2_l2_clip(k + (size_t)t * kdim, NULL, NKV, HD, 22.6274166f);
            memcpy(v + (size_t)t * kdim, vt + (size_t)t * kdim, sizeof(float) * kdim);
            float *row = v + (size_t)t * kdim;
            for (int hh = 0; hh < NKV; ++hh) in2_rmsnorm_unweighted(row + hh * HD, HD, eps, row + hh * HD);
        }
    }

    /* 4. relative bias table from the normed input, plus sinks. */
    float *Wrel = in2_tensor_to_f32(ctx, &L->wrel, NULL);
    float *rcl = in2_tensor_to_f32(ctx, &L->rel_clip, NULL);
    float *P = in2_tensor_to_f32(ctx, &L->rel_bias, NULL);
    float *Wsk = in2_tensor_to_f32(ctx, &L->sink, NULL);
    const int DREL = 16, EXT = cfg->swa_window;
    float *rs = (float *)calloc((size_t)NH * DREL, sizeof(float));
    float *bias = (float *)calloc((size_t)NH * EXT, sizeof(float));
    float *sink = (float *)calloc((size_t)n_tok * NH, sizeof(float));
    if (Wrel && rcl && P && Wsk && rs && bias && sink) {
        for (int t = 0; t < n_tok; ++t) {
            in2_matvec(Wrel, xn + (size_t)t * d, d, NH * DREL, rs);
            in2_l2_clip(rs, rcl, NH, DREL, 8.0f);
            in2_rel_bias_table(rs, P, NH, DREL, EXT, bias);        /* per token; see note below */
            in2_matvec(Wsk, xn + (size_t)t * d, d, NH, sink + (size_t)t * NH);
        }
    } else { rc = WUBU_INKLING2_ERR_IO; if (DBG) fprintf(stderr, "[L%d] rel/sink failed Wrel=%p rcl=%p P=%p Wsk=%p\n", il, (void*)Wrel, (void*)rcl, (void*)P, (void*)Wsk); }
    if (DBG) fprintf(stderr, "[L%d] step4 rc=%d\n", il, rc);

    /* 5. attention per head. The bias table is token-dependent, so it is recomputed per token; the
     *    loop below therefore walks queries and rebuilds only the distances it needs. */
    float *attn_out = (float *)calloc((size_t)n_tok * qdim, sizeof(float));
    float *alpha = (float *)calloc((size_t)n_tok, sizeof(float));
    float *logits = (float *)calloc((size_t)n_tok, sizeof(float));
    if (attn_out && alpha && logits && rc == WUBU_INKLING2_OK) {
        for (int t = 0; t < n_tok; ++t) {
            const int kvh_stride = NH / NKV;
            /* rebuild this token's bias table (same weights, same xn) */
            in2_matvec(Wrel, xn + (size_t)t * d, d, NH * DREL, rs);
            in2_l2_clip(rs, rcl, NH, DREL, 8.0f);
            in2_rel_bias_table(rs, P, NH, DREL, EXT, bias);
            for (int hh = 0; hh < NH; ++hh) {
                const int kvh = hh / kvh_stride;
                int n_key = 0;
                for (int j = 0; j <= t; ++j) {
                    const int dist = (pos0 + t) - (pos0 + j);
                    if (dist >= EXT) continue;                    /* sliding window + rel extent */
                    const float *qa = q + (size_t)t * qdim + hh * HD;
                    const float *kb = k + (size_t)j * kdim + kvh * HD;
                    double acc = 0.0;
                    for (int e = 0; e < HD; ++e) acc += (double)qa[e] * kb[e];
                    logits[n_key] = (float)acc * (1.0f / sqrtf((float)HD)) + bias[hh * EXT + dist];
                    ++n_key;
                }
                in2_softmax_sink(logits, n_key, sink[(size_t)t * NH + hh], alpha);
                float *o = attn_out + (size_t)t * qdim + hh * HD;
                for (int j = 0; j < n_key; ++j) {
                    const int dist = t - j;   /* keys are consecutive, so index maps directly */
                    (void)dist;
                    const float a = alpha[j];
                    const float *vb = v + (size_t)j * kdim + kvh * HD;
                    for (int e = 0; e < HD; ++e) o[e] += a * vb[e];
                }
            }
        }
    } else { rc = WUBU_INKLING2_ERR_IO; if (DBG) fprintf(stderr, "[L%d] step5 failed attn_out=%p alpha=%p logits=%p\n", il, (void*)attn_out, (void*)alpha, (void*)logits); }
    if (DBG) fprintf(stderr, "[L%d] step5 rc=%d\n", il, rc);

    /* 6. gated output projection into its own buffer: h still holds the layer input, which the
     *    residual needs. sigmoid(attn_gate(xn)) * attn_out, then attn_output. */
    float *Wg = in2_tensor_to_f32(ctx, &L->wgate, NULL);
    float *Wo = in2_tensor_to_f32(ctx, &L->wo, NULL);
    float *gate = (float *)calloc((size_t)n_tok * qdim, sizeof(float));
    float *ao = (float *)calloc((size_t)n_tok * d, sizeof(float));
    if (Wg && Wo && gate && ao && rc == WUBU_INKLING2_OK) {
        for (int t = 0; t < n_tok; ++t) {
            in2_matvec(Wg, xn + (size_t)t * d, d, qdim, gate + (size_t)t * qdim);
            for (int i = 0; i < qdim; ++i) {
                const float gv = 1.0f / (1.0f + expf(-gate[(size_t)t * qdim + i]));
                gate[(size_t)t * qdim + i] = attn_out[(size_t)t * qdim + i] * gv;
            }
            /* attn_output is (qdim, d): out[o] = sum_u W[o*qdim + u] * gate[u] */
            for (int o = 0; o < d; ++o) {
                double acc = 0.0;
                for (int u = 0; u < qdim; ++u) acc += (double)Wo[(size_t)o * qdim + u] * gate[(size_t)t * qdim + u];
                ao[(size_t)t * d + o] = (float)acc;
            }
        }
    } else { rc = WUBU_INKLING2_ERR_IO; if (DBG) fprintf(stderr, "[L%d] step6 failed Wg=%p Wo=%p gate=%p ao=%p\n", il, (void*)Wg, (void*)Wo, (void*)gate, (void*)ao); }
    if (DBG) fprintf(stderr, "[L%d] step6 rc=%d\n", il, rc);

    if (getenv("IN2_DUMPX") && n_tok == 1) {
        FILE *q1 = fopen("d_xn.bin","wb"); if (q1) { fwrite(xn,sizeof(float),(size_t)n_tok*d,q1); fclose(q1); }
        FILE *q2 = fopen("d_q.bin","wb");  if (q2) { fwrite(q,sizeof(float),(size_t)n_tok*qdim,q2); fclose(q2); }
        FILE *q3 = fopen("d_k.bin","wb");  if (q3) { fwrite(k,sizeof(float),(size_t)n_tok*kdim,q3); fclose(q3); }
        FILE *q4 = fopen("d_v.bin","wb");  if (q4) { fwrite(v,sizeof(float),(size_t)n_tok*kdim,q4); fclose(q4); }
        FILE *q5 = fopen("d_attn.bin","wb"); if (q5) { fwrite(attn_out,sizeof(float),(size_t)n_tok*qdim,q5); fclose(q5); }
        FILE *q6 = fopen("d_ao.bin","wb"); if (q6) { fwrite(ao,sizeof(float),(size_t)n_tok*d,q6); fclose(q6); }
        FILE *q7 = fopen("d_gate.bin","wb"); if (q7) { fwrite(gate,sizeof(float),(size_t)n_tok*qdim,q7); fclose(q7); }
    }

    /* 7. post-attention short conv, then the residual rule:
     *    boundary (il % block_size == 0): the attention output becomes the residual and block_residual
     *    is banked into block_states (a no-op here, single block);
     *    otherwise: residual = block_residual + attention output. */
    float *pc = in2_tensor_to_f32(ctx, &L->attn_post_conv, NULL);
    float *res = (float *)calloc((size_t)n_tok * d, sizeof(float));
    if (pc && res && rc == WUBU_INKLING2_OK) {
        in2_shortconv(ao, n_tok, d, pc, 4, res);
        const int boundary = (cfg->attn_res_block > 0) && (il % cfg->attn_res_block == 0);
        for (int t = 0; t < n_tok; ++t)
            for (int i = 0; i < d; ++i) {
                const size_t idx = (size_t)t * d + i;
                if (!boundary) res[idx] = h[idx] + res[idx];
            }
    } else { rc = WUBU_INKLING2_ERR_IO; if (DBG) fprintf(stderr, "[L%d] step7 failed pc=%p res=%p\n", il, (void*)pc, (void*)res); }
    if (DBG) fprintf(stderr, "[L%d] step7 rc=%d\n", il, rc);

    /* 8. mlp_input: with a single block in block_states the learned-query attention has one key, so
     *    softmax is degenerate and the mix reduces to an unweighted RMS norm. */
    float *fn = in2_tensor_to_f32(ctx, &L->ffn_norm, NULL);
    float *moein = (float *)calloc((size_t)n_tok * d, sizeof(float));
    float *moeout = (float *)calloc((size_t)n_tok * d, sizeof(float));
    if (fn && moein && moeout && rc == WUBU_INKLING2_OK) {
        for (int t = 0; t < n_tok; ++t) {
            float *tmp = (float *)calloc((size_t)d, sizeof(float));
            in2_rmsnorm_unweighted(res + (size_t)t * d, d, eps, tmp);
            in2_rmsnorm(tmp, fn, d, eps, moein + (size_t)t * d);
            free(tmp);
        }
        rc = in2_moe(ctx, m, cfg, il, moein, n_tok, moeout);
    } else { rc = WUBU_INKLING2_ERR_IO; if (DBG) fprintf(stderr, "[L%d] step8 failed fn=%p moein=%p moeout=%p (moe rc=%d)\n", il, (void*)fn, (void*)moein, (void*)moeout, rc); }
    if (DBG) fprintf(stderr, "[L%d] step8 rc=%d\n", il, rc);

    if (getenv("IN2_DUMPX") && n_tok == 1) {
        FILE *r1 = fopen("d_res.bin","wb"); if (r1) { fwrite(res,sizeof(float),(size_t)n_tok*d,r1); fclose(r1); }
        FILE *r2 = fopen("d_moein.bin","wb"); if (r2) { fwrite(moein,sizeof(float),(size_t)n_tok*d,r2); fclose(r2); }
        FILE *r3 = fopen("d_moe.bin","wb"); if (r3) { fwrite(moeout,sizeof(float),(size_t)n_tok*d,r3); fclose(r3); }
    }

    /* 9. post-mlp conv, then residual + h. */
    float *pm = in2_tensor_to_f32(ctx, &L->ffn_post_conv, NULL);
    float *mo = (float *)calloc((size_t)n_tok * d, sizeof(float));
    if (pm && mo && rc == WUBU_INKLING2_OK) {
        in2_shortconv(moeout, n_tok, d, pm, 4, mo);
        for (size_t i = 0; i < (size_t)n_tok * d; ++i) h[i] = res[i] + mo[i];
    } else rc = WUBU_INKLING2_ERR_IO;

    free(an); free(xn); free(Wq); free(qcl); free(q);
    free(Wk); free(Wv); free(kc); free(vc); free(kraw); free(vraw); free(kt); free(vt); free(k); free(v);
    free(Wrel); free(rcl); free(P); free(Wsk); free(rs); free(bias); free(sink);
    free(attn_out); free(alpha); free(logits); free(Wg); free(Wo); free(gate); free(ao);
    free(pc); free(res); free(fn); free(moein); free(moeout); free(pm); free(mo);
    return rc;
}

/* MoE over n_tok tokens: route, then accumulate the selected experts. */
int in2_moe(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
            int il, const float *x, int n_tok, float *out)
{
    const wubu_in2_layer_t *L = &m->layer[il];
    const int d = cfg->d_model, NE = cfg->n_expert, TOPK = cfg->n_expert_used, DFF = cfg->d_ff_expert;
    float *Wr = in2_tensor_to_f32(ctx, &L->router, NULL);
    float *cb = in2_tensor_to_f32(ctx, &L->exp_bias, NULL);
    float *gs = in2_tensor_to_f32(ctx, &L->gate_up_scale, NULL);
    float *ds = in2_tensor_to_f32(ctx, &L->down_scale, NULL);
    if (!Wr) return WUBU_INKLING2_ERR_IO;

    float *gu = (float *)malloc((size_t)2 * DFF * d * sizeof(float));
    float *dn = (float *)malloc((size_t)d * DFF * sizeof(float));
    float *g  = (float *)malloc((size_t)2 * DFF * sizeof(float));
    float *act = (float *)malloc((size_t)DFF * sizeof(float));
    float *eo = (float *)malloc((size_t)d * sizeof(float));
    int rc = (gu && dn && g && act && eo) ? WUBU_INKLING2_OK : WUBU_INKLING2_ERR_IO;
    if (getenv("IN2_DEBUG"))
        fprintf(stderr, "[moe L%d] d=%d NE=%d TOPK=%d DFF=%d scale=%.3f bufs=%p/%p/%p/%p/%p rc=%d\n",
                il, d, NE, TOPK, DFF, (double)cfg->expert_weights_scale, (void*)gu, (void*)dn, (void*)g,
                (void*)act, (void*)eo, rc);
    if (rc != WUBU_INKLING2_OK) { free(Wr); free(cb); free(gs); free(ds); free(gu); free(dn); free(g); free(act); free(eo); return rc; }

    for (int t = 0; t < n_tok && rc == WUBU_INKLING2_OK; ++t) {
        int idx[256]; float w[256];
        const int got = in2_router(x + (size_t)t * d, Wr, cb, NE, d, TOPK, cfg->expert_weights_scale, idx, w);
        for (int i = 0; i < d; ++i) out[(size_t)t * d + i] = 0.0f;
        for (int k = 0; k < got; ++k) {
            const int e = idx[k];
            const float gsc = gs ? gs[e] : 1.0f, dsc = ds ? ds[e] : 1.0f;
            const int r1 = in2_dequant_slice(ctx, &L->gate_up, (int64_t)e * 2 * DFF * d, (int64_t)2 * DFF * d, gu);
            const int r2 = in2_dequant_slice(ctx, &L->down,    (int64_t)e * d * DFF,    (int64_t)d * DFF,    dn);
            if (r1 != WUBU_INKLING2_OK || r2 != WUBU_INKLING2_OK) {
                if (getenv("IN2_DEBUG")) fprintf(stderr, "[moe L%d] dequant e%d rc %d/%d  types %d/%d\n", il, e, r1, r2,
                        L->gate_up.info ? L->gate_up.info->ggml_type : -1, L->down.info ? L->down.info->ggml_type : -1);
                rc = WUBU_INKLING2_ERR_IO; break;
            }
            in2_matvec(gu, x + (size_t)t * d, d, 2 * DFF, g);
            for (int i = 0; i < 2 * DFF; ++i) g[i] *= gsc;
            in2_expert_act(g, g + DFF, DFF, act);
            in2_matvec(dn, act, DFF, d, eo);
            for (int i = 0; i < d; ++i) out[(size_t)t * d + i] += w[k] * eo[i] * dsc;
        }
    }
    free(Wr); free(cb); free(gs); free(ds); free(gu); free(dn); free(g); free(act); free(eo);
    return rc;
}

/* Dequantise a whole tensor into f32. Caller frees. Returns NULL on failure. */
float *in2_tensor_to_f32(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t *n_out)
{
    if (!ctx || !t || !t->info) return NULL;
    int64_t ne = 1;
    for (int d = 0; d < t->info->n_dims; ++d) ne *= t->info->dims[d];
    float *buf = (float *)malloc((size_t)ne * sizeof(float));
    if (!buf) return NULL;
    if (wubu_inkling2_read_f32(ctx, t, buf, ne) != WUBU_INKLING2_OK) { free(buf); return NULL; }
    if (n_out) *n_out = ne;
    return buf;
}

/* Token embedding lookup: token_embd is (n_embd, n_vocab) so token t occupies columns t.
 * Rows are the embedding axis, which is dims[0], so the gather is strided over the vocab axis. */
int in2_embed(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
              int token, float *out)
{
    if (!m->tok_embd.info) return WUBU_INKLING2_ERR_MISSING_TENSOR;
    const gguf_tensor_info *ti = m->tok_embd.info;
    if (ti->n_dims != 2 || ti->dims[0] != (int64_t)cfg->d_model) return WUBU_INKLING2_ERR_SHAPE;
    if (token < 0 || token >= cfg->vocab_size) return WUBU_INKLING2_ERR_SHAPE;

    /* The quantised rows are per-token columns; the cheapest correct path is to dequantise the whole
     * tensor once. A streaming reader will replace this, but correctness first. */
    int64_t ne = 0;
    float *emb = in2_tensor_to_f32(ctx, &m->tok_embd, &ne);
    if (!emb) return WUBU_INKLING2_ERR_IO;
    const int64_t stride = ti->dims[0];
    for (int64_t i = 0; i < stride; ++i) out[i] = emb[(int64_t)token * stride + i];
    free(emb);
    return WUBU_INKLING2_OK;
}
