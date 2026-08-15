/* wubu_megakernel.c — Photon 2.0 fused decode megakernel (C11, opaque, minimal)
 *
 * The megakernel fuses: RMSNorm → QKV → GQA attention → residual → RMSNorm →
 * FFN (GELU) → residual, all in a single call. The PSO pattern pre-configures
 * the fused decode function at create time so the hot path is a single
 * function dispatch with no setup overhead.
 */
#define _POSIX_C_SOURCE 200809L
#include "wubu_megakernel.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_activations.h"

struct wubu_megakernel {
    wubu_megakernel_cfg_t cfg;
};

wubu_megakernel_t *wubu_megakernel_create(const wubu_megakernel_cfg_t *cfg) {
    if (!cfg || cfg->d_model <= 0 || cfg->n_heads <= 0 || cfg->d_head <= 0 ||
        cfg->n_kv_heads <= 0 || cfg->d_ff <= 0)
        return NULL;
    if (cfg->n_heads % cfg->n_kv_heads != 0) return NULL;
    wubu_megakernel_t *mk = (wubu_megakernel_t *)calloc(1, sizeof(*mk));
    if (!mk) return NULL;
    mk->cfg = *cfg;
    /* PSO pre-compilation: in a GPU backend, this is where we'd JIT-compile
     * the fused kernel. On CPU, we pre-compute the config and validate
     * the weight layout. The function pointer dispatch is the same. */
    return mk;
}

void wubu_megakernel_free(wubu_megakernel_t *mk) {
    if (!mk) return;
    free(mk);
}

/* gelu_fast: use wubu_activations.h */

/* Fused single-token decode — the Photon 2.0 megakernel hot path.
 * All sub-operations are inlined into one function call. */
int wubu_megakernel_decode(const wubu_megakernel_t *mk,
                            const float *ctx,
                            const float *qkv_weight,
                            const float *attn_weight,
                            const float *ffh_weight,
                            const float *ffo_weight,
                            const float *rms_norm1,
                            const float *rms_norm2,
                            float *kv_cache, int pos,
                            float *out) {
    if (!mk || !ctx || !qkv_weight || !attn_weight || !ffh_weight ||
        !ffo_weight || !rms_norm1 || !rms_norm2 || !kv_cache || !out)
        return -1;

    int d = mk->cfg.d_model;
    int n_heads = mk->cfg.n_heads;
    int n_kv = mk->cfg.n_kv_heads;
    int d_head = mk->cfg.d_head;
    int d_ff = mk->cfg.d_ff;
    float eps = mk->cfg.rms_epsilon;
    int heads_per_kv = n_heads / n_kv;

    float *tmp = (float *)malloc((size_t)d * sizeof(float));
    if (!tmp) return -1;

    /* Phase 1: RMSNorm on input (pre-attention) */
    float norm1_rms = 0;
    for (int i = 0; i < d; i++)
        norm1_rms += ctx[i] * ctx[i];
    norm1_rms = sqrtf(norm1_rms / d + eps);
    for (int i = 0; i < d; i++)
        tmp[i] = ctx[i] / norm1_rms * rms_norm1[i];

    /* Phase 2: QKV projection (fused matmul)
     * qkv_weight is [3*d_model, d_model] but K,V are d_kv = n_kv * d_head.
     * Layout: [Q: d_model*d_model | K: d_model*(n_kv*d_head) | V: d_model*(n_kv*d_head)] */
    int d_kv = n_kv * d_head;
    float *q = (float *)malloc((size_t)(d + 2 * d_kv) * sizeof(float));
    if (!q) { free(tmp); return -1; }
    float *k = q + d;
    float *v = k + d_kv;

    /* Q = tmp @ qkv_weight[0:d*d]  (column-major or row-major? Assume row-major) */
    for (int i = 0; i < d; i++) {
        float dot = 0;
        const float *row = qkv_weight + (size_t)i * d;
        for (int j = 0; j < d; j++)
            dot += tmp[j] * row[j];
        q[i] = dot;
    }
    /* K = tmp @ qkv_weight[d*d : d*d + d*d_kv] */
    const float *kw = qkv_weight + (size_t)d * d;
    for (int i = 0; i < d_kv; i++) {
        float dot = 0;
        const float *row = kw + (size_t)i * d;
        for (int j = 0; j < d; j++)
            dot += tmp[j] * row[j];
        k[i] = dot;
    }
    /* V = tmp @ qkv_weight[d*(d+d_kv) : ...] */
    const float *vw = kw + (size_t)d_kv * d;
    for (int i = 0; i < d_kv; i++) {
        float dot = 0;
        const float *row = vw + (size_t)i * d;
        for (int j = 0; j < d; j++)
            dot += tmp[j] * row[j];
        v[i] = dot;
    }

    /* Phase 3: Store K, V into KV cache
     * kv_cache layout: [n_kv_heads * d_head * max_pos] (head-major, contiguous)
     * pos is the current position; we write K and V at that position.
     * For single-token decode, we also need past KV (positions 0..pos-1). */
    /* Write K[pos] and V[pos] into the cache */
    for (int h = 0; h < n_kv; h++) {
        for (int j = 0; j < d_head; j++) {
            kv_cache[(size_t)h * d_head * (pos + 1) + j * (pos + 1) + pos] = k[h * d_head + j];
            kv_cache[(size_t)(n_kv + h) * d_head * (pos + 1) + j * (pos + 1) + pos] = v[h * d_head + j];
        }
    }

    /* Phase 3b: Causal GQA attention
     * For each query head h: compute dot(q[h], k[kv_head]) / sqrt(d_head)
     * for all positions 0..pos, softmax, weighted sum of V. */
    float *q_per_head = q;  /* q is [n_heads * d_head] */
    float *attn_out = (float *)calloc(d, sizeof(float));
    if (!attn_out) { free(tmp); free(q); return -1; }

    float inv_sqrt_d = 1.0f / sqrtf((float)d_head);
    int cache_stride = (pos + 1);

    for (int h = 0; h < n_heads; h++) {
        int kv_head = h / heads_per_kv;
        float *qh = q_per_head + (size_t)h * d_head;
        const float *kh = kv_cache + (size_t)kv_head * d_head * cache_stride;
        const float *vh = kv_cache + (size_t)(n_kv + kv_head) * d_head * cache_stride;

        /* Compute attention scores for positions 0..pos */
        float *scores = (float *)malloc((size_t)(pos + 1) * sizeof(float));
        if (!scores) { free(tmp); free(q); free(attn_out); return -1; }

        float max_score = -1e30f;
        for (int i = 0; i <= pos; i++) {
            float dot = 0;
            for (int j = 0; j < d_head; j++)
                dot += qh[j] * kh[j * cache_stride + i];
            dot *= inv_sqrt_d;
            scores[i] = dot;
            if (dot > max_score) max_score = dot;
        }

        /* Softmax */
        float sum = 0;
        for (int i = 0; i <= pos; i++) {
            scores[i] = expf(scores[i] - max_score);
            sum += scores[i];
        }
        if (sum < 1e-8f) sum = 1e-8f;
        for (int i = 0; i <= pos; i++)
            scores[i] /= sum;

        /* Weighted sum of values */
        float *oh = attn_out + (size_t)h * d_head;
        for (int j = 0; j < d_head; j++) {
            float val = 0;
            for (int i = 0; i <= pos; i++)
                val += scores[i] * vh[j * cache_stride + i];
            oh[j] = val;
        }
        free(scores);
    }

    /* Phase 4: Residual (ctx + attn_out projected through O)
     * First project: attn_out = attn_out @ attn_weight [d, d] */
    float *residual = (float *)malloc((size_t)d * sizeof(float));
    if (!residual) { free(tmp); free(q); free(attn_out); return -1; }
    memcpy(residual, ctx, (size_t)d * sizeof(float));

    /* attn_out = attn_out @ attn_weight (row-major: attn_weight is [d, d]) */
    for (int i = 0; i < d; i++) {
        float dot = 0;
        const float *row = attn_weight + (size_t)i * d;
        for (int j = 0; j < d; j++)
            dot += attn_out[j] * row[j];
        tmp[i] = dot;  /* reuse tmp as projected attn output */
    }

    /* Residual add 1 */
    for (int i = 0; i < d; i++)
        tmp[i] = residual[i] + tmp[i];

    /* Phase 5: RMSNorm (pre-FFN) */
    float norm2_rms = 0;
    for (int i = 0; i < d; i++)
        norm2_rms += tmp[i] * tmp[i];
    norm2_rms = sqrtf(norm2_rms / d + eps);
    for (int i = 0; i < d; i++)
        tmp[i] = tmp[i] / norm2_rms * rms_norm2[i];

    /* Phase 6: FFN (GELU activation)
     * ffh_weight: [d_ff, d], ffo_weight: [d, d_ff] */
    float *ffn = (float *)calloc(d_ff, sizeof(float));
    if (!ffn) { free(tmp); free(q); free(attn_out); free(residual); return -1; }

    /* ffn = tmp @ ffh_weight (up-projection) + GELU */
    for (int i = 0; i < d_ff; i++) {
        float dot = 0;
        const float *row = ffh_weight + (size_t)i * d;
        for (int j = 0; j < d; j++)
            dot += tmp[j] * row[j];
        ffn[i] = gelu_fast(dot);
    }

    /* Phase 7: Residual add 2
     * out = tmp(residual) + ffn @ ffo_weight (down-projection) */
    for (int i = 0; i < d; i++) {
        float dot = 0;
        const float *row = ffo_weight + (size_t)i * d_ff;
        for (int j = 0; j < d_ff; j++)
            dot += ffn[j] * row[j];
        out[i] = tmp[i] + dot;
    }

    free(tmp); free(q); free(attn_out); free(residual); free(ffn);
    return 0;
}
