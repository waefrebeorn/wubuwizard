/* lfm2_attn.c -- LFM2.5 GQA attention + RoPE + KV cache (C11, self-contained).
 * SPDX-License-Identifier: WaefreBeorn-UMV3 */
#include "lfm2_attn.h"
#include "lfm2_math.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void rope(float *vec, int hd, int pos, float theta) {
    for (int i = 0; i < hd; i += 2) {
        float freq = powf(theta, -2.0f * i / hd);
        float ang = pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float a = vec[i], b = vec[i + 1];
        vec[i] = a * c - b * s;
        vec[i + 1] = a * s + b * c;
    }
}

void lfm2_gqa(const float *q_proj, const float *k_proj, const float *v_proj,
               const float *o_proj, const float *q_ln, const float *k_ln,
               int nq, int nkv, int hd, int d,
               float rope_theta, const float *x, int T,
               float *kv_cache_layer, int kv_max_t, int start_pos,
               float *attn_out) {
    int kv_dim = nkv * hd;
    int q_dim = nq * hd;   /* NOTE: q_dim may differ from d (MiniCPM5: 2048 vs 1536) */
    int Ttot = start_pos + T;
    float *q = (float *)malloc((size_t)T * q_dim * sizeof(float));
    float *k = (float *)malloc((size_t)T * kv_dim * sizeof(float));
    float *v = (float *)malloc((size_t)T * kv_dim * sizeof(float));
    lfm2_matmul_f32(x, q_proj, T, d, q_dim, q);
    lfm2_matmul_f32(x, k_proj, T, d, kv_dim, k);
    lfm2_matmul_f32(x, v_proj, T, d, kv_dim, v);

    /* write new K/V into cache (layout: [kv_max_t, kv_dim] K block, then V block) */
    if (kv_cache_layer) {
        memcpy(kv_cache_layer + (size_t)start_pos * kv_dim, k, (size_t)T * kv_dim * sizeof(float));
        memcpy(kv_cache_layer + (size_t)kv_max_t * kv_dim + (size_t)start_pos * kv_dim,
               v, (size_t)T * kv_dim * sizeof(float));
    }

    /* q/k layernorm per head — OPTIONAL (MiniCPM5-class models have no
     * per-head norms; q_ln/k_ln NULL = skip, identity). */
    if (q_ln || k_ln) {
        for (int t = 0; t < T; t++) {
            for (int hh = 0; hh < nq; hh++)
                if (q_ln) lfm2_rmsnorm(q + (size_t)t * q_dim + hh * hd, q_ln, hd, 1e-5f);
            for (int hh = 0; hh < nkv; hh++)
                if (k_ln) lfm2_rmsnorm(k + (size_t)t * kv_dim + hh * hd, k_ln, hd, 1e-5f);
        }
    }

    /* RoPE on q,k at absolute positions [start_pos, start_pos+T) */
    for (int t = 0; t < T; t++) {
        int pos = start_pos + t;
        for (int hh = 0; hh < nq; hh++) rope(q + (size_t)t * q_dim + hh * hd, hd, pos, rope_theta);
        for (int hh = 0; hh < nkv; hh++) rope(k + (size_t)t * kv_dim + hh * hd, hd, pos, rope_theta);
    }

    /* attention output is q_dim-wide per token (nq*hd), then o_proj
     * maps q_dim -> d. LFM2.5 had q_dim==d; MiniCPM5 q_dim=2048 > d=1536. */
    float *out = (float *)malloc((size_t)T * q_dim * sizeof(float));
    memset(out, 0, (size_t)T * q_dim * sizeof(float));
    const float scale = 1.0f / sqrtf((float)hd);
    int q_per_kv = nq / nkv;

    for (int t = 0; t < T; t++) {
        for (int hh = 0; hh < nq; hh++) {
            int kvh = hh / q_per_kv;
            const float *Q = q + (size_t)t * q_dim + hh * hd;
            float *O = out + (size_t)t * q_dim + hh * hd;
            float maxs = -1e30f;
            float *scores = (float *)malloc(Ttot * sizeof(float));
            for (int tp = 0; tp < Ttot; tp++) {
                const float *K = kv_cache_layer
                    ? kv_cache_layer + (size_t)tp * kv_dim + kvh * hd
                    : k + (size_t)tp * kv_dim + kvh * hd;
                float s = 0.0f;
                for (int i = 0; i < hd; i++) s += Q[i] * K[i];
                s *= scale;
                scores[tp] = s;
                if (s > maxs) maxs = s;
            }
            float sum = 0.0f;
            for (int tp = 0; tp < Ttot; tp++) { scores[tp] = expf(scores[tp] - maxs); sum += scores[tp]; }
            float inv = sum > 0 ? 1.0f / sum : 0.0f;
            for (int tp = 0; tp < Ttot; tp++) {
                const float *V = kv_cache_layer
                    ? kv_cache_layer + (size_t)kv_max_t * kv_dim + (size_t)tp * kv_dim + kvh * hd
                    : v + (size_t)tp * kv_dim + kvh * hd;
                float wv = scores[tp] * inv;
                for (int i = 0; i < hd; i++) O[i] += wv * V[i];
            }
            free(scores);
        }
    }

    lfm2_matmul_f32(out, o_proj, T, q_dim, d, attn_out);
    free(q); free(k); free(v); free(out);
}
