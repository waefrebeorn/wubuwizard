/* lfm2_forward.c -- LFM2.5 forward orchestrator (C11, self-contained).
 * SPDX-License-Identifier: WaefreBeorn-UMV3 */
#include "lfm2_forward.h"
#include "lfm2_math.h"
#include "lfm2_conv.h"
#include "lfm2_attn.h"
#include "lfm2_ffn.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

bool lfm2_forward(const lfm2_model_t *m, const float *emb, int B, int T, float *logits) {
    if (B != 1) { fprintf(stderr, "lfm2: only B=1 supported\n"); return false; }
    int d = m->d_model;
    float *h = (float *)malloc((size_t)T * d * sizeof(float));
    memcpy(h, emb, (size_t)T * d * sizeof(float));
    /* HF Lfm2Model: embedding_norm is applied ONCE, AFTER all layers (not
     * before layer 0, not twice). Start the residual stream from raw embed. */
    float *scratch = (float *)malloc((size_t)T * d * sizeof(float));
    float *tmp = (float *)malloc((size_t)T * d * sizeof(float)); /* normalized input */

    for (int l = 0; l < m->n_layers; l++) {
        lfm2_layer_t *L = &m->layers[l];
        if (getenv("LFM2_NLAYERS")) {
            int nl = atoi(getenv("LFM2_NLAYERS"));
            if (l >= nl) break;
        }

        /* Lazy materialize: dequantize THIS layer's quantized GGUF weights
         * to F32, run, then release — peak RAM = one layer, not the model. */
        if (!lfm2_layer_materialize(L, d, m->conv_dim, m->ff_dim)) {
            fprintf(stderr, "lfm2: layer %d materialize failed\n", l);
            free(h); free(scratch); free(tmp);
            return false;
        }

        /* operator path: tmp = operator_norm(h); op -> scratch; residual add */
        for (int t = 0; t < T; t++) {
            memcpy(tmp + (size_t)t * d, h + (size_t)t * d, d * sizeof(float));
            lfm2_rmsnorm(tmp + (size_t)t * d, L->op_norm, d, 1e-5f);
        }
        if (m->is_conv[l]) {
            lfm2_conv(L->in_proj, L->conv_w, L->out_proj, L->conv_k,
                      m->conv_dim, d, tmp, T, scratch);
        } else {
            float *kvc = m->kv_cache + (size_t)l * 2 * m->n_kv_heads * m->head_dim * m->kv_max_t;
            lfm2_gqa(L->q_proj, L->k_proj, L->v_proj, L->o_proj, L->q_ln, L->k_ln,
                     m->n_q_heads, m->n_kv_heads, m->head_dim, d, m->rope_theta,
                     tmp, T, kvc, m->kv_max_t, 0 /* start_pos: fresh prefill */, scratch);
        }
        for (size_t i = 0; i < (size_t)T * d; i++) h[i] += scratch[i];

        /* ffn path: tmp = ffn_norm(h); ffn -> scratch; residual add */
        for (int t = 0; t < T; t++) {
            memcpy(tmp + (size_t)t * d, h + (size_t)t * d, d * sizeof(float));
            lfm2_rmsnorm(tmp + (size_t)t * d, L->ffn_norm, d, 1e-5f);
        }
        lfm2_ffn(L->w1, L->w2, L->w3, m->ff_dim, d, tmp, T, scratch);
        for (size_t i = 0; i < (size_t)T * d; i++) h[i] += scratch[i];

        if (getenv("LFM2_DEBUG")) {
            const float *hp = h + (size_t)(T - 1) * d;
            float ss = 0.0f; for (int q = 0; q < d; q++) ss += hp[q] * hp[q];
            fprintf(stderr, "L%d h_norm=%.4f\n", l, sqrtf(ss / d));
        }
        if (getenv("LFM2_LAYER_DUMP") && (l < 3 || l == 29)) {
            const float *hp = h + (size_t)(T - 1) * d;
            fprintf(stderr, "L%d:", l);
            for (int q = 0; q < 8; q++) fprintf(stderr, " %.7g", hp[q]);
            fprintf(stderr, "\n");
        }
        lfm2_layer_release(L);   /* free this layer's F32 — next layer rematerializes */
    }

    /* embedding_norm applied ONCE, after all layers (HF Lfm2Model) + tied lm_head */
    if (getenv("LFM2_HIDDEN")) {
        const float *hp = h + (size_t)(T - 1) * d;
        fprintf(stderr, "[hidden pre-norm] ");
        for (int q = 0; q < 8 && q < d; q++) fprintf(stderr, "%.7g ", hp[q]);
        double hs = 0;
        for (int q = 0; q < d; q++) hs += (double)hp[q] * hp[q];
        fprintf(stderr, "| rms=%.6f\n", sqrt(hs / d));
    }
    lfm2_rmsnorm(h + (size_t)(T - 1) * d, m->embed_norm, d, 1e-5f);
    if (getenv("LFM2_HIDDEN")) {
        const float *hp = h + (size_t)(T - 1) * d;
        fprintf(stderr, "[hidden post-norm] ");
        for (int q = 0; q < 8 && q < d; q++) fprintf(stderr, "%.7g ", hp[q]);
        fprintf(stderr, "\n");
    }
    if (getenv("LFM2_HIDDEN_ALL")) {
        /* apply the final norm to every position (llama.cpp result_norm
         * covers the whole sequence) and dump per-position stats */
        float *hn = (float *)malloc((size_t)T * d * sizeof(float));
        for (int t = 0; t < T; t++) {
            memcpy(hn + (size_t)t * d, h + (size_t)t * d, (size_t)d * sizeof(float));
            lfm2_rmsnorm(hn + (size_t)t * d, m->embed_norm, d, 1e-5f);
        }
        for (int p = 0; p < T; p++) {
            const float *hp = hn + (size_t)p * d;
            double ss = 0;
            for (int q = 0; q < d; q++) ss += (double)hp[q] * hp[q];
            fprintf(stderr, "pos%d rms=%.5f f4=[%.5g %.5g %.5g %.5g]\n", p,
                    sqrt(ss / d), hp[0], hp[1], hp[2], hp[3]);
        }
        free(hn);
    }
    if (m->embed) {
        lfm2_matmul_f32(h + (size_t)(T - 1) * d, m->embed, 1, d, m->vocab_size, logits);
    } else if (m->q_embed) {
        /* quantized tied lm_head: dequantize the head row-block on demand */
        lfm2_lmhead_q(h + (size_t)(T - 1) * d, m, logits);
    }
    free(h); free(scratch); free(tmp);
    return true;
}

/* ---- lazy per-layer materialization (quantized GGUF weights) ---- */

/* Dequantize one tensor into L->var (malloc'd F32). Returns 1 on success. */
static int dequant_layer_tensor(float **dst, const uint8_t *q, int qtype,
                                int64_t n_elems) {
    if (!q || !dst) return 0;
    float *f = (float *)malloc((size_t)n_elems * sizeof(float));
    if (!f) return 0;
    gguf_dequantize(q, qtype, n_elems, f);
    *dst = f;
    return 1;
}

bool lfm2_layer_materialize(lfm2_layer_t *L, int d_model, int conv_dim,
                            int ff_dim) {
    if (!L) return false;
    if (L->in_proj || L->q_proj || L->w1) return true; /* already materialized */
    int kv_dim = d_model / 4; /* LFM2.5 config inference: kv = d/4 (512 for 2048) */
    /* conv block */
    if (L->q_in_proj && !L->in_proj)
        dequant_layer_tensor(&L->in_proj, L->q_in_proj, L->q_in_proj_t,
                             (int64_t)3 * conv_dim * d_model);
    if (L->q_conv_w && !L->conv_w)
        dequant_layer_tensor(&L->conv_w, L->q_conv_w, L->q_conv_w_t,
                             (int64_t)conv_dim * L->conv_k);
    if (L->q_out_proj && !L->out_proj)
        dequant_layer_tensor(&L->out_proj, L->q_out_proj, L->q_out_proj_t,
                             (int64_t)d_model * conv_dim);
    /* GQA block */
    if (L->q_q_proj && !L->q_proj)
        dequant_layer_tensor(&L->q_proj, L->q_q_proj, L->q_q_proj_t,
                             (int64_t)d_model * d_model);
    if (L->q_k_proj && !L->k_proj)
        dequant_layer_tensor(&L->k_proj, L->q_k_proj, L->q_k_proj_t,
                             (int64_t)kv_dim * d_model);
    if (L->q_v_proj && !L->v_proj)
        dequant_layer_tensor(&L->v_proj, L->q_v_proj, L->q_v_proj_t,
                             (int64_t)kv_dim * d_model);
    if (L->q_o_proj && !L->o_proj)
        dequant_layer_tensor(&L->o_proj, L->q_o_proj, L->q_o_proj_t,
                             (int64_t)d_model * d_model);
    /* FFN */
    if (L->q_w1 && !L->w1)
        dequant_layer_tensor(&L->w1, L->q_w1, L->q_w1_t, (int64_t)ff_dim * d_model);
    if (L->q_w2 && !L->w2)
        dequant_layer_tensor(&L->w2, L->q_w2, L->q_w2_t, (int64_t)d_model * ff_dim);
    if (L->q_w3 && !L->w3)
        dequant_layer_tensor(&L->w3, L->q_w3, L->q_w3_t, (int64_t)ff_dim * d_model);
    return (L->in_proj || L->q_proj || L->w1) ? true : false;
}

void lfm2_layer_release(lfm2_layer_t *L) {
    if (!L) return;
    free(L->in_proj);  L->in_proj  = NULL;
    free(L->conv_w);   L->conv_w   = NULL;
    free(L->out_proj); L->out_proj = NULL;
    free(L->q_proj);   L->q_proj   = NULL;
    free(L->k_proj);   L->k_proj   = NULL;
    free(L->v_proj);   L->v_proj   = NULL;
    free(L->o_proj);   L->o_proj   = NULL;
    free(L->w1);       L->w1       = NULL;
    free(L->w2);       L->w2       = NULL;
    free(L->w3);       L->w3       = NULL;
    /* norms are tiny and loaded F32 — keep them (freed in lfm2_free) */
}

/* Quantized lm_head: logits[v] = dot(h, row_v) for all v, dequantizing
 * each vocab row on the fly from the GGUF blob. */
void lfm2_lmhead_q(const float *h, const lfm2_model_t *m, float *logits) {
    const int d = m->d_model;
    float *tmp = (float *)malloc((size_t)d * sizeof(float));
    if (!tmp) { for (int v = 0; v < m->vocab_size; v++) logits[v] = 0.0f; return; }
    for (int v = 0; v < m->vocab_size; v++) {
        const uint8_t *row = m->q_embed + (size_t)v * m->embed_bytes_per_row;
        gguf_dequantize(row, m->q_embed_type, d, tmp);
        double s = 0.0;
        for (int i = 0; i < d; i++) s += (double)h[i] * tmp[i];
        logits[v] = (float)s;
    }
    free(tmp);
}
