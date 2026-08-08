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

extern int lfm2_dump_layer;
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
        lfm2_dump_layer = l;
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
            lfm2_rmsnorm(tmp + (size_t)t * d, L->op_norm, d, m->rms_eps);
        }
        if (getenv("LFM2_DBGLAYER")) {
            int dl = atoi(getenv("LFM2_DBGLAYER"));
            if (l == dl) {
                const float *hp = h + (size_t)(T - 1) * d;
                const float *np = tmp + (size_t)(T - 1) * d;
                double hs = 0, ns = 0;
                for (int q = 0; q < d; q++) { hs += (double)hp[q] * hp[q]; ns += (double)np[q] * np[q]; }
                fprintf(stderr, "DBG L%d pre-norm rms=%.4f op_norm_out rms=%.4f op_norm[0..3]=%.4g %.4g %.4g %.4g\n",
                        l, sqrt(hs / d), sqrt(ns / d),
                        L->op_norm[0], L->op_norm[1], L->op_norm[2], L->op_norm[3]);
            }
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

        if (getenv("LFM2_DBGLAYER")) {
            int dl = atoi(getenv("LFM2_DBGLAYER"));
            if (l == dl) {
                double as = 0;
                for (size_t q = 0; q < (size_t)T * d; q++) as += (double)scratch[q] * scratch[q];
                fprintf(stderr, "DBG L%d ATTN scratch rms=%.4f\n", l, sqrt(as / ((size_t)T * d)));
            }
        }

        /* ffn path: tmp = ffn_norm(h); ffn -> scratch; residual add */
        for (int t = 0; t < T; t++) {
            memcpy(tmp + (size_t)t * d, h + (size_t)t * d, d * sizeof(float));
            lfm2_rmsnorm(tmp + (size_t)t * d, L->ffn_norm, d, m->rms_eps);
        }
        lfm2_ffn(L->w1, L->w2, L->w3, m->ff_dim, d, tmp, T, scratch);
        if (getenv("LFM2_DBGLAYER")) {
            int dl = atoi(getenv("LFM2_DBGLAYER"));
            if (l == dl) {
                /* dump FFN inputs + intermediate for numpy cross-check */
                FILE *fp = fopen("/tmp/lfm2_ffn_in.bin", "wb");
                if (fp) { fwrite(tmp, sizeof(float), (size_t)T * d, fp); fclose(fp); }
                fp = fopen("/tmp/lfm2_w1.bin", "wb");
                if (fp) { fwrite(L->w1, sizeof(float), (size_t)m->ff_dim * d, fp); fclose(fp); }
                fp = fopen("/tmp/lfm2_w2.bin", "wb");
                if (fp) { fwrite(L->w2, sizeof(float), (size_t)d * m->ff_dim, fp); fclose(fp); }
                fp = fopen("/tmp/lfm2_w3.bin", "wb");
                if (fp) { fwrite(L->w3, sizeof(float), (size_t)m->ff_dim * d, fp); fclose(fp); }
                fp = fopen("/tmp/lfm2_ffn_out.bin", "wb");
                if (fp) { fwrite(scratch, sizeof(float), (size_t)T * d, fp); fclose(fp); }
                fprintf(stderr, "DBG L%d FFN dumps written (in/w1/w2/w3/out)\n", l);
            }
        }
        if (getenv("LFM2_DBGLAYER")) {
            int dl = atoi(getenv("LFM2_DBGLAYER"));
            if (l == dl) {
                double fs = 0, tis = 0;
                for (size_t q = 0; q < (size_t)T * d; q++) fs += (double)scratch[q] * scratch[q];
                for (size_t q = 0; q < (size_t)T * d; q++) tis += (double)tmp[q] * tmp[q];
                fprintf(stderr, "DBG L%d FFN in rms=%.4f out rms=%.4f (ff_norm[0..3]=%.4g %.4g %.4g %.4g)\n",
                        l, sqrt(tis / ((size_t)T * d)), sqrt(fs / ((size_t)T * d)),
                        L->ffn_norm[0], L->ffn_norm[1], L->ffn_norm[2], L->ffn_norm[3]);
                fprintf(stderr, "DBG L%d w1[0..3]=%.6g %.6g %.6g %.6g w2[0..3]=%.6g %.6g %.6g %.6g w3[0..3]=%.6g %.6g %.6g %.6g\n",
                        l, L->w1[0], L->w1[1], L->w1[2], L->w1[3],
                        L->w2[0], L->w2[1], L->w2[2], L->w2[3],
                        L->w3[0], L->w3[1], L->w3[2], L->w3[3]);
                fprintf(stderr, "DBG L%d w2 ne=%lld w2_type=%d (embed_off=%lld q_embed_off=%lld)\n",
                        l, (long long)L->q_w2_ne, L->q_w2_t,
                        (long long)(m->embed ? -1 : (L->q_w2 ? (L->q_w2 - m->q_embed) : 0)),
                        (long long)(m->q_embed ? (L->q_w2 ? (L->q_w2 - m->q_embed) : -1) : -2));
            }
        }
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
    lfm2_rmsnorm(h + (size_t)(T - 1) * d, m->embed_norm, d, m->rms_eps);
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
            lfm2_rmsnorm(hn + (size_t)t * d, m->embed_norm, d, m->rms_eps);
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
    } else if (m->q_lm_head) {
        /* separate untied quantized lm_head (output.weight) */
        lfm2_lmhead_q_from(h + (size_t)(T - 1) * d, m, logits,
                           m->q_lm_head, m->q_lm_head_type,
                           m->lm_head_bytes_per_row);
    } else if (m->q_embed) {
        /* quantized tied lm_head: dequantize the head row-block on demand */
        lfm2_lmhead_q_from(h + (size_t)(T - 1) * d, m, logits,
                           m->q_embed, m->q_embed_type,
                           m->embed_bytes_per_row);
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
    /* conv block */
    if (L->q_in_proj && !L->in_proj)
        dequant_layer_tensor(&L->in_proj, L->q_in_proj, L->q_in_proj_t,
                             L->q_in_proj_ne ? L->q_in_proj_ne : (int64_t)3 * conv_dim * d_model);
    if (L->q_conv_w && !L->conv_w)
        dequant_layer_tensor(&L->conv_w, L->q_conv_w, L->q_conv_w_t,
                             L->q_conv_w_ne ? L->q_conv_w_ne : (int64_t)conv_dim * L->conv_k);
    if (L->q_out_proj && !L->out_proj)
        dequant_layer_tensor(&L->out_proj, L->q_out_proj, L->q_out_proj_t,
                             L->q_out_proj_ne ? L->q_out_proj_ne : (int64_t)d_model * conv_dim);
    /* GQA block — exact per-tensor element counts captured at load time
     * (MiniCPM5 q_proj is [1536, 2048] not [d,d]; k/v are [d, 256]). */
    if (L->q_q_proj && !L->q_proj)
        dequant_layer_tensor(&L->q_proj, L->q_q_proj, L->q_q_proj_t,
                             L->q_q_proj_ne ? L->q_q_proj_ne : (int64_t)d_model * d_model);
    if (L->q_k_proj && !L->k_proj)
        dequant_layer_tensor(&L->k_proj, L->q_k_proj, L->q_k_proj_t,
                             L->q_k_proj_ne ? L->q_k_proj_ne : (int64_t)d_model * d_model / 4);
    if (L->q_v_proj && !L->v_proj)
        dequant_layer_tensor(&L->v_proj, L->q_v_proj, L->q_v_proj_t,
                             L->q_v_proj_ne ? L->q_v_proj_ne : (int64_t)d_model * d_model / 4);
    if (L->q_o_proj && !L->o_proj)
        dequant_layer_tensor(&L->o_proj, L->q_o_proj, L->q_o_proj_t,
                             L->q_o_proj_ne ? L->q_o_proj_ne : (int64_t)d_model * d_model);
    /* FFN */
    if (L->q_w1 && !L->w1)
        dequant_layer_tensor(&L->w1, L->q_w1, L->q_w1_t,
                             L->q_w1_ne ? L->q_w1_ne : (int64_t)ff_dim * d_model);
    if (L->q_w2 && !L->w2)
        dequant_layer_tensor(&L->w2, L->q_w2, L->q_w2_t,
                             L->q_w2_ne ? L->q_w2_ne : (int64_t)d_model * ff_dim);
    if (L->q_w3 && !L->w3)
        dequant_layer_tensor(&L->w3, L->q_w3, L->q_w3_t,
                             L->q_w3_ne ? L->q_w3_ne : (int64_t)ff_dim * d_model);
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
 * each vocab row on the fly from the GGUF blob. src may be the tied
 * embed or a separate untied output.weight (MiniCPM5). */
void lfm2_lmhead_q_from(const float *h, const lfm2_model_t *m, float *logits,
                        const uint8_t *src, int src_type, int bytes_per_row) {
    const int d = m->d_model;
    float *tmp = (float *)malloc((size_t)d * sizeof(float));
    if (!tmp) { for (int v = 0; v < m->vocab_size; v++) logits[v] = 0.0f; return; }
    for (int v = 0; v < m->vocab_size; v++) {
        const uint8_t *row = src + (size_t)v * bytes_per_row;
        gguf_dequantize(row, src_type, d, tmp);
        double s = 0.0;
        for (int i = 0; i < d; i++) s += (double)h[i] * tmp[i];
        logits[v] = (float)s;
    }
    free(tmp);
}
