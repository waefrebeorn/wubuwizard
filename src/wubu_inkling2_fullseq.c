/* wubu_inkling2_fullseq.c -- the full-attention layer over a token sequence.
 *
 * Same maths as in2_layer_full but generalised to n_tok with causal masking, so prefill and generation
 * share one code path. Differences from the sliding layer: three depth-mixed states with their own input
 * norms, tau-scaled q, dynamic k/v short convs, and attention over every earlier key (the relative bias
 * still dies outside 0..1023).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_inkling2.h"
#include "wubu_inkling2_kv.h"

void in2_rmsnorm_unweighted(const float *x, int n, float eps, float *out);
void in2_rmsnorm(const float *x, const float *w, int n, float eps, float *out);
void in2_matvec(const float *W, const float *x, int n_in, int n_out, float *y);
void in2_shortconv(const float *x, int n_tok, int channels, const float *w, int k, float *out);
void in2_shortconv_dynamic(const float *x, int n_tok, int channels, const float *stat,
                           const float *dyn_proj, const float *dyn_input, int hidden,
                           int n_head, int k, float *out);
void in2_l2_clip(float *x, const float *scale, int n_head, int dim, float threshold);
void in2_rel_bias_table(const float *rel_states, const float *P, int n_head, int d_rel, int extent, float *bias);
void in2_mix(const float *blocks, int n_blocks, int block_stride, int pos_off,
             const float *partial, const float *queries, int n_q, int d, float eps, float *out);
void in2_message_gate(const float *h, const float *message, const float *down, const float *forget,
                      const float *msgp, int d, int rank, float eps, float *out);
int  in2_moe(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
             int il, const float *x, int n_tok, float *out);
float *in2_tensor_to_f32(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t *n_out);

int in2_matvec_nt(gguf_ctx *ctx, const wubu_in2_tensor_t *t,
                  const float *X, int nt, int n_in, int n_out, float *Y);
const float *in2_tensor_borrow(gguf_ctx *, const wubu_in2_tensor_t *);

int  in2_layer_full_seq(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                       int il, float *h, int n_tok, int pos0, const float *blocks, int n_blocks,
                       const float *msgs)
{
    const wubu_in2_layer_t *L = &m->layer[il];
    const int d = cfg->d_model, NH = cfg->n_head, NKV = cfg->n_kv_head, HD = cfg->head_dim;
    const int qdim = NH * HD, kdim = NKV * HD, DREL = 16, EXT = 1024;
    int vvis = 0, kv_pos0 = pos0;
    in2_kv_t *kv = in2_kv_current(&vvis, &kv_pos0);
    if (!kv) kv_pos0 = pos0;
    if (kv && getenv("IN2_KV_TRACE")) fprintf(stderr, "FULL il=%d visit=%d pos0=%d n_tok=%d\n", il, vvis, kv_pos0, n_tok);
    const float eps = cfg->rms_eps;
    const size_t ND = (size_t)n_tok * d;
    extern double in2_wall_now(void);
    extern double in2_fph[8];
    double _fph = in2_wall_now();
    #define FP2(i) do { const double _n = in2_wall_now(); in2_fph[i] += _n - _fph; _fph = _n; } while (0)

    /* Borrowed rather than copied: in2_mix takes const float*, so the dequant cache's own buffer is usable
     * directly and the pointer must not be freed here. MEASURED SIZE: this tensor is 2304 floats (9 KB), not
     * the 2304x2304 matrix an earlier comment claimed - reading the GGUF header shows dims=[2304] type=0 for
     * blk.N.attn_res_query.weight. The change is therefore neutral in time (~3.00 vs 3.03 tok/s); it is kept
     * only because it removes the per-visit allocation. */
    const float *arq = in2_tensor_borrow(ctx, &L->attn_res_q);
    float *qn = in2_tensor_to_f32(ctx, &L->q_inp_norm, NULL);
    float *kn = in2_tensor_to_f32(ctx, &L->k_inp_norm, NULL);
    float *vn = in2_tensor_to_f32(ctx, &L->v_inp_norm, NULL);
    if (!arq || !qn || !kn || !vn) {
        free(qn); free(kn); free(vn);
        return WUBU_INKLING2_ERR_IO;
    }

    FP2(0); /* loads */
    /* 1. three depth-mixed states per position (q, k, v inputs), then the message gates. */
    float *state = (float *)calloc((size_t)3 * ND, sizeof(float));
    if (!state) { free(qn); free(kn); free(vn); return WUBU_INKLING2_ERR_IO; }
    /* boundary: exclude the entry this layer's caller already appended (see the sliding layer) */
    const int n_attn = (cfg->attn_res_block > 0 && il % cfg->attn_res_block == 0)
                     ? n_blocks - 1 : n_blocks;
    for (int t = 0; t < n_tok; ++t) {
        in2_mix(blocks, n_attn, n_tok * d, t * d, h + (size_t)t * d, arq, 3, d, eps,
                state + (size_t)t * 3 * d);
    }
    if (il == 5 && getenv("IN2_DUMP_FULLMIX")) {
        char mp[512]; snprintf(mp, sizeof mp, "%s/L%02d_mix.bin", getenv("IN2_DUMP_FULLMIX"), il);
        FILE *mf = fopen(mp, "wb");
        if (mf) { fwrite(state, sizeof(float), (size_t)n_tok * 3 * d, mf); fclose(mf); }
    }

    for (int sl = 0; sl < 3 && msgs; ++sl) {
        /* Borrowed: read-only message-gate weights. These were copied and freed per visit. */
        const float *dw = in2_tensor_borrow(ctx, &L->loop_down[sl]);
        const float *fw = in2_tensor_borrow(ctx, &L->loop_forget[sl]);
        const float *gw = in2_tensor_borrow(ctx, &L->loop_gain[sl]);
        if (dw && fw && gw) {
            for (int t = 0; t < n_tok; ++t) {
                float *dst = state + (size_t)t * 3 * d + (size_t)sl * d;
                float *gated = (float *)calloc((size_t)d, sizeof(float));
                in2_message_gate(dst, msgs + ((size_t)t * 4 + sl) * d, dw, fw, gw, d, cfg->loop_gate_rank, eps, gated);
                memcpy(dst, gated, sizeof(float) * d);
                free(gated);
            }
        }
        /* dw/fw/gw are borrowed from the dequant cache, not owned here */
    }

    float *xq = (float *)calloc(ND, sizeof(float));
    float *xk = (float *)calloc(ND, sizeof(float));
    float *xv = (float *)calloc(ND, sizeof(float));
    if (!xq || !xk || !xv) {
        free(qn); free(kn); free(vn); free(state); free(xq); free(xk); free(xv);
        return WUBU_INKLING2_ERR_IO;
    }
    for (int t = 0; t < n_tok; ++t) {
        const float *s = state + (size_t)t * 3 * d;
        in2_rmsnorm(s,         qn, d, eps, xq + (size_t)t * d);
        in2_rmsnorm(s + d,     kn, d, eps, xk + (size_t)t * d);
        in2_rmsnorm(s + 2 * d, vn, d, eps, xv + (size_t)t * d);
    }

    if (il == 5 && getenv("IN2_DUMP_FULLQ")) {
        /* the three depth-mixed states, their norms' outputs, and q - so the state->norm->q/k/v mapping
         * (an ordering assumption) can be checked rather than trusted */
        FILE *qf = fopen(getenv("IN2_DUMP_FULLQ"), "wb");
        if (qf) {
            fwrite(state, sizeof(float), 3 * (size_t)n_tok * d, qf);
            fwrite(xq, sizeof(float), (size_t)n_tok * d, qf);
            fwrite(xk, sizeof(float), (size_t)n_tok * d, qf);
            fwrite(xv, sizeof(float), (size_t)n_tok * d, qf);
            fclose(qf);
        }
    }

    FP2(1); /* mix+gates */
    /* 2. q with scaled clip and tau (tau is 1.0 at positions below 8192, so it only bites on long context) */
    const float *Wq = in2_tensor_borrow(ctx, &L->wq);   /* table-owned, never freed */
    float *qcl = in2_tensor_to_f32(ctx, &L->q_clip, NULL);
    float *q = (float *)calloc((size_t)n_tok * qdim, sizeof(float));
    if (!Wq || !q) { free(qn); free(kn); free(vn); free(state); free(xq); free(xk); free(xv);
                     free(qcl); free(q); return WUBU_INKLING2_ERR_IO; }
    for (int t = 0; t < n_tok; ++t) {
        float *qt = q + (size_t)t * qdim;
        in2_matvec(Wq, xq + (size_t)t * d, d, qdim, qt);
        in2_l2_clip(qt, qcl, NH, HD, cfg->qk_clip);
        /* tau = 1 + alpha * log(max(1, (pos+1)/floor)); reads 1.0 for every position below the floor.
         * It scales the relative-bias query as well as q (llama.cpp inkling2.cpp:357-358), which the
         * port previously missed - invisible at short context, wrong past the floor. */
        const float tau = 1.0f + cfg->log_scale_alpha *
            logf(fmaxf(((float)(pos0 + t) + 1.0f) / (float)cfg->log_scale_floor, 1.0f));
        for (int i = 0; i < qdim; ++i) qt[i] *= tau;
    }

    FP2(2); /* q */
    /* 3. k and v with the dynamic short conv */
    /* Borrowed, not copied. in2_tensor_to_f32 returns memory the caller owns, so every one of these paid a full
     * matrix memcpy per full-layer visit (wk/wv are 9.4 MB each) and the copies were used once and freed. The
     * borrowed pointer is the dequant cache's own buffer - identical floats, no copy - and must NOT be freed, which is
     * why the free list at the end of this function no longer mentions them. */
    const float *Wk = in2_tensor_borrow(ctx, &L->wk);
    const float *Wv = in2_tensor_borrow(ctx, &L->wv);
    const float *kc = in2_tensor_borrow(ctx, &L->k_conv);
    const float *vc = in2_tensor_borrow(ctx, &L->v_conv);
    const float *kp = in2_tensor_borrow(ctx, &L->k_conv_proj);
    const float *vp = in2_tensor_borrow(ctx, &L->v_conv_proj);
    float *kraw = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *vraw = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *kt = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *vt = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *k = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *v = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    int rc = WUBU_INKLING2_OK;
    if (!Wk || !Wv || !kc || !vc || !kp || !vp || !kraw || !vraw || !kt || !vt || !k || !v) rc = WUBU_INKLING2_ERR_IO;
    if (rc == WUBU_INKLING2_OK) {
        for (int t = 0; t < n_tok; ++t) {
            in2_matvec(Wk, xk + (size_t)t * d, d, kdim, kraw + (size_t)t * kdim);
            in2_matvec(Wv, xv + (size_t)t * d, d, kdim, vraw + (size_t)t * kdim);
        }
        if (kv) {
            /* Prepend the three cached raw rows as if they were part of this batch, run the tested conv over
             * history+new, keep only the new outputs. Placed HERE, before the loop that builds k[]/v[] from kt/vt -
             * an earlier attempt inserted this after that loop, so the rows were built from an unfilled buffer and
             * the run quietly used zeros (22.0042 instead of 26.9567). */
            const int H3 = 3, rows = H3 + n_tok;
            float *hx = (float *)calloc((size_t)rows * (size_t)d, sizeof(float));
            float *hk = (float *)malloc((size_t)rows * (size_t)kdim * sizeof(float));
            float *hv = (float *)malloc((size_t)rows * (size_t)kdim * sizeof(float));
            float *okb = (float *)malloc((size_t)rows * (size_t)kdim * sizeof(float));
            float *ovb = (float *)malloc((size_t)rows * (size_t)kdim * sizeof(float));
            if (hx && hk && hv && okb && ovb) {
                const float *hkh = in2_kv_hist_k(kv, vvis, 3);
                const float *hvh = in2_kv_hist_v(kv, vvis, 3);
                if (hkh) memcpy(hk, hkh, (size_t)H3 * (size_t)kdim * sizeof(float));
                if (hvh) memcpy(hv, hvh, (size_t)H3 * (size_t)kdim * sizeof(float));
                memcpy(hk + (size_t)H3 * kdim, kraw, (size_t)n_tok * (size_t)kdim * sizeof(float));
                memcpy(hv + (size_t)H3 * kdim, vraw, (size_t)n_tok * (size_t)kdim * sizeof(float));
                memcpy(hx + (size_t)H3 * d, xk, (size_t)n_tok * (size_t)d * sizeof(float));
                in2_shortconv_dynamic(hk, rows, kdim, kc, kp, hx, d, NKV, 4, okb);
                memcpy(kt, okb + (size_t)H3 * kdim, (size_t)n_tok * (size_t)kdim * sizeof(float));
                memcpy(hx + (size_t)H3 * d, xv, (size_t)n_tok * (size_t)d * sizeof(float));
                in2_shortconv_dynamic(hv, rows, kdim, vc, vp, hx, d, NKV, 4, ovb);
                memcpy(vt, ovb + (size_t)H3 * kdim, (size_t)n_tok * (size_t)kdim * sizeof(float));
                for (int t = 0; t < n_tok; ++t)
                    in2_kv_store_raw(kv, vvis, kraw + (size_t)t * kdim, vraw + (size_t)t * kdim);
            } else rc = WUBU_INKLING2_ERR_IO;
            free(hx); free(hk); free(hv); free(okb); free(ovb);
        } else {
            in2_shortconv_dynamic(kraw, n_tok, kdim, kc, kp, xk, d, NKV, 4, kt);
        }
        if (il == 5 && getenv("IN2_DUMP_DYNCONV")) {
            FILE *df = fopen(getenv("IN2_DUMP_DYNCONV"), "wb");
            if (df) {
                fwrite(kraw, sizeof(float), (size_t)n_tok * kdim, df);   /* pre-conv k */
                fwrite(kt,   sizeof(float), (size_t)n_tok * kdim, df);   /* post-conv k */
                fwrite(kc,   sizeof(float), 4 * (size_t)kdim, df);       /* static filter */
                fwrite(xk,   sizeof(float), (size_t)n_tok * d, df);      /* dynamic input */
                fwrite(kp,   sizeof(float), 32 * (size_t)d, df);         /* the [2304,32] projection */
                fclose(df);
            }
        }
        if (!kv) in2_shortconv_dynamic(vraw, n_tok, kdim, vc, vp, xv, d, NKV, 4, vt);
        for (int t = 0; t < n_tok; ++t) {
            memcpy(k + (size_t)t * kdim, kt + (size_t)t * kdim, sizeof(float) * kdim);
            in2_l2_clip(k + (size_t)t * kdim, NULL, NKV, HD, 22.6274166f);
            memcpy(v + (size_t)t * kdim, vt + (size_t)t * kdim, sizeof(float) * kdim);
            for (int hh = 0; hh < NKV; ++hh) {
                float *row = v + (size_t)t * kdim + hh * HD;
                in2_rmsnorm_unweighted(row, HD, eps, row);
            }
            /* Stored here, before attention: the read path serves these same rows, and the earlier attempt wrote V
             * only at the end of the layer, so the same visit read calloc'd zeros. */
            if (kv) {
                float *ck = in2_kv_k(kv, vvis, kv_pos0 + t);
                float *cv = in2_kv_v(kv, vvis, kv_pos0 + t);
                if (ck) memcpy(ck, k + (size_t)t * kdim, sizeof(float) * (size_t)kdim);
                if (cv) memcpy(cv, v + (size_t)t * kdim, sizeof(float) * (size_t)kdim);
                /* raw rows are pushed by the conv branch above, once each */
            }
        }
    }

    if (il == 5 && getenv("IN2_DUMP_FULLKV")) {
        FILE *df = fopen(getenv("IN2_DUMP_FULLKV"), "wb");
        if (df) {
            fwrite(q, sizeof(float), (size_t)n_tok * qdim, df);
            fwrite(k, sizeof(float), (size_t)n_tok * kdim, df);
            fwrite(v, sizeof(float), (size_t)n_tok * kdim, df);
            fclose(df);
        }
    }

    FP2(3); /* kv+conv */
    /* 4. attention: causal, over every earlier key (no window), with the per-query relative bias and sink */
    const float *Wrel = in2_tensor_borrow(ctx, &L->wrel);
    float *rcl = in2_tensor_to_f32(ctx, &L->rel_clip, NULL);
    float *P = in2_tensor_to_f32(ctx, &L->rel_bias, NULL);
    const float *Wsk = in2_tensor_borrow(ctx, &L->sink);
    float *rs = (float *)calloc((size_t)NH * DREL, sizeof(float));
    float *bias = (float *)calloc((size_t)NH * EXT, sizeof(float));
    float *sink = (float *)calloc((size_t)n_tok * NH, sizeof(float));
    /* sized for the visible prefix, not the batch: with a cache n_key = pos0 + t + 1 */
    float *logit = (float *)malloc(sizeof(float) * ((size_t)n_tok + (size_t)pos0 + 1u));
    float *alpha = (float *)malloc(sizeof(float) * ((size_t)n_tok + (size_t)pos0 + 1u));
    float *attn_out = (float *)calloc((size_t)n_tok * qdim, sizeof(float));
    if (!(Wrel && rcl && P && Wsk && rs && bias && sink && logit && alpha && attn_out)) rc = WUBU_INKLING2_ERR_IO;

    if (rc == WUBU_INKLING2_OK) {
        for (int t = 0; t < n_tok; ++t) {
            /* IN2_FULL_KERNEL: the int8 kernel path the sliding layers already use, instead of the fp32 float path.
             * Behind a flag because it changes top1 (149876 -> 220) when applied: the engine is inconsistent today -
             * sliding int8, full fp32 - and which contract it keeps is a decision that needs a text-level A/B, not a
             * one-token comparison. Default keeps the fp32 path so nothing silently changes. */
            if (getenv("IN2_FULL_KERNEL")) in2_matvec_nt(ctx, &L->wrel, xq + (size_t)t * d, 1, d, NH * DREL, rs);
            else                        in2_matvec(Wrel, xq + (size_t)t * d, d, NH * DREL, rs);
            in2_l2_clip(rs, rcl, NH, DREL, cfg->rel_clip);
            /* the relative-bias query carries the same log-position tau as q (inkling2.cpp:357-358).
             * No effect below the 8192 floor, which is why it survived every short-context check. */
            {
                const float tau = 1.0f + cfg->log_scale_alpha *
                    logf(fmaxf(((float)(pos0 + t) + 1.0f) / (float)cfg->log_scale_floor, 1.0f));
                if (tau != 1.0f) for (int i = 0; i < NH * DREL; ++i) rs[i] *= tau;
            }
            in2_rel_bias_table(rs, P, NH, DREL, EXT, bias);
            if (getenv("IN2_FULL_KERNEL")) in2_matvec_nt(ctx, &L->sink, xq + (size_t)t * d, 1, d, NH, sink + (size_t)t * NH);
            else                          in2_matvec(Wsk, xq + (size_t)t * d, d, NH, sink + (size_t)t * NH);
            for (int hh = 0; hh < NH; ++hh) {
                const int kvh = hh / (NH / NKV);
                int n_key = 0;
                const int gp = kv_pos0 + t;
                /* the cache is a ring over the window, so keys older than it are no longer resident */
                const int jstart = (g_kv_window > 0 && gp + 1 > g_kv_window) ? gp + 1 - g_kv_window : 0;
                for (int j = jstart; j <= gp; ++j) {      /* causal, global positions */
                    const int dist = gp - j;
                    const float *kbp;
                    if (kv) { kbp = in2_kv_k(kv, vvis, j); if (!kbp) continue; kbp += kvh * HD; }
                    else    { kbp = k + (size_t)(j - kv_pos0) * kdim + kvh * HD; }
                    double acc = 0.0;
                    for (int e = 0; e < HD; ++e) acc += (double)q[(size_t)t * qdim + hh * HD + e] * (double)kbp[e];
                    logit[n_key] = (float)acc * (1.0f / sqrtf((float)HD))
                                 + (dist < EXT ? bias[hh * EXT + dist] : 0.0f);
                    ++n_key;
                }
                /* sink-aware softmax: the weights sum to less than 1 by the sink's share */
                float mx = sink[(size_t)t * NH + hh];
                for (int i = 0; i < n_key; ++i) if (logit[i] > mx) mx = logit[i];
                double tot = exp((double)(sink[(size_t)t * NH + hh] - mx));
                for (int i = 0; i < n_key; ++i) { alpha[i] = expf(logit[i] - mx); tot += alpha[i]; }
                const float inv = (float)(1.0 / tot);
                for (int i = 0; i < n_key; ++i) alpha[i] *= inv;
                float *o = attn_out + (size_t)t * qdim + hh * HD;
                for (int j = 0; j < n_key; ++j) {
                    const float *vbp;
                    if (kv) { vbp = in2_kv_v(kv, vvis, j); if (!vbp) continue; vbp += kvh * HD; }
                    else    { vbp = v + (size_t)(j - kv_pos0) * kdim + kvh * HD; }
                    for (int e = 0; e < HD; ++e) o[e] += alpha[j] * vbp[e];
                }
            }
        }
    }

    FP2(4); /* attention */
    /* 5. gated output projection */
    const float *Wg = in2_tensor_borrow(ctx, &L->wgate);
    const float *Wo = in2_tensor_borrow(ctx, &L->wo);
    float *gate = (float *)calloc((size_t)n_tok * qdim, sizeof(float));
    float *ao = (float *)calloc(ND, sizeof(float));
    if (Wg && Wo && gate && ao && rc == WUBU_INKLING2_OK) {
        for (int t = 0; t < n_tok; ++t) {
            float *gt = gate + (size_t)t * qdim;
            if (getenv("IN2_FULL_KERNEL")) in2_matvec_nt(ctx, &L->wgate, xq + (size_t)t * d, 1, d, qdim, gt);
            else                          in2_matvec(Wg, xq + (size_t)t * d, d, qdim, gt);
            for (int i = 0; i < qdim; ++i)
                gt[i] = attn_out[(size_t)t * qdim + i] * (1.0f / (1.0f + expf(-gt[i])));
            if (getenv("IN2_FULL_KERNEL")) in2_matvec_nt(ctx, &L->wo, gt, 1, qdim, d, ao + (size_t)t * d);
            else                          in2_matvec(Wo, gt, qdim, d, ao + (size_t)t * d);
        }
    } else rc = WUBU_INKLING2_ERR_IO;

    if (il == 5 && getenv("IN2_DUMP_FULLATTN")) {
        FILE *af = fopen(getenv("IN2_DUMP_FULLATTN"), "wb");
        if (af) {
            fwrite(attn_out, sizeof(float), (size_t)n_tok * qdim, af);   /* pre-gate attention out */
            fwrite(ao, sizeof(float), (size_t)n_tok * d, af);            /* post attn_output projection */
            fwrite(bias, sizeof(float), (size_t)NH * EXT, af);
            fwrite(sink, sizeof(float), (size_t)n_tok * NH, af);
            fclose(af);
        }
    }

    FP2(5); /* gate_after */
    /* 6. post-attn conv + the boundary rule */
    float *pc = in2_tensor_to_f32(ctx, &L->attn_post_conv, NULL);
    float *res = (float *)calloc(ND, sizeof(float));
    if (pc && res && rc == WUBU_INKLING2_OK) {
        in2_conv_stream_res(vvis, ao, n_tok, d, pc, 4, res);
        const int boundary = (cfg->attn_res_block > 0) && (il % cfg->attn_res_block == 0);
        if (!boundary) for (size_t i = 0; i < ND; ++i) res[i] = h[i] + res[i];
    } else rc = WUBU_INKLING2_ERR_IO;

    /* 7. mlp_input mix (+ mlp message gate), ffn norm, MoE, post-mlp conv, residual */
    float *fq = in2_tensor_to_f32(ctx, &L->ffn_res_q, NULL);
    float *mixed = (float *)calloc(ND, sizeof(float));
    float *fn = in2_tensor_to_f32(ctx, &L->ffn_norm, NULL);
    float *moein = (float *)calloc(ND, sizeof(float));
    float *moeout = (float *)calloc(ND, sizeof(float));
    if (fq && mixed && fn && moein && moeout && rc == WUBU_INKLING2_OK) {
        for (int t = 0; t < n_tok; ++t) {
            in2_mix(blocks, n_blocks, n_tok * d, t * d, res + (size_t)t * d, fq, 1, d, eps,
                    mixed + (size_t)t * d);
            if (msgs) {
                const float *dw = in2_tensor_borrow(ctx, &L->loop_down[3]);
                const float *fw = in2_tensor_borrow(ctx, &L->loop_forget[3]);
                const float *gw = in2_tensor_borrow(ctx, &L->loop_gain[3]);
                if (dw && fw && gw) {
                    float *gated = (float *)calloc((size_t)d, sizeof(float));
                    in2_message_gate(mixed + (size_t)t * d, msgs + ((size_t)t * 4 + 3) * d, dw, fw, gw,
                                     d, cfg->loop_gate_rank, eps, gated);
                    memcpy(mixed + (size_t)t * d, gated, sizeof(float) * d);
                    free(gated);
                }
                /* dw/fw/gw are borrowed from the dequant cache, not owned here */
            }
            in2_rmsnorm(mixed + (size_t)t * d, fn, d, eps, moein + (size_t)t * d);
        }
        rc = in2_moe(ctx, m, cfg, il, moein, n_tok, moeout);
    } else rc = WUBU_INKLING2_ERR_IO;

    float *pm = in2_tensor_to_f32(ctx, &L->ffn_post_conv, NULL);
    float *mo = (float *)calloc(ND, sizeof(float));
    if (pm && mo && rc == WUBU_INKLING2_OK) {
        const char *fl = getenv("IN2_DUMPX_LAYER");
        if (getenv("IN2_DUMPX") && n_tok == 1 && fl && atoi(fl) == il) {
            FILE *f1 = fopen("d_fres.bin","wb"); if (f1) { fwrite(res,sizeof(float),(size_t)ND,f1); fclose(f1); }
            FILE *f2 = fopen("d_fmoe.bin","wb"); if (f2) { fwrite(moeout,sizeof(float),(size_t)ND,f2); fclose(f2); }
        }
        in2_conv_stream_moe(vvis, moeout, n_tok, d, pm, 4, mo);
        for (size_t i = 0; i < ND; ++i) h[i] = res[i] + mo[i];
    } else rc = WUBU_INKLING2_ERR_IO;

    free(qn); free(kn); free(vn); free(state); free(xq); free(xk); free(xv);
    free(qcl); free(q);   /* Wk/Wv/kc/vc/kp/vp are borrowed from the dequant cache, not owned here */
    free(kraw); free(vraw); free(kt); free(vt); free(k); free(v);
    free(rcl); free(P); free(rs); free(bias); free(sink);
    free(logit); free(alpha); free(attn_out); free(gate); free(ao);
    free(pc); free(res); free(fq); free(mixed); free(fn); free(moein); free(moeout); free(pm); free(mo);
    if (kv) in2_kv_advance();
    FP2(6); /* mlp+moe+postconv */
    return rc;
}
