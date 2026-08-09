/*
 * wubu_backprop.c -- the REAL backward pass + REAL Muon for the
 * WuBu seed (12 layers, 7 Q heads / 1 KV head GQA, 448-dim, 16384
 * vocab, 1228 FFN).
 *
 * This replaces the two broken pieces the audit found in the first
 * trainer:
 *   1. the REAL per-layer backward (chain rule through EVERY path:
 *      embedding -> 12 blocks (attention q/k/v/o/g + qk-norm + rope +
 *      softmax + gated residual + swiglu + selectors) -> final norm ->
 *      tied head). Each layer gets ITS OWN gradient, not a shared proxy.
 *   2. the REAL Muon: Nesterov momentum (0.95) -> Newton-Schulz 5
 *      orthogonalization (a=3.4445, b=-4.7750, c=2.0315, the paper's
 *      whole point) -> the Moonlight per-matrix scaled step.
 *      Matrices = Muon, embed/head/norms/selectors = AdamW (the
 *      confirmed reference split).
 *
 * Activation recording: bp->x_in[l] holds the residual input to layer
 * l and is left as the layer's REAL output (pre-blend). The selector
 * blend (which the next layer consumes) lives in bp->sel_out[l].
 * Every other buffer is captured per (layer, token) for the backward.
 *
 * Verified by tools/test_backprop.c with finite differences (the DA
 * doctrine: tests != correct, so we check the gradients numerically).
 */
#include "wubu_backprop.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* the GPU dispatch (the wubu_model.h pattern, same as wubu.c):
 * the trainer rides cuBLAS when the GPU is present and falls back to
 * the CPU loops otherwise. The symbols are weak so a CPU-only link
 * still works. */
#include "gpu_wubu.h"
#if defined(__GNUC__)
#define BP_WEAK __attribute__((weak))
#else
#define BP_WEAK
#endif
BP_WEAK int gpu_wubu_init(void);
BP_WEAK int gpu_wubu_ready(void);
BP_WEAK void gpu_wubu_mark_weights_dirty(void);
BP_WEAK int gpu_wubu_matmul(float *y, const float *w, const float *x,
                             int M, int N, int K);
BP_WEAK int gpu_wubu_matmul_tx(float *y, const float *a, const float *b,
                                int M, int N, int K);
BP_WEAK int gpu_wubu_matmul_nt(float *y, const float *w, const float *x,
                                int M, int N, int K);
BP_WEAK int gpu_wubu_ns5(float *X, int rows, int cols);
BP_WEAK int gpu_wubu_ns5_gram(float *X, int rows, int cols);
BP_WEAK int gpu_wubu_attn(float *out, const float *q, const float *k,
                           const float *v, int seq, int heads, int dim,
                           int local_win, int is_full);
BP_WEAK int gpu_wubu_attn_backward(float *dq, float *dk, float *dv,
                                    const float *q, const float *k,
                                    const float *v, const float *o,
                                    const float *dao, int seq, int heads,
                                    int dim, int local_win, int is_full);
static int g_gpu_tried = 0;
/* tiny matmuls cost more in upload/launch than they save -- the GPU
 * threshold from the hardware table (RTX 4050, 6GB) */
#define GPU_MIN_FLOP 1000000

#define D  WUBU_DIM
#define FF WUBU_FFN_DIM

static float *calloc_f(size_t n)
{
    return (float *)calloc(n ? n : 1, sizeof(float));
}

int wubu_bp_alloc(wubu_bp_t *bp, int max_seq)
{
    if (!bp || max_seq <= 0) return -1;
    memset(bp, 0, sizeof(*bp));
    int L = WUBU_LAYERS;
    size_t sd = (size_t)max_seq * D;
    bp->layers = L;
    bp->cap_seq = max_seq;
    bp->x_in      = calloc_f((size_t)L * sd);
    bp->emb_in    = calloc_f(sd);
    bp->attn_norm = calloc_f((size_t)L * sd);
    bp->q_pre     = calloc_f((size_t)L * sd);
    bp->k_pre     = calloc_f((size_t)L * (size_t)max_seq * 64);
    bp->q         = calloc_f((size_t)L * sd);
    bp->k         = calloc_f((size_t)L * (size_t)max_seq * 64);
    bp->v         = calloc_f((size_t)L * (size_t)max_seq * 64);
    bp->attn_out  = calloc_f((size_t)L * sd);
    bp->o_out     = calloc_f((size_t)L * sd);
    bp->g_val     = calloc_f((size_t)L * sd);
    bp->ffn_norm  = calloc_f((size_t)L * sd);
    bp->ffn_gate  = calloc_f((size_t)L * (size_t)max_seq * 2 * FF);
    bp->ffn_up    = calloc_f((size_t)L * (size_t)max_seq * FF);
    bp->ffn_out   = calloc_f((size_t)L * sd);
    bp->sel_out   = calloc_f((size_t)L * sd);
    bp->ckpt      = calloc_f(sd);
    bp->sel_w0    = calloc_f((size_t)L);
    bp->final_h   = calloc_f(sd);
    bp->logits    = calloc_f((size_t)max_seq * WUBU_VOCAB);
    bp->s_dq      = calloc_f(sd);
    bp->s_dk      = calloc_f((size_t)max_seq * 64);
    bp->s_dv      = calloc_f((size_t)max_seq * 64);
    bp->s_dao     = calloc_f(sd);
    bp->s_dfg     = calloc_f((size_t)max_seq * 2 * FF);
    bp->s_dfu     = calloc_f((size_t)max_seq * FF);
    bp->s_dfn     = calloc_f(sd);
    bp->s_dan     = calloc_f(sd);
    bp->s_dffn_out= calloc_f(sd);
    bp->s_do      = calloc_f(sd);
    bp->s_dg      = calloc_f(sd);
    bp->s_dx      = calloc_f(sd);
    bp->s_dxentry = calloc_f(sd);
    if (!bp->x_in || !bp->emb_in || !bp->attn_norm || !bp->q_pre ||
        !bp->k_pre || !bp->q || !bp->k || !bp->v || !bp->attn_out ||
        !bp->o_out || !bp->g_val || !bp->ffn_norm || !bp->ffn_gate ||
        !bp->ffn_up || !bp->ffn_out || !bp->sel_out || !bp->ckpt ||
        !bp->sel_w0 || !bp->final_h || !bp->logits || !bp->s_dq || !bp->s_dk ||
        !bp->s_dv || !bp->s_dao || !bp->s_dfg || !bp->s_dfu ||
        !bp->s_dfn || !bp->s_dan || !bp->s_dffn_out || !bp->s_do ||
        !bp->s_dg || !bp->s_dx || !bp->s_dxentry) {
        wubu_bp_free(bp);
        return -1;
    }
    return 0;
}

void wubu_bp_free(wubu_bp_t *bp)
{
    if (!bp) return;
    free(bp->x_in); free(bp->emb_in); free(bp->attn_norm);
    free(bp->q_pre); free(bp->k_pre); free(bp->q); free(bp->k);
    free(bp->v); free(bp->attn_out); free(bp->o_out); free(bp->g_val);
    free(bp->ffn_norm); free(bp->ffn_gate); free(bp->ffn_up);
    free(bp->ffn_out); free(bp->sel_out); free(bp->ckpt);
    free(bp->sel_w0); free(bp->final_h); free(bp->logits);
    free(bp->s_dq); free(bp->s_dk); free(bp->s_dv); free(bp->s_dao);
    free(bp->s_dfg); free(bp->s_dfu); free(bp->s_dfn); free(bp->s_dan);
    free(bp->s_dffn_out); free(bp->s_do); free(bp->s_dg);
    free(bp->s_dx); free(bp->s_dxentry);
    memset(bp, 0, sizeof(*bp));
}

/* ---- tiny math helpers (the reference's exact formulas) ---- */

static float rms_norm(float *out, const float *x, const float *w, int n)
{
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + WUBU_EPS);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
    return r;
}

static float silu(float v) { return v / (1.0f + expf(-v)); }
static float silu_deriv(float v)   /* v = the CLIPPED pre-activation */
{
    float s = 1.0f / (1.0f + expf(-v));
    return s * (1.0f + v * (1.0f - s));
}
static float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

/* out[s, o] = sum_i w[o, i] * x[s, i]  (w is [out, in] row-major).
 * Rides the cuBLAS path for the big products; CPU otherwise. */
static void mm(float *out, const float *w, const float *x,
               int out_n, int in_n, int seq)
{
    if (!g_gpu_tried) { g_gpu_tried = 1; if (gpu_wubu_init) gpu_wubu_init(); }
    if (gpu_wubu_ready && gpu_wubu_ready() && gpu_wubu_matmul &&
        (long)seq * out_n * in_n >= GPU_MIN_FLOP &&
        gpu_wubu_matmul(out, w, x, seq, out_n, in_n))
        return;
    for (int s = 0; s < seq; s++) {
        const float *xs = x + (size_t)s * in_n;
        float *os = out + (size_t)s * out_n;
        for (int o = 0; o < out_n; o++) {
            float acc = 0;
            const float *wr = w + (size_t)o * in_n;
            for (int i = 0; i < in_n; i++) acc += wr[i] * xs[i];
            os[o] = acc;
        }
    }
}

/* partial RoPE on the q rows: each 64-wide head row rotates its first
 * 32 channels at its own position (the released path, applied per
 * head). k is a single 64-wide row per position. */
static void rope_q(float *q, int seq, const float *cos_tbl,
                   const float *sin_tbl)
{
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < WUBU_HEADS; h++) {
            float *row = q + ((size_t)s * D + (size_t)h * 64);
            const float *c = cos_tbl + (size_t)s * WUBU_ROPE_DIM;
            const float *si = sin_tbl + (size_t)s * WUBU_ROPE_DIM;
            for (int i = 0; i < WUBU_ROPE_DIM / 2; i++) {
                float x0 = row[i], x1 = row[WUBU_ROPE_DIM / 2 + i];
                row[i] = x0 * c[i] - x1 * si[i];
                row[WUBU_ROPE_DIM / 2 + i] = x0 * si[i] + x1 * c[i];
            }
        }
    }
}
static void rope_k(float *k, int seq, const float *cos_tbl,
                   const float *sin_tbl)
{
    for (int s = 0; s < seq; s++) {
        float *row = k + (size_t)s * 64;
        const float *c = cos_tbl + (size_t)s * WUBU_ROPE_DIM;
        const float *si = sin_tbl + (size_t)s * WUBU_ROPE_DIM;
        for (int i = 0; i < WUBU_ROPE_DIM / 2; i++) {
            float x0 = row[i], x1 = row[WUBU_ROPE_DIM / 2 + i];
            row[i] = x0 * c[i] - x1 * si[i];
            row[WUBU_ROPE_DIM / 2 + i] = x0 * si[i] + x1 * c[i];
        }
    }
}
/* the transposed rotations (the backward): same angles, negated */
static void unrope_q(float *dq, int seq, const float *cos_tbl,
                     const float *sin_tbl)
{
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < WUBU_HEADS; h++) {
            float *row = dq + ((size_t)s * D + (size_t)h * 64);
            const float *c = cos_tbl + (size_t)s * WUBU_ROPE_DIM;
            const float *si = sin_tbl + (size_t)s * WUBU_ROPE_DIM;
            for (int i = 0; i < WUBU_ROPE_DIM / 2; i++) {
                float g0 = row[i], g1 = row[WUBU_ROPE_DIM / 2 + i];
                row[i] = g0 * c[i] + g1 * si[i];
                row[WUBU_ROPE_DIM / 2 + i] = -g0 * si[i] + g1 * c[i];
            }
        }
    }
}
static void unrope_k(float *dk, int seq, const float *cos_tbl,
                     const float *sin_tbl)
{
    for (int s = 0; s < seq; s++) {
        float *row = dk + (size_t)s * 64;
        const float *c = cos_tbl + (size_t)s * WUBU_ROPE_DIM;
        const float *si = sin_tbl + (size_t)s * WUBU_ROPE_DIM;
        for (int i = 0; i < WUBU_ROPE_DIM / 2; i++) {
            float g0 = row[i], g1 = row[WUBU_ROPE_DIM / 2 + i];
            row[i] = g0 * c[i] + g1 * si[i];
            row[WUBU_ROPE_DIM / 2 + i] = -g0 * si[i] + g1 * c[i];
        }
    }
}

/* rms_norm backward. y = x * r * w, r = 1/sqrt(mean(x^2)+eps).
 * Accumulates dx into dx_out (may equal dy -- the per-element reads
 * happen before the writes) and dw into dw_out. */
static void rms_norm_backward(const float *x, const float *w,
                              const float *dy, float *dx_out, float *dw_out,
                              int n)
{
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + WUBU_EPS);
    float dot = 0;
    for (int i = 0; i < n; i++) dot += dy[i] * w[i] * x[i];
    float c = -r * r * r * dot / (float)n;
    for (int i = 0; i < n; i++) {
        float dz = dy[i] * w[i] * r;
        if (dx_out) dx_out[i] += dz + c * x[i];
        if (dw_out) dw_out[i] += dy[i] * x[i] * r;
    }
}

/* ---- the tied head: mean-reduced next-token CE + gradients ----
 * loss = mean_s CE(softmax(logits[s]), tokens[s+1])
 * The logits were computed ONCE by the forward (one GEMM into
 * bp->logits) -- this function only reads them. If dh_out: accumulates
 * dL/d(final_h)[s] into it. If demb: accumulates dL/d(embedding)[v]
 * into it (the head IS the embedding, tied). */
static float head_ce(wubu_model_t *m, wubu_bp_t *bp,
                     const uint16_t *tokens, int n_tokens,
                     float *dh_out, float *demb)
{
    int seq = bp->seq;
    float n_pos = (float)(seq - 1);
    float loss = 0;
    for (int s = 0; s < seq - 1; s++) {
        uint16_t target = tokens[s + 1];
        float *lg = bp->logits + (size_t)s * WUBU_VOCAB;
        float maxv = lg[0];
        for (int v = 1; v < WUBU_VOCAB; v++) if (lg[v] > maxv) maxv = lg[v];
        double sum = 0, lt = 0;
        for (int v = 0; v < WUBU_VOCAB; v++) {
            double p = exp((double)(lg[v] - maxv));
            if (v == target) lt = (double)lg[v];
            sum += p;
        }
        loss += (float)(((double)maxv + log(sum) - lt) / (double)n_pos);
        if (dh_out || demb) {
            /* convert the logits row in-place into the softmax error
             * g[s,v] = (p - onehot) / n_pos, then the two gradient
             * GEMMs (GPU when present):
             *   dh[s,d]  += sum_v g[s,v] * e[v,d]   (nt: g @ embedding)
             *   demb[v,d]+= sum_s g[s,v] * h[s,d]   (tx: g^T @ final_h)
             * The logits are consumed -- nothing needs them after. */
            for (int v = 0; v < WUBU_VOCAB; v++)
                lg[v] = (float)(exp((double)(lg[v] - maxv)) / sum
                                - (v == target ? 1.0 : 0.0)) / n_pos;
        }
    }
    if (dh_out || demb) {
        int np = seq - 1;
        if (demb && gpu_wubu_ready && gpu_wubu_ready() && gpu_wubu_matmul_tx &&
            (long)np * WUBU_VOCAB * D >= GPU_MIN_FLOP &&
            gpu_wubu_matmul_tx(demb, bp->logits, bp->final_h, WUBU_VOCAB, D, np)) {
            /* GPU: demb = g^T @ h */
        } else if (demb) {
            for (int s = 0; s < np; s++) {
                const float *g = bp->logits + (size_t)s * WUBU_VOCAB;
                const float *h = bp->final_h + (size_t)s * D;
                float *ga = demb;
                for (int v = 0; v < WUBU_VOCAB; v++) {
                    float gv = g[v];
                    if (gv == 0.0f) continue;
                    for (int d = 0; d < D; d++) ga[(size_t)v * D + d] += gv * h[d];
                }
            }
        }
        if (dh_out && gpu_wubu_ready && gpu_wubu_ready() && gpu_wubu_matmul_nt &&
            (long)np * WUBU_VOCAB * D >= GPU_MIN_FLOP &&
            gpu_wubu_matmul_nt(dh_out, m->embedding, bp->logits, np, D, WUBU_VOCAB)) {
            /* GPU: dh = g @ embedding */
        } else if (dh_out) {
            for (int s = 0; s < np; s++) {
                const float *g = bp->logits + (size_t)s * WUBU_VOCAB;
                float *dh = dh_out + (size_t)s * D;
                for (int d = 0; d < D; d++) {
                    float acc = 0;
                    for (int v = 0; v < WUBU_VOCAB; v++)
                        acc += g[v] * m->embedding[(size_t)v * D + d];
                    dh[d] += acc;
                }
            }
        }
    }
    return loss;
}

/* is layer l a selector layer? (layers 3, 7, 11) */
static int is_sel(const wubu_model_t *m, int l)
{
    return l >= 0 && l < m->n_layers && m->fire_sel[l];
}

/* The hybrid GQA attention on the CPU: the FD oracle (the FD tests
 * verify the grads through this exact math) AND the fallback when the
 * GPU tile is absent. OpenMP over the positions (each row is
 * independent). */
static void cpu_attn_loop(float *acc, const float *q, const float *k,
                          const float *v, int seq, int is_full)
{
    const int hwd = WUBU_HEADS * 64;   /* NOT 'D' -- the bp #defines D */
#pragma omp parallel for schedule(static)
    for (int s = 0; s < seq; s++) {
        float *acc_s = acc + (size_t)s * hwd;
        memset(acc_s, 0, hwd * sizeof(float));
        for (int h = 0; h < WUBU_HEADS; h++) {
            const float *qrow = q + (size_t)s * hwd + (size_t)h * 64;
            float maxv = -1e30f;
            int lo = is_full ? 0
                             : (s > WUBU_LOCAL_WIN ? s - WUBU_LOCAL_WIN + 1 : 0);
            int kv_n = 0;
            float probs[WUBU_LOCAL_WIN + 2];
            for (int t = lo; t <= s; t++) {
                const float *krow = k + (size_t)t * 64;
                float dot = 0;
                for (int i = 0; i < 64; i++) dot += qrow[i] * krow[i];
                dot *= 1.0f / sqrtf(64.0f);
                if (dot > maxv) maxv = dot;
                probs[kv_n++] = dot;
            }
            float sum = 0;
            for (int i = 0; i < kv_n; i++) {
                probs[i] = expf(probs[i] - maxv);
                sum += probs[i];
            }
            for (int i = 0; i < kv_n; i++) probs[i] /= sum;
            for (int i = 0; i < kv_n; i++) {
                const float *vrow = v + (size_t)(lo + i) * 64;
                for (int d = 0; d < 64; d++)
                    acc_s[h * 64 + d] += probs[i] * vrow[d];
            }
        }
    }
}

/* The hybrid GQA attention backward on the CPU: the FD oracle (the
 * same math the FD tests verify) AND the fallback when the GPU tile is
 * absent. OpenMP over the positions (each row is independent). */
static void cpu_attn_backward_loop(float *dq, float *dk, float *dv,
                                   const float *q, const float *k,
                                   const float *v, const float *dao,
                                   int seq, int is_full)
{
    const int hwd = WUBU_HEADS * 64;
    float inv = 1.0f / sqrtf(64.0f);
    /* the dk/dv accumulate ACROSS rows (the shared KV!) -- the naive
     * parallel-for raced on them; each thread owns its partials */
#pragma omp parallel
    {
        float *dk_part = (float *)calloc((size_t)seq * 64, sizeof(float));
        float *dv_part = (float *)calloc((size_t)seq * 64, sizeof(float));
#pragma omp for schedule(static)
        for (int s = 0; s < seq; s++) {
        for (int h = 0; h < WUBU_HEADS; h++) {
            const float *qrow = q + (size_t)s * hwd + (size_t)h * 64;
            int lo = is_full ? 0
                             : (s > WUBU_LOCAL_WIN ? s - WUBU_LOCAL_WIN + 1 : 0);
            int kv_n = 0;
            float probs[WUBU_LOCAL_WIN + 2];
            float maxv = -1e30f;
            for (int t = lo; t <= s; t++) {
                const float *krow = k + (size_t)t * 64;
                float dot = 0;
                for (int i = 0; i < 64; i++) dot += qrow[i] * krow[i];
                dot *= inv;
                if (dot > maxv) maxv = dot;
                probs[kv_n++] = dot;
            }
            float sum = 0;
            for (int i = 0; i < kv_n; i++) { probs[i] = expf(probs[i] - maxv); sum += probs[i]; }
            for (int i = 0; i < kv_n; i++) probs[i] /= sum;
            const float *dao_h = dao + (size_t)s * hwd + (size_t)h * 64;
            float *dq_h  = dq  + (size_t)s * hwd + (size_t)h * 64;
            float mean = 0;
            for (int i = 0; i < kv_n; i++) {
                const float *vrow = v + (size_t)(lo + i) * 64;
                float dvdot = 0;
                for (int d = 0; d < 64; d++) dvdot += dao_h[d] * vrow[d];
                mean += probs[i] * dvdot;
            }
            for (int i = 0; i < kv_n; i++) {
                const float *krow = k + (size_t)(lo + i) * 64;
                const float *vrow = v + (size_t)(lo + i) * 64;
                float dvdot = 0;
                for (int d = 0; d < 64; d++) dvdot += dao_h[d] * vrow[d];
                float dscore = probs[i] * (dvdot - mean) * inv;
                float *dk_t = dk_part + (size_t)(lo + i) * 64;
                float *dv_t = dv_part + (size_t)(lo + i) * 64;
                for (int d = 0; d < 64; d++) {
                    dq_h[d]  += dscore * krow[d];
                    dk_t[d]  += dscore * qrow[d];
                    dv_t[d]  += probs[i] * dao_h[d];
                }
            }
        }
        }
#pragma omp critical
        {
            for (int i = 0; i < seq * 64; i++) {
                dk[i] += dk_part[i];
                dv[i] += dv_part[i];
            }
        }
        free(dk_part);
        free(dv_part);
    }
}

float wubu_bp_forward(wubu_model_t *m, wubu_buf_t *b, wubu_bp_t *bp,
                       const uint16_t *tokens, int n_tokens)
{
    if (!m || !b || !bp || !tokens || n_tokens < 2 ||
        n_tokens > bp->cap_seq) return 0;
    int seq = n_tokens;
    bp->seq = seq;
    memset(bp->sel_w0, 0, (size_t)WUBU_LAYERS * sizeof(float));

    /* embedding -> x_in[0] (saved to emb_in for layer 0's backward
     * and the first selector's checkpoint) */
    float *x0 = bp->x_in;
    for (int s = 0; s < seq; s++) {
        uint16_t tok = tokens[s];
        const float *e = m->embedding + (size_t)tok * D;
        memcpy(x0 + (size_t)s * D, e, D * sizeof(float));
    }
    memcpy(bp->emb_in, x0, (size_t)seq * D * sizeof(float));
    memcpy(bp->ckpt, x0, (size_t)seq * D * sizeof(float));

    int Ln = m->n_layers;
    for (int l = 0; l < Ln; l++) {
        wubu_block_t *blk = &m->blocks[l];
        float *x_in_l  = bp->x_in      + (size_t)l * seq * D;
        float *x_out_l = (l + 1 < Ln)
                             ? bp->x_in + (size_t)(l + 1) * seq * D : NULL;
        float *an_l    = bp->attn_norm + (size_t)l * seq * D;
        float *q_pre_l = bp->q_pre     + (size_t)l * seq * D;
        float *k_pre_l = bp->k_pre     + (size_t)l * seq * 64;
        float *q_l     = bp->q         + (size_t)l * seq * D;
        float *k_l     = bp->k         + (size_t)l * seq * 64;
        float *v_l     = bp->v         + (size_t)l * seq * 64;
        float *ao_l    = bp->attn_out  + (size_t)l * seq * D;
        float *o_l     = bp->o_out     + (size_t)l * seq * D;
        float *g_l     = bp->g_val     + (size_t)l * seq * D;
        float *fn_l    = bp->ffn_norm  + (size_t)l * seq * D;
        float *fg_l    = bp->ffn_gate  + (size_t)l * seq * 2 * FF;
        float *fu_l    = bp->ffn_up    + (size_t)l * seq * FF;
        float *fo_l    = bp->ffn_out   + (size_t)l * seq * D;
        float *sel_l   = bp->sel_out   + (size_t)l * seq * D;

        /* attention norm */
        for (int s = 0; s < seq; s++)
            rms_norm(an_l + (size_t)s * D, x_in_l + (size_t)s * D,
                     blk->attn_norm, D);
        /* q/k/v projections (save the pre-norm q/k for the backward) */
        mm(q_pre_l, blk->q_proj, an_l, WUBU_HEADS * 64, D, seq);
        mm(k_pre_l, blk->k_proj, an_l, 64, D, seq);
        mm(v_l, blk->v_proj, an_l, 64, D, seq);
        memcpy(q_l, q_pre_l, (size_t)seq * D * sizeof(float));
        memcpy(k_l, k_pre_l, (size_t)seq * 64 * sizeof(float));
        /* qk-norm + rope */
        for (int s = 0; s < seq; s++) {
            for (int h = 0; h < WUBU_HEADS; h++) {
                float *qr = q_l + (size_t)s * D + (size_t)h * 64;
                rms_norm(qr, qr, blk->q_norm, 64);
            }
            rms_norm(k_l + (size_t)s * 64, k_l + (size_t)s * 64,
                     blk->k_norm, 64);
        }
        rope_q(q_l, seq, b->cos_tbl, b->sin_tbl);
        rope_k(k_l, seq, b->cos_tbl, b->sin_tbl);

        /* GQA attention (7 Q heads share the single KV head). The GPU
         * tile (the PowerVR/FlashAttention principle) when ready and
         * the sequence is worth the upload; the CPU loop below stays
         * the FD oracle and the fallback. */
        int is_full = ((l + 1) % WUBU_FULL_EVERY == 0);
        if (gpu_wubu_ready && gpu_wubu_ready() && gpu_wubu_attn &&
            seq >= 32) {
            float *acc = ao_l;
            if (gpu_wubu_attn(acc, q_l, k_l, v_l, seq, WUBU_HEADS, 64,
                               WUBU_LOCAL_WIN, is_full)) {
                /* the GPU tile path: ao_l holds the attention output */
            } else {
                memset(acc, 0, (size_t)seq * D * sizeof(float));
                cpu_attn_loop(acc, q_l, k_l, v_l, seq, is_full);
            }
        } else {
            memset(ao_l, 0, (size_t)seq * D * sizeof(float));
            cpu_attn_loop(ao_l, q_l, k_l, v_l, seq, is_full);
        }
        /* o_proj + gate; gated residual: x = x + o * sigmoid(g) */
        mm(o_l, blk->o_proj, ao_l, D, D, seq);
        mm(g_l, blk->g_proj, an_l, D, D, seq);
        for (int s = 0; s < seq; s++) {
            float *xs = x_in_l + (size_t)s * D;
            for (int d = 0; d < D; d++)
                xs[d] += o_l[(size_t)s * D + d] *
                         (1.0f / (1.0f + expf(-g_l[(size_t)s * D + d])));
        }
        /* ffn norm */
        for (int s = 0; s < seq; s++)
            rms_norm(fn_l + (size_t)s * D, x_in_l + (size_t)s * D,
                     blk->ffn_norm, D);
        /* gate_up + bounded swiglu */
        mm(fg_l, blk->gate_up, fn_l, 2 * FF, D, seq);
        for (int s = 0; s < seq; s++) {
            const float *g = fg_l + (size_t)s * 2 * FF;
            float *u = fu_l + (size_t)s * FF;
            for (int d = 0; d < FF; d++) {
                float gv = g[d], uv = g[d + FF];
                if (gv > WUBU_CLIP) gv = WUBU_CLIP;
                if (uv > WUBU_CLIP) uv = WUBU_CLIP;
                if (uv < -WUBU_CLIP) uv = -WUBU_CLIP;
                u[d] = silu(gv) * uv;
            }
        }
        mm(fo_l, blk->down, fu_l, D, FF, seq);
        for (int s = 0; s < seq; s++) {
            float *xs = x_in_l + (size_t)s * D;
            for (int d = 0; d < D; d++) xs[d] += fo_l[(size_t)s * D + d];
        }
        /* residual selector: blend into sel_out[l]; x_in[l] keeps the
         * layer's REAL output (the backward needs it exact). */
        if (is_sel(m, l)) {
            float *sw = m->selectors[(l + 1) / WUBU_SELECT_EVERY - 1];
            float w0sum = 0;
            for (int s = 0; s < seq; s++) {
                float *cp = bp->ckpt + (size_t)s * D;
                float *cur = x_in_l + (size_t)s * D;
                float *bl = sel_l + (size_t)s * D;
                float sc = 0, ss2 = 0;
                for (int d = 0; d < D; d++) {
                    float ncp = cp[d] * (1.0f / sqrtf(D * 1.0f));
                    float ncu = cur[d] * (1.0f / sqrtf(D * 1.0f));
                    sc += sw[d] * ncp;
                    ss2 += sw[d] * ncu;
                }
                float w0 = expf(sc), w1 = expf(ss2);
                float ws = w0 + w1 + 1e-9f;
                w0 /= ws; w1 /= ws;
                for (int d = 0; d < D; d++)
                    bl[d] = w0 * cp[d] + w1 * cur[d];
                memcpy(cp, bl, D * sizeof(float));
                w0sum += w0;
            }
            bp->sel_w0[l] = w0sum / (float)seq;
        }
        /* the residual stream chains: the next layer's input = the
         * layer output (or the selector blend). */
        if (x_out_l) {
            const float *src = is_sel(m, l) ? sel_l : x_in_l;
            memcpy(x_out_l, src, (size_t)seq * D * sizeof(float));
        }
    }
    /* final norm: the input is the last layer's output (the blend if
     * the last layer is a selector layer) */
    int Llast = m->n_layers - 1;
    const float *xlast = is_sel(m, Llast)
                             ? bp->sel_out + (size_t)Llast * seq * D
                             : bp->x_in + (size_t)Llast * seq * D;
    for (int s = 0; s < seq; s++)
        rms_norm(bp->final_h + (size_t)s * D, xlast + (size_t)s * D,
                 m->final_norm, D);
    /* the head logits: ONE GEMM (GPU when present) -- head_ce only
     * reads them, so the per-step cost is a single pass over the vocab */
    mm(bp->logits, m->embedding, bp->final_h, WUBU_VOCAB, D, seq);
    return head_ce(m, bp, tokens, n_tokens, NULL, NULL);
}

/* acc[s, i<in_w] = sum_{o<out_w} w[o, i] * x[s, o]  (the backward's
 * input-gradient products: the stored weight applied to a gradient),
 * GPU when big enough. */
static void mm_t(float *acc, const float *w, const float *x,
                 int out_w, int in_w, int seq)
{
    if (gpu_wubu_ready && gpu_wubu_ready() && gpu_wubu_matmul_nt &&
        (long)seq * out_w * in_w >= GPU_MIN_FLOP &&
        gpu_wubu_matmul_nt(acc, w, x, seq, in_w, out_w))
        return;
    for (int s = 0; s < seq; s++) {
        const float *xs = x + (size_t)s * out_w;
        float *as = acc + (size_t)s * in_w;
        for (int i = 0; i < in_w; i++) {
            float accv = 0;
            for (int o = 0; o < out_w; o++) accv += w[(size_t)o * in_w + i] * xs[o];
            as[i] = accv;
        }
    }
}

/* wg[o, i] += sum_s dy[s, o] * inp[s, i]  (the backward's weight-gradient
 * outer products), GPU when big enough. */
static void wg_t(float *wg, const float *dy, const float *inp,
                 int out_w, int in_w, int seq)
{
    if (gpu_wubu_ready && gpu_wubu_ready() && gpu_wubu_matmul_tx &&
        (long)seq * out_w * in_w >= GPU_MIN_FLOP &&
        gpu_wubu_matmul_tx(wg, dy, inp, out_w, in_w, seq))
        return;
    for (int o = 0; o < out_w; o++) {
        float *wr = wg + (size_t)o * in_w;
        for (int i = 0; i < in_w; i++) {
            float acc = 0;
            for (int s = 0; s < seq; s++) acc += dy[(size_t)s * out_w + o] * inp[(size_t)s * in_w + i];
            wr[i] += acc;
        }
    }
}

/* ---- the REAL backward pass ---- */

float wubu_bp_backward(wubu_model_t *m, wubu_buf_t *b, wubu_bp_t *bp,
                        wubu_train_t *tr, const uint16_t *tokens,
                        int n_tokens)
{
    if (!m || !b || !bp || !tr || !tokens || bp->seq != n_tokens) return 0;
    int seq = n_tokens;
    const float *cos_tbl = b->cos_tbl, *sin_tbl = b->sin_tbl;
    float *demb = tr->emb_g;

    /* ---- head: softmax CE vs the tied embedding ---- */
    float *dh_final = calloc_f((size_t)seq * D);
    float *dlast    = calloc_f((size_t)seq * D); /* dL/d(last layer out) */
    float *dc       = calloc_f((size_t)seq * D); /* the checkpoint grad */
    if (!dh_final || !dlast || !dc) { free(dh_final); free(dlast); free(dc); return 0; }
    float loss = head_ce(m, bp, tokens, n_tokens, dh_final, demb);

    /* final norm backward: dh_final -> dlast (+ final_norm grads) */
    int Llast = m->n_layers - 1;
    const float *xlast = is_sel(m, Llast)
                             ? bp->sel_out + (size_t)Llast * seq * D
                             : bp->x_in + (size_t)Llast * seq * D;
    for (int s = 0; s < seq; s++)
        rms_norm_backward(xlast + (size_t)s * D, m->final_norm,
                          dh_final + (size_t)s * D,
                          dlast + (size_t)s * D, tr->norm_g[4 * WUBU_LAYERS], D);

    /* ---- per-layer backward (REVERSED) ---- */
    for (int l = m->n_layers - 1; l >= 0; l--) {
        wubu_block_t *blk = &m->blocks[l];
        float *x_in_l  = bp->x_in      + (size_t)l * seq * D;
        float *an_l    = bp->attn_norm + (size_t)l * seq * D;
        float *q_pre_l = bp->q_pre     + (size_t)l * seq * D;
        float *k_pre_l = bp->k_pre     + (size_t)l * seq * 64;
        float *q_l     = bp->q         + (size_t)l * seq * D;
        float *k_l     = bp->k         + (size_t)l * seq * 64;
        float *v_l     = bp->v         + (size_t)l * seq * 64;
        float *ao_l    = bp->attn_out  + (size_t)l * seq * D;
        float *o_l     = bp->o_out     + (size_t)l * seq * D;
        float *g_l     = bp->g_val     + (size_t)l * seq * D;
        float *fn_l    = bp->ffn_norm  + (size_t)l * seq * D;
        float *fg_l    = bp->ffn_gate  + (size_t)l * seq * 2 * FF;
        float *fu_l    = bp->ffn_up    + (size_t)l * seq * FF;
        float *sel_l   = bp->sel_out   + (size_t)l * seq * D;
        float *dx      = bp->s_dx;        /* dL/d(layer output) */
        float *dxe     = bp->s_dxentry;   /* dL/d(layer input)   */
        float *dfo     = bp->s_dffn_out;
        float *dfu     = bp->s_dfu;
        float *dfg     = bp->s_dfg;       /* [seq, 2*FF] */
        float *dfn     = bp->s_dfn;
        float *x1      = bp->s_dg;        /* scratch: x after attn */
        float *dao     = bp->s_dao;
        float *dod     = bp->s_do;
        float *dgd     = bp->s_dg;        /* reused: dL/dg after x1 */
        float *dan     = bp->s_dan;
        float *dq      = bp->s_dq;
        float *dk      = bp->s_dk;
        float *dv      = bp->s_dv;

        /* the incoming gradient: from the layer above (or the final
         * norm for l == L-1). */
        if (l == m->n_layers - 1)
            memcpy(dx, dlast, (size_t)seq * D * sizeof(float));
        else
            memcpy(dx, dxe, (size_t)seq * D * sizeof(float));
        /* every scratch accumulator is zeroed fresh per layer */
        memset(dxe, 0, (size_t)seq * D * sizeof(float));
        memset(dan, 0, (size_t)seq * D * sizeof(float));
        memset(dao, 0, (size_t)seq * D * sizeof(float));
        memset(dfn, 0, (size_t)seq * D * sizeof(float));

        /* ---- residual selector: route the blend gradient into the
         * layer (w1) and the checkpoint (w0) ---- */
        if (is_sel(m, l)) {
            int sel = (l + 1) / WUBU_SELECT_EVERY - 1;
            float *sw = m->selectors[sel];
            /* selectors are 1-D params -> their grad lives in the
             * AdamW norm slots ([4*L + 1 + sel]) */
            float *sg = tr->norm_g[4 * WUBU_LAYERS + 1 + sel];
            /* the checkpoint BEFORE this blend = the previous selector
             * blend, or the embedding for the first selector */
            float *ckpt_pre = (l == WUBU_SELECT_EVERY - 1)
                                  ? bp->emb_in
                                  : bp->sel_out + (size_t)(l - WUBU_SELECT_EVERY) * seq * D;
            for (int s = 0; s < seq; s++) {
                float *cp = ckpt_pre + (size_t)s * D;
                float *cur = x_in_l + (size_t)s * D;
                float sc = 0, ss2 = 0;
                for (int d = 0; d < D; d++) {
                    float ncp = cp[d] * (1.0f / sqrtf(D * 1.0f));
                    float ncu = cur[d] * (1.0f / sqrtf(D * 1.0f));
                    sc += sw[d] * ncp;
                    ss2 += sw[d] * ncu;
                }
                float w0 = expf(sc), w1 = expf(ss2);
                float ws = w0 + w1 + 1e-9f;
                w0 /= ws; w1 /= ws;
                float *dbl = dx + (size_t)s * D;
                float *dck = dc + (size_t)s * D;
                float dot_cp = 0, dot_cur = 0;
                for (int d = 0; d < D; d++) {
                    /* the blend gradient = the layer-above gradient +
                     * the checkpoint chain from later selectors */
                    float db = dbl[d] + dck[d];
                    dbl[d] = db * w1;      /* dL/d(layer output)   */
                    dck[d] = db * w0;      /* dL/d(checkpoint)     */
                    dot_cp += db * cp[d];
                    dot_cur += db * cur[d];
                }
                /* selector weight gradient (softmax chain) */
                float dsc  = w0 * (dot_cp * (1 - w0) - dot_cur * w1);
                float dss2 = w1 * (dot_cur * (1 - w1) - dot_cp * w0);
                for (int d = 0; d < D; d++) {
                    float ncp = cp[d] * (1.0f / sqrtf(D * 1.0f));
                    float ncu = cur[d] * (1.0f / sqrtf(D * 1.0f));
                    sg[d] += dsc * ncp + dss2 * ncu;
                }
            }
        }

        /* ---- FFN path ---- */
        /* dL/dfo = dx (the ffn add identity); down grads + dL/dfu */
        memcpy(dfo, dx, (size_t)seq * D * sizeof(float));
        mm_t(dfu, blk->down, dfo, D, FF, seq);       /* dfu[s,d] = Σ_o down[o,d]*dfo[s,o] */
        wg_t(tr->down_g[l], dfo, fu_l, D, FF, seq);  /* down_g[o,d] += Σ_s dfo[s,o]*fu[s,d] */
        /* swiglu backward: u = silu(clip(g)) * clip(u) */
        for (int s = 0; s < seq; s++) {
            const float *fg = fg_l + (size_t)s * 2 * FF;
            const float *du = dfu + (size_t)s * FF;
            float *dgu = dfg + (size_t)s * 2 * FF;
            for (int d = 0; d < FF; d++) {
                float g_raw = fg[d], u_raw = fg[d + FF];
                float gv = g_raw > WUBU_CLIP ? WUBU_CLIP : g_raw;
                float uv = u_raw > WUBU_CLIP ? WUBU_CLIP : u_raw;
                if (uv < -WUBU_CLIP) uv = -WUBU_CLIP;
                float g_ok = (g_raw <= WUBU_CLIP) ? 1.0f : 0.0f;
                float u_ok = (u_raw <= WUBU_CLIP && u_raw >= -WUBU_CLIP)
                                 ? 1.0f : 0.0f;
                dgu[d]        = silu_deriv(gv) * uv * du[d] * g_ok;
                dgu[d + FF]   = silu(gv) * du[d] * u_ok;
            }
        }
        /* gate_up grads + dL/dfn */
        mm_t(dfn, blk->gate_up, dfg, 2 * FF, D, seq);   /* dfn[s,i] = Σ_o gate_up[o,i]*dfg[s,o] */
        wg_t(tr->gate_up_g[l], dfg, fn_l, 2 * FF, D, seq);
        /* ffn_norm backward: x1 = x_entry + o*sigmoid(g) (recomputed) */
        for (int s = 0; s < seq; s++) {
            const float *o_s = o_l + (size_t)s * D;
            const float *g_s = g_l + (size_t)s * D;
            float *x1_s = x1 + (size_t)s * D;
            for (int d = 0; d < D; d++)
                x1_s[d] = o_s[d] * sigm(g_s[d]);
        }
        {
            /* x1 also needs the entry -- layer input + the gated add */
            const float *x_entry = (l == 0)
                                       ? bp->emb_in
                                       : (is_sel(m, l - 1)
                                              ? bp->sel_out + (size_t)(l - 1) * seq * D
                                              : bp->x_in + (size_t)(l - 1) * seq * D);
            for (int s = 0; s < seq; s++) {
                const float *xe = x_entry + (size_t)s * D;
                float *x1_s = x1 + (size_t)s * D;
                for (int d = 0; d < D; d++) x1_s[d] += xe[d];
            }
        }
        for (int s = 0; s < seq; s++)
            rms_norm_backward(x1 + (size_t)s * D, blk->ffn_norm,
                              dfn + (size_t)s * D,
                              dxe + (size_t)s * D, tr->norm_g[4 * l + 1], D);
        /* the ffn-add identity: dL/dx1 += dx */
        for (int s = 0; s < seq; s++) {
            const float *df = dx + (size_t)s * D;
            float *dx1_s = dxe + (size_t)s * D;
            for (int d = 0; d < D; d++) dx1_s[d] += df[d];
        }

        /* ---- attention path (dx1_total = dxe) ---- */
        /* gated residual: x1 = x_entry + o*sigmoid(g) */
        for (int s = 0; s < seq; s++) {
            const float *dx1_s = dxe + (size_t)s * D;
            const float *o_s = o_l + (size_t)s * D;
            const float *g_s = g_l + (size_t)s * D;
            float *do_s = dod + (size_t)s * D;
            float *dg_s = dgd + (size_t)s * D;
            for (int d = 0; d < D; d++) {
                float sg = sigm(g_s[d]);
                do_s[d] = dx1_s[d] * sg;
                dg_s[d] = dx1_s[d] * o_s[d] * sg * (1.0f - sg);
            }
        }
        /* o_proj grads + dL/dattn_out ; g_proj grads + dL/dan */
        mm_t(dao, blk->o_proj, dod, D, D, seq);         /* dao[s,i] = Σ_o o_proj[o,i]*dod[s,o] */
        wg_t(tr->o_proj_g[l], dod, ao_l, D, D, seq);    /* o_proj_g[o,i] += Σ_s dod[s,o]*ao[s,i] */
        mm_t(dod, blk->g_proj, dgd, D, D, seq);        /* into scratch, then dan += */
        wg_t(tr->g_proj_g[l], dgd, an_l, D, D, seq);
        for (int i = 0; i < seq * D; i++) dan[i] += dod[i];
        /* attention backward (GQA: all heads share the KV rows). The
         * GPU tile when ready (the FD oracle is the CPU loop below --
         * the fallback AND the reference the tests run through). */
        int is_full = ((l + 1) % WUBU_FULL_EVERY == 0);
        memset(dq, 0, (size_t)seq * D * sizeof(float));
        memset(dk, 0, (size_t)seq * 64 * sizeof(float));
        memset(dv, 0, (size_t)seq * 64 * sizeof(float));
        if (gpu_wubu_ready && gpu_wubu_ready() && gpu_wubu_attn_backward &&
            seq >= 32) {
            if (!gpu_wubu_attn_backward(dq, dk, dv, q_l, k_l, v_l, ao_l, dao,
                                         seq, WUBU_HEADS, 64,
                                         WUBU_LOCAL_WIN, is_full)) {
                memset(dq, 0, (size_t)seq * D * sizeof(float));
                memset(dk, 0, (size_t)seq * 64 * sizeof(float));
                memset(dv, 0, (size_t)seq * 64 * sizeof(float));
                cpu_attn_backward_loop(dq, dk, dv, q_l, k_l, v_l, dao,
                                       seq, is_full);
            }
        } else {
            cpu_attn_backward_loop(dq, dk, dv, q_l, k_l, v_l, dao,
                                   seq, is_full);
        }
        /* rope backward, then the qk-norm backward (needs the pre-norm
         * q/k saved by the forward) */
        unrope_q(dq, seq, cos_tbl, sin_tbl);
        unrope_k(dk, seq, cos_tbl, sin_tbl);
        for (int s = 0; s < seq; s++) {
            for (int h = 0; h < WUBU_HEADS; h++)
                rms_norm_backward(q_pre_l + (size_t)s * D + (size_t)h * 64,
                                  blk->q_norm,
                                  dq + (size_t)s * D + (size_t)h * 64,
                                  dq + (size_t)s * D + (size_t)h * 64,
                                  tr->norm_g[4 * l + 2], 64);
            rms_norm_backward(k_pre_l + (size_t)s * 64, blk->k_norm,
                              dk + (size_t)s * 64, dk + (size_t)s * 64,
                              tr->norm_g[4 * l + 3], 64);
        }
        /* q/k/v projection grads + dL/dan from the attention path */
        mm_t(dod, blk->q_proj, dq, WUBU_HEADS * 64, D, seq);
        wg_t(tr->q_proj_g[l], dq, an_l, WUBU_HEADS * 64, D, seq);
        for (int i = 0; i < seq * D; i++) dan[i] += dod[i];
        mm_t(dod, blk->k_proj, dk, 64, D, seq);
        wg_t(tr->k_proj_g[l], dk, an_l, 64, D, seq);
        for (int i = 0; i < seq * D; i++) dan[i] += dod[i];
        mm_t(dod, blk->v_proj, dv, 64, D, seq);
        wg_t(tr->v_proj_g[l], dv, an_l, 64, D, seq);
        for (int i = 0; i < seq * D; i++) dan[i] += dod[i];
        /* attn_norm backward: x_entry (the layer input) */
        {
            const float *x_entry = (l == 0)
                                       ? bp->emb_in
                                       : (is_sel(m, l - 1)
                                              ? bp->sel_out + (size_t)(l - 1) * seq * D
                                              : bp->x_in + (size_t)(l - 1) * seq * D);
            for (int s = 0; s < seq; s++)
                rms_norm_backward(x_entry + (size_t)s * D, blk->attn_norm,
                                  dan + (size_t)s * D,
                                  dxe + (size_t)s * D, tr->norm_g[4 * l + 0], D);
        }
        /* the gated-residual identity is already in dxe (dx1_total);
         * the attn_norm backward accumulated on top. dxe is now
         * dL/d(layer input) -- passed to the layer below. */
        (void)sel_l;
    }

    /* ---- embedding gradients: the input-token path (the residual
     * stream into layer 0 + the checkpoint chain) ---- */
    for (int s = 0; s < seq; s++) {
        uint16_t tok = tokens[s];
        float *ga = demb + (size_t)tok * D;
        const float *dx0 = bp->s_dxentry + (size_t)s * D;
        const float *dck = dc + (size_t)s * D;
        for (int d = 0; d < D; d++) ga[d] += dx0[d] + dck[d];
    }

    free(dh_final);
    free(dlast);
    free(dc);
    tr->loss_sum += loss;
    tr->micro_steps++;
    return loss;
}

/* ---------- the REAL Muon step (Newton-Schulz 5) ---------- */

static void adamw_update(float *w, float *g, float *m, float *v, size_t n,
                         float lr, float wd, uint32_t step)
{
    float b1 = 0.9f, b2 = 0.95f, eps = 1e-8f;
    float bc1 = 1.0f - powf(b1, (float)step);
    float bc2 = 1.0f - powf(b2, (float)step);
    for (size_t i = 0; i < n; i++) {
        float gw = g[i] + wd * w[i];
        m[i] = b1 * m[i] + (1 - b1) * gw;
        v[i] = b2 * v[i] + (1 - b2) * gw * gw;
        float mh = m[i] / bc1, vh = v[i] / bc2;
        w[i] -= lr * mh / (sqrtf(vh) + eps);
        g[i] = 0;
    }
}

/* the Newton-Schulz 5 orthogonalization in place on X [rows, cols].
 * (a, b, c) = (3.4445, -4.7750, 2.0315); A = X X^T; B = bA + cA^2;
 * X = aX + BX. Tall matrices are transposed first (work on the
 * columns' orthogonal basis), then the result is transposed back.
 * scratch must hold 2 * trows * trows floats (A + B); tmp must hold
 * rows * cols floats. */
static void ns5_inplace(float *X, int rows, int cols, float *scratch,
                        float *tmp)
{
    float nrm = 0;
    for (int i = 0; i < rows * cols; i++) nrm += X[i] * X[i];
    nrm = sqrtf(nrm);
    if (nrm > 1e-12f)
        for (int i = 0; i < rows * cols; i++) X[i] /= nrm;
    else
        return;   /* a zero matrix stays zero */
#ifdef DBG_NS5
    printf("ns5 rows=%d cols=%d nrm=%g maxin=%g\n", rows, cols, nrm,
           (double)fabs(X[0]) + 1.0);
#endif

    int transposed = 0, trows = rows, tcols = cols;
    if (rows > cols) { transposed = 1; trows = cols; tcols = rows; }

    /* if transposed, build X^T into tmp and work there */
    float *M = X;
    if (transposed) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                tmp[j * rows + i] = X[i * cols + j];
        M = tmp;
    }
    float *A = scratch, *B = scratch + (size_t)trows * trows;
    const float a = 3.4445f, bb = -4.7750f, c = 2.0315f;
    for (int it = 0; it < 5; it++) {
        /* A[i,j] = sum_k M[i,k] M[j,k] */
        for (int i = 0; i < trows; i++) {
            for (int j = 0; j < trows; j++) {
                float acc = 0;
                const float *Mi = M + (size_t)i * tcols;
                const float *Mj = M + (size_t)j * tcols;
                for (int k = 0; k < tcols; k++) acc += Mi[k] * Mj[k];
                A[i * trows + j] = acc;
            }
        }
        /* B = b*A + c*(A@A) ; M = a*M + B@M */
        for (int i = 0; i < trows; i++) {
            for (int j = 0; j < trows; j++) {
                float aij = A[i * trows + j], a2 = 0;
                for (int k = 0; k < trows; k++) a2 += A[i * trows + k] * A[k * trows + j];
                B[i * trows + j] = bb * aij + c * a2;
            }
        }
        for (int i = 0; i < trows; i++) {
            for (int j = 0; j < tcols; j++) {
                float acc = a * M[i * tcols + j];
                for (int k = 0; k < trows; k++) acc += B[i * trows + k] * M[k * tcols + j];
                M[i * tcols + j] = acc;
            }
        }
        /* renormalize: the (a,b,c) polynomial escapes its attraction
         * basin for spread singular-value spectra in fp32; bounding the
         * Frobenius norm every iteration keeps the same convergence
         * dynamics (the ratios are scale-invariant) without the blowup */
        double ss = 0;
        for (int i = 0; i < trows; i++)
            for (int j = 0; j < tcols; j++) ss += (double)M[i * tcols + j] * M[i * tcols + j];
        float inv = (float)(1.0 / (sqrt(ss) + 1e-12));
        for (int i = 0; i < trows; i++)
            for (int j = 0; j < tcols; j++) M[i * tcols + j] *= inv;
    }
    if (transposed) {
        /* copy M^T back into X */
        for (int i = 0; i < trows; i++)
            for (int j = 0; j < tcols; j++)
                X[j * trows + i] = M[i * tcols + j];
    }
}

/* the 1-D AdamW slots (norms + selectors): weight pointer + size */
static float *norm_slot_weight(wubu_model_t *m, int slot, int *size)
{
    int L = WUBU_LAYERS;
    if (slot < 4 * L) {
        int l = slot / 4, k = slot % 4;
        wubu_block_t *blk = &m->blocks[l];
        switch (k) {
            case 0: *size = D; return blk->attn_norm;
            case 1: *size = D; return blk->ffn_norm;
            case 2: *size = WUBU_HEAD_DIM; return blk->q_norm;  /* per-head */
            default:*size = WUBU_HEAD_DIM; return blk->k_norm;  /* per-head */
        }
    }
    if (slot == 4 * L) { *size = D; return m->final_norm; }
    *size = D;
    return m->selectors[slot - (4 * L + 1)];
}

/* the per-matrix Muon update: Nesterov momentum, NS5, scaled step */
static double dot_grad(const float *g, size_t n)
{
    double s = 0;
    for (size_t i = 0; i < n; i++) s += (double)g[i] * g[i];
    return s;
}
static void scale_grad(float *g, size_t n, float s)
{
    for (size_t i = 0; i < n; i++) g[i] *= s;
}

static void muon_matrix(float *w, float *g, float *mom, int rows, int cols,
                        float lr, float wd, float mu, float *scratch,
                        float *look, float *trans)
{
    size_t n = (size_t)rows * cols;
    /* Nesterov momentum: buf = mu*buf + g ; X = buf + mu*buf */
    for (size_t i = 0; i < n; i++) mom[i] = mu * mom[i] + g[i];
    for (size_t i = 0; i < n; i++) look[i] = mom[i] + mu * mom[i];
    /* the NS5 orthogonalization: GPU when present (the optimizer was
     * the last CPU bottleneck -- ~61 GFLOP/step of NS5), CPU otherwise.
     * The NS5 does 15 GEMMs per call, so even a 448x448 matrix
     * (~1.35 GFLOP) is worth the upload -- the threshold is tiny.
     * The Gram variant (the square-space iteration, Tri Dao 2026) is
     * preferred: ~5x fewer rectangular FLOPs, identical math. */
    if (gpu_wubu_ready && gpu_wubu_ready() && (gpu_wubu_ns5_gram || gpu_wubu_ns5) &&
        (size_t)rows * cols >= 4096) {
        if (gpu_wubu_ns5_gram && gpu_wubu_ns5_gram(look, rows, cols)) {
            /* the Gram path: look is already orthogonalized */
        } else if (gpu_wubu_ns5 && gpu_wubu_ns5(look, rows, cols)) {
            /* the standard path */
        } else {
            ns5_inplace(look, rows, cols, scratch, trans);
        }
    } else if (getenv("REPRO_SKIP") && strstr(getenv("REPRO_SKIP"), "ns5")) {
        /* TEMP BISECT: skip the orthogonalization, keep mom/look/w/g */
    } else {
        ns5_inplace(look, rows, cols, scratch, trans);
    }
    /* the Moonlight per-matrix scale: normalize the NS update so its
     * RMS = 0.2 -- a bounded step that reuses the AdamW-scale LR */
    double ss = 0;
    for (size_t i = 0; i < n; i++) ss += (double)look[i] * look[i];
    float rms = (float)sqrt(ss / (double)n);
    float s = (rms > 1e-12f) ? 0.2f / rms : 0.0f;
    for (size_t i = 0; i < n; i++) {
        w[i] *= (1.0f - lr * wd);          /* decoupled weight decay */
        w[i] -= lr * look[i] * s;
        g[i] = 0;
    }
}

int wubu_bp_muon_step(wubu_model_t *m, wubu_train_t *tr,
                       const wubu_train_cfg_t *cfg, uint32_t step)
{
    if (!m || !tr || !cfg) return -1;
    const char *skip = getenv("REPRO_SKIP");   /* TEMP BISECT: clip|muon|adam */
    float mu_lr = cfg->muon_lr > 0 ? cfg->muon_lr : cfg->lr;
    float ad_lr = cfg->adam_lr > 0 ? cfg->adam_lr : cfg->lr;
    float wd = cfg->weight_decay;
    float mu = cfg->muon_momentum > 0 ? cfg->muon_momentum : 0.95f;

    /* global grad-norm clip (over every trainable gradient) */
    if (cfg->grad_clip > 0 && (!skip || strcmp(skip, "clip") != 0)) {
        double n2 = 0;
        for (int i = 0; i < m->n_layers; i++) {
            n2 += dot_grad(tr->q_proj_g[i], WUBU_DIM * (WUBU_HEADS * WUBU_HEAD_DIM));
            n2 += dot_grad(tr->k_proj_g[i], WUBU_DIM * (WUBU_KV_HEADS * WUBU_HEAD_DIM));
            n2 += dot_grad(tr->v_proj_g[i], WUBU_DIM * (WUBU_KV_HEADS * WUBU_HEAD_DIM));
            n2 += dot_grad(tr->o_proj_g[i], WUBU_DIM * (WUBU_HEADS * WUBU_HEAD_DIM));
            n2 += dot_grad(tr->g_proj_g[i], WUBU_DIM * (WUBU_HEADS * WUBU_HEAD_DIM));
            n2 += dot_grad(tr->gate_up_g[i], WUBU_DIM * (2 * WUBU_FFN_DIM));
            n2 += dot_grad(tr->down_g[i], WUBU_FFN_DIM * WUBU_DIM);
        }
        n2 += dot_grad(tr->emb_g, WUBU_VOCAB * WUBU_DIM);
        for (int i = 0; i < WUBU_NORM_SLOTS; i++) {
            int sz = 0;
            (void)norm_slot_weight(m, i, &sz);
            n2 += dot_grad(tr->norm_g[i], (size_t)sz);
        }
        float gn = (float)sqrt(n2);
        if (gn > cfg->grad_clip) {
            float s = cfg->grad_clip / gn;
            for (int i = 0; i < m->n_layers; i++) {
                scale_grad(tr->q_proj_g[i], WUBU_DIM * (WUBU_HEADS * WUBU_HEAD_DIM), s);
                scale_grad(tr->k_proj_g[i], WUBU_DIM * (WUBU_KV_HEADS * WUBU_HEAD_DIM), s);
                scale_grad(tr->v_proj_g[i], WUBU_DIM * (WUBU_KV_HEADS * WUBU_HEAD_DIM), s);
                scale_grad(tr->o_proj_g[i], WUBU_DIM * (WUBU_HEADS * WUBU_HEAD_DIM), s);
                scale_grad(tr->g_proj_g[i], WUBU_DIM * (WUBU_HEADS * WUBU_HEAD_DIM), s);
                scale_grad(tr->gate_up_g[i], WUBU_DIM * (2 * WUBU_FFN_DIM), s);
                scale_grad(tr->down_g[i], WUBU_FFN_DIM * WUBU_DIM, s);
            }
            scale_grad(tr->emb_g, WUBU_VOCAB * WUBU_DIM, s);
            for (int i = 0; i < WUBU_NORM_SLOTS; i++) {
                int sz = 0;
                (void)norm_slot_weight(m, i, &sz);
                scale_grad(tr->norm_g[i], (size_t)sz, s);
            }
        }
    }

    /* Muon group: the 2D hidden matrices */
    if (!skip || strcmp(skip, "muon") != 0) {
    {
        size_t max_cells = (size_t)WUBU_DIM * (2 * WUBU_FFN_DIM);  /* gate_up */
        size_t max_sq = (size_t)WUBU_DIM * WUBU_DIM;               /* NS A/B mats */
        size_t max_fd = (size_t)WUBU_FFN_DIM * WUBU_DIM;           /* down */
        size_t scratch_n = 2 * max_sq + 2 * max_cells + max_fd;
        float *scratch = (float *)malloc(scratch_n * sizeof(float));
        float *look = scratch + 2 * max_sq;
        float *trans = look + max_cells;
        if (!scratch) return -1;
        for (int i = 0; i < m->n_layers; i++) {
            wubu_block_t *blk = &m->blocks[i];
            muon_matrix(blk->q_proj,  tr->q_proj_g[i],  tr->q_proj_m[i],
                        WUBU_DIM, WUBU_HEADS * WUBU_HEAD_DIM, mu_lr, wd, mu, scratch, look, trans);
            muon_matrix(blk->k_proj,  tr->k_proj_g[i],  tr->k_proj_m[i],
                        WUBU_DIM, WUBU_KV_HEADS * WUBU_HEAD_DIM, mu_lr, wd, mu, scratch, look, trans);
            muon_matrix(blk->v_proj,  tr->v_proj_g[i],  tr->v_proj_m[i],
                        WUBU_DIM, WUBU_KV_HEADS * WUBU_HEAD_DIM, mu_lr, wd, mu, scratch, look, trans);
            muon_matrix(blk->o_proj,  tr->o_proj_g[i],  tr->o_proj_m[i],
                        WUBU_DIM, WUBU_HEADS * WUBU_HEAD_DIM, mu_lr, wd, mu, scratch, look, trans);
            muon_matrix(blk->g_proj,  tr->g_proj_g[i],  tr->g_proj_m[i],
                        WUBU_DIM, WUBU_HEADS * WUBU_HEAD_DIM, mu_lr, wd, mu, scratch, look, trans);
            muon_matrix(blk->gate_up, tr->gate_up_g[i], tr->gate_up_m[i],
                        WUBU_DIM, 2 * WUBU_FFN_DIM, mu_lr, wd, mu, scratch, look, trans);
            muon_matrix(blk->down,    tr->down_g[i],    tr->down_m[i],
                        WUBU_FFN_DIM, WUBU_DIM, mu_lr, wd, mu, scratch, look, trans);
        }
        free(scratch);
    }
    }

    /* AdamW group: the embedding, the norms, the selectors */
    if (!skip || strcmp(skip, "adam") != 0) {
    adamw_update(m->embedding, tr->emb_g, tr->emb_m, tr->emb_v,
                 WUBU_VOCAB * WUBU_DIM, ad_lr, wd, step);
    for (int i = 0; i < WUBU_NORM_SLOTS; i++) {
        int sz = 0;
        float *w = norm_slot_weight(m, i, &sz);
        if (!w) continue;
        adamw_update(w, tr->norm_g[i], tr->norm_m[i], tr->norm_v[i],
                     (size_t)sz, ad_lr, wd, step);
    }
    }
    /* the weights changed: the GPU cache re-uploads on the next matmul */
    if (gpu_wubu_mark_weights_dirty) gpu_wubu_mark_weights_dirty();
    return 0;
}
