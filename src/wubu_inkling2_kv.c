/* INKLING2 KV CACHE - storage layer.
 *
 * The forward pass currently recomputes the whole prefix on every decode step (24 layers x 3 loop passes over
 * every position), which makes generation O(n^2). This module holds the per-visit K/V that has to survive
 * between steps; the forward pass keeps computing everything until the read path lands, so nothing here is on
 * the hot path yet.
 *
 * Two things this model forces that a plain transformer cache does not:
 *
 * 1. PER VISIT, NOT PER LAYER. The middle 12 layers are re-run three times with the same weights but a
 *    different input, so each pass produces different K/V at the same position. A cache keyed by layer index
 *    would return the last pass's keys for all three. Visit index is therefore the identity.
 *
 * 2. CONV HISTORY. K and V each go through a 4-tap causal conv AFTER the projection, so producing K at
 *    position p needs the raw (pre-conv) K at p-1..p-3. Those raw rows are cached alongside the post-conv
 *    keys, or incremental decoding silently changes the first few keys of every step.
 *
 * Sliding layers never attend beyond `window` keys, so their rows live in a ring; full layers grow with the
 * context and are stored directly.
 *
 * C11 only, no allocation on the read path, no global state.
 */
#include "wubu_inkling2_kv.h"
#include "wubu_kv_q8.h"
#include "wubu_tqvec.h"
#include <stdio.h>
#include <math.h>

#include <stdlib.h>
#include <string.h>
#include <math.h>

static int in2_kv_cap(const in2_kv_t *kv, int visit)
{
    return kv->cap ? kv->cap[visit] : kv->n_pos_cap;
}

static size_t in2_kv_off(const in2_kv_t *kv, int visit)
{
    return kv->off ? kv->off[visit] : (size_t)visit * (size_t)kv->n_pos_cap;
}

/* Ring within the visit's own capacity. A full-attention visit is sized to the context, so pos % cap == pos for every
 * position that exists; a sliding visit wraps inside its window. Uniform-capacity callers therefore behave exactly as
 * before, which is why this change cannot alter a single key. */
static int in2_kv_row_slot(const in2_kv_t *kv, int visit, int pos)
{
    const int cap = in2_kv_cap(kv, visit);
    if (cap <= 0) return pos;
    return pos % cap;
}

int in2_kv_init(in2_kv_t *kv, int n_visit, int dim, int window, int n_pos_cap)
{
    if (!kv || n_visit <= 0 || dim <= 0 || n_pos_cap <= 0) return -1;
    int *rows = (int *)malloc((size_t)n_visit * sizeof(int));
    if (!rows) return -1;
    /* A window makes the ring the window, not the cap: the cap is only an upper bound on positions. That distinction
     * is the whole reason kv_selftest can store 10 positions into a 4-row ring. */
    for (int v = 0; v < n_visit; ++v) rows[v] = (window > 0 && window < n_pos_cap) ? window : n_pos_cap;
    const int rc = in2_kv_init_pervisit(kv, n_visit, dim, window, rows);
    if (rc == 0) kv->n_pos_max = n_pos_cap;
    free(rows);
    return rc;
}

size_t in2_kv_bytes(const in2_kv_t *kv)
{
    if (!kv || !kv->cap) return 0;
    size_t rows = 0;
    for (int v = 0; v < kv->n_visit; ++v) rows += (size_t)kv->cap[v];
    if (kv->tq) return rows * kv->rowq * 2u + rows * sizeof(float) * 2u;  /* codes + per-row norms */
    if (kv->q8) return rows * kv->rowq * 2u;
    return rows * (size_t)kv->dim * sizeof(float) * 2u;
}

int in2_kv_init_pervisit(in2_kv_t *kv, int n_visit, int dim, int window, const int *rows)
{
    if (!kv || n_visit <= 0 || dim <= 0) return -1;
    memset(kv, 0, sizeof *kv);
    kv->n_visit = n_visit;
    kv->dim = dim;
    kv->window = window;

    kv->cap = (int *)malloc((size_t)n_visit * sizeof(int));
    kv->off = (size_t *)malloc((size_t)n_visit * sizeof(size_t));
    if (!kv->cap || !kv->off) { in2_kv_free(kv); return -1; }

    size_t total = 0;
    int maxcap = 0;
    for (int v = 0; v < n_visit; ++v) {
        /* NULL rows means the old uniform behaviour: every visit gets the same capacity (passed as n_pos_cap by
         * in2_kv_init, which forwards here with rows == NULL and cap 0 - see below). */
        kv->cap[v] = rows ? rows[v] : 0;
        if (kv->cap[v] <= 0) { free(kv->cap); free(kv->off); memset(kv, 0, sizeof *kv); return -1; }
        kv->off[v] = total;
        total += (size_t)kv->cap[v];
        if (kv->cap[v] > maxcap) maxcap = kv->cap[v];
    }
    kv->n_pos_cap = maxcap;
    kv->n_pos_max = maxcap;

    /* The cache is one allocation laid out visit by visit at the offsets just computed. Sizing it from the sum of the
     * per-visit rows is the entire point: a 1M context over 8 full visits plus 64 sliding visits costs
     * 8 x 1M x 1024 x 8 B = 68.7 GB in f32, against 603 GB for the uniform layout that in2_kv_init used to build. */
    const size_t elems = total * (size_t)dim;
    /* TurboQuant tier: 4-8 bits/coord, half the memory of q8_0 at 4 bits. Set q8=1 as well so every existing
     * "quantised" guard keeps working; tq != NULL is what distinguishes the two formats. */
    const char *tqe = getenv("IN2_KV_TQ");
    const size_t nrows = elems / (size_t)dim;
    if (tqe) {
        int bits = atoi(tqe);
        if (bits < 2) bits = 4;
        if (bits > 8) bits = 8;
        /* OPEN DEFECT: b=5 and b=8 produce wrong output in the ENGINE while the module round-trip is correct and
         * monotone at every width (worst max-err 0.002 at b=8, d=1024, 300 structured rows). Refuse rather than
         * emit wrong tokens; b=4/6/7 are verified to answer correctly. */
        /* OPEN DEFECT: b=5 and b=8 produce wrong output in the ENGINE while the module round-trip is correct and
         * monotone at every width (worst max-err 0.002 at b=8, d=1024, 300 structured rows). Clamp to a verified
         * width once, loudly, instead of refusing: refusing here crashed because init is called per visit. */
        if (bits == 5 || bits == 8) {
            static int warned = 0;
            const int clamped = (bits == 5) ? 6 : 7;
            if (!warned) {
                warned = 1;
                fprintf(stderr, "[IN2_KV_TQ] %d bits is known-broken in-engine (module is fine, wiring is not) - clamping to %d\n", bits, clamped);
            }
            bits = clamped;
        }
        kv->tq = wubu_tqvec_init(dim, bits, 0xC0FFEEull);
        if (!kv->tq) { in2_kv_free(kv); return -1; }
        kv->q8   = 1;
        kv->rowq = wubu_tqvec_code_bytes(kv->tq);
        kv->kq   = (unsigned char *)calloc(1, nrows * kv->rowq);
        kv->vq   = (unsigned char *)calloc(1, nrows * kv->rowq);
        kv->kn   = (float *)calloc(nrows, sizeof(float));
        kv->vn   = (float *)calloc(nrows, sizeof(float));
        kv->encsc = (float *)malloc((size_t)dim * sizeof(float));
        kv->kscr = (float *)malloc((size_t)dim * sizeof(float));
        kv->vscr = (float *)malloc((size_t)dim * sizeof(float));
        if (!kv->kq || !kv->vq || !kv->kn || !kv->vn || !kv->encsc || !kv->kscr || !kv->vscr) { in2_kv_free(kv); return -1; }
    } else if (getenv("IN2_KV_Q8")) {
        kv->q8 = 1;
        kv->rowq = kvq8_row_bytes(dim);
        kv->kq   = (unsigned char *)calloc(1, elems / (size_t)dim * kv->rowq);
        kv->vq   = (unsigned char *)calloc(1, elems / (size_t)dim * kv->rowq);
        kv->kscr = (float *)malloc((size_t)dim * sizeof(float));
        kv->vscr = (float *)malloc((size_t)dim * sizeof(float));
        if (!kv->kq || !kv->vq || !kv->kscr || !kv->vscr) { in2_kv_free(kv); return -1; }
    } else {
        kv->k    = (float *)calloc(elems, sizeof(float));
        kv->v    = (float *)calloc(elems, sizeof(float));
    }
    kv->kraw = (float *)calloc((size_t)n_visit * 3u * (size_t)dim, sizeof(float));
    kv->vraw = (float *)calloc((size_t)n_visit * 3u * (size_t)dim, sizeof(float));
    if ((!kv->q8 && (!kv->k || !kv->v)) || !kv->kraw || !kv->vraw) { in2_kv_free(kv); return -1; }
    kv->n_stored = 0;
    return 0;
}

void in2_kv_free(in2_kv_t *kv)
{
    if (!kv) return;
    free(kv->k); free(kv->v); free(kv->kraw); free(kv->vraw);
    free(kv->kq); free(kv->vq); free(kv->kscr); free(kv->vscr); free(kv->cap); free(kv->off);
    wubu_tqvec_free(kv->tq); free(kv->kn); free(kv->vn); free(kv->encsc);
    memset(kv, 0, sizeof *kv);
}

float *in2_kv_k(in2_kv_t *kv, int visit, int pos)
{
    if (!kv || (!kv->k && !kv->q8) || visit < 0 || visit >= kv->n_visit || pos < 0 || pos >= kv->n_pos_max) return NULL;
    const size_t el = in2_kv_off(kv, visit) + (size_t)in2_kv_row_slot(kv, visit, pos);
    if (kv->tq) {
        wubu_tqvec_decode(kv->tq, kv->kq + el * kv->rowq, kv->kscr);
        const float s = kv->kn ? kv->kn[el] : 1.0f;
        for (int i = 0; i < kv->dim; ++i) kv->kscr[i] *= s;
        return kv->kscr;
    }
    if (kv->q8) {
        kvq8_dequantize_row(kv->kq + el * kv->rowq, kv->kscr, kv->dim);
        return kv->kscr;   /* read-only, valid until the next in2_kv_k() call - no caller holds two K rows */
    }
    return kv->k + el * (size_t)kv->dim;
}

float *in2_kv_v(in2_kv_t *kv, int visit, int pos)
{
    if (!kv || (!kv->v && !kv->q8) || visit < 0 || visit >= kv->n_visit || pos < 0 || pos >= kv->n_pos_max) return NULL;
    const size_t el = in2_kv_off(kv, visit) + (size_t)in2_kv_row_slot(kv, visit, pos);
    if (kv->tq) {
        wubu_tqvec_decode(kv->tq, kv->vq + el * kv->rowq, kv->vscr);
        const float s = kv->vn ? kv->vn[el] : 1.0f;
        for (int i = 0; i < kv->dim; ++i) kv->vscr[i] *= s;
        return kv->vscr;
    }
    if (kv->q8) {
        kvq8_dequantize_row(kv->vq + el * kv->rowq, kv->vscr, kv->dim);
        return kv->vscr;
    }
    return kv->v + el * (size_t)kv->dim;
}

void in2_kv_store_post_k(in2_kv_t *kv, int visit, int pos, const float *row)
{
    if (!kv || (!kv->k && !kv->q8) || !row || visit < 0 || visit >= kv->n_visit || pos < 0 || pos >= kv->n_pos_max) return;
    const size_t el = in2_kv_off(kv, visit) + (size_t)in2_kv_row_slot(kv, visit, pos);
    if (kv->tq) {
        double n2 = 0.0;
        for (int i = 0; i < kv->dim; ++i) n2 += (double)row[i] * (double)row[i];
        const float nrm = (float)sqrt(n2);
        kv->kn[el] = nrm > 0.0f ? nrm : 1.0f;
        for (int i = 0; i < kv->dim; ++i) kv->encsc[i] = row[i] / kv->kn[el];
        wubu_tqvec_encode(kv->tq, kv->encsc, kv->kq + el * kv->rowq);
        if (getenv("IN2_TQ_DBG")) {
            static int once = 0;
            if (!once) {
                once = 1;
                float *chk = (float *)malloc((size_t)kv->dim * sizeof(float));
                wubu_tqvec_decode(kv->tq, kv->kq + el * kv->rowq, chk);
                double mx = 0.0, nr = 0.0;
                for (int i = 0; i < kv->dim; ++i) {
                    const double a = chk[i] * (double)kv->kn[el];
                    const double dd = fabs(a - (double)row[i]);
                    if (dd > mx) mx = dd;
                    nr += (double)row[i] * (double)row[i];
                }
                fprintf(stderr, "[TQDBG] dim=%d bits=%d rowq=%zu |row|=%.4f norm_stored=%.4f max_roundtrip_err=%.6f\n",
                        kv->dim, wubu_tqvec_bits(kv->tq), kv->rowq, sqrt(nr), (double)kv->kn[el], mx);
                free(chk);
            }
        }
        return;
    }
    if (kv->q8) { kvq8_quantize_row(row, kv->kq + el * kv->rowq, kv->dim); return; }
    float *dst = kv->k + el * (size_t)kv->dim;
    memcpy(dst, row, (size_t)kv->dim * sizeof(float));
}

void in2_kv_store_post_v(in2_kv_t *kv, int visit, int pos, const float *row)
{
    if (!kv || (!kv->v && !kv->q8) || !row || visit < 0 || visit >= kv->n_visit || pos < 0 || pos >= kv->n_pos_max) return;
    const size_t el = in2_kv_off(kv, visit) + (size_t)in2_kv_row_slot(kv, visit, pos);
    if (kv->tq) {
        double n2 = 0.0;
        for (int i = 0; i < kv->dim; ++i) n2 += (double)row[i] * (double)row[i];
        const float nrm = (float)sqrt(n2);
        kv->vn[el] = nrm > 0.0f ? nrm : 1.0f;
        for (int i = 0; i < kv->dim; ++i) kv->encsc[i] = row[i] / kv->vn[el];
        wubu_tqvec_encode(kv->tq, kv->encsc, kv->vq + el * kv->rowq);
        return;
    }
    if (kv->q8) { kvq8_quantize_row(row, kv->vq + el * kv->rowq, kv->dim); return; }
    float *dst = kv->v + el * (size_t)kv->dim;
    memcpy(dst, row, (size_t)kv->dim * sizeof(float));
}

/* A sliding ring slot is only valid while it still holds the position asked for; once the window has moved
 * past it the slot belongs to a newer key. Callers use this to fall back instead of attending to stale rows. */
int in2_kv_resident(const in2_kv_t *kv, int pos)
{
    if (!kv || pos < 0 || pos >= kv->n_stored) return 0;
    /* A position is resident if it is still inside the ring of the visits that stored it - the sliding window for a
     * sliding visit, the whole context for a full one. kv->window is the sliding ring, so this stays conservative. */
    if (kv->window > 0 && (long) pos < (long) kv->n_stored - (long) kv->window) return 0;
    return 1;
}

/* Stored in ring order, so earlier conv history is simply the three previous stored raw rows. */
void in2_kv_store_raw(in2_kv_t *kv, int visit, const float *krow, const float *vrow)
{
    if (!kv || visit < 0 || visit >= kv->n_visit) return;
    float *kh = kv->kraw + (size_t)visit * 3u * (size_t)kv->dim;
    float *vh = kv->vraw + (size_t)visit * 3u * (size_t)kv->dim;
    memmove(kh, kh + kv->dim, (size_t)kv->dim * 2u * sizeof(float));
    memmove(vh, vh + kv->dim, (size_t)kv->dim * 2u * sizeof(float));
    memcpy(kh + (size_t)kv->dim * 2u, krow, (size_t)kv->dim * sizeof(float));
    memcpy(vh + (size_t)kv->dim * 2u, vrow, (size_t)kv->dim * sizeof(float));
}

const float *in2_kv_hist_k(const in2_kv_t *kv, int visit, int back)
{
    if (!kv || visit < 0 || visit >= kv->n_visit || back < 1 || back > 3) return NULL;
    return kv->kraw + (size_t)visit * 3u * (size_t)kv->dim + (size_t)(3 - back) * (size_t)kv->dim;
}

const float *in2_kv_hist_v(const in2_kv_t *kv, int visit, int back)
{
    if (!kv || visit < 0 || visit >= kv->n_visit || back < 1 || back > 3) return NULL;
    return kv->vraw + (size_t)visit * 3u * (size_t)kv->dim + (size_t)(3 - back) * (size_t)kv->dim;
}

/* One position of the 4-tap causal conv, using cached raw rows for the taps that reach into the past.
 *
 * in2_shortconv() only knows about the batch it is handed, so when decode processes a single new position the
 * first taps see zero padding and the keys silently change. This is the same arithmetic (tap 0 = current, then
 * the three cached rows, left-padded with zeros at the start of a sequence) applied one row at a time.
 *
 * hist3 is the base of a three-row window in time order (oldest first, like in2_kv_hist_*(.., 3)), so tap 1
 * reads hist3[(3-1)*channels + c] and tap 3 reads hist3[0 + c]. At the start of a sequence the unfilled slots are
 * the zeros calloc left there, which is exactly the left padding the batch version applies. */
void in2_kv_conv_row(const float *hist3, const float *xrow, int channels, const float *w, int k, float *out)
{
    const float scale = 1.0f / sqrtf((float)(k + 1));
    for (int c = 0; c < channels; ++c) {
        /* Accumulate in the SAME order as in2_shortconv - tap 0 (oldest) first, the current row last. The earlier
         * version seeded acc with the current row and then added taps 0..k-2, which is a different summation order
         * and therefore a few ULP different: kt differed from the batch conv by 3.8e-6 with kraw bit-exact, and the
         * recurrent loop turned that 1e-6 into ~1e-1 over 48 visits x 3 passes. Skipping a term whose source is
         * before the start of the sequence is also what the batch version does, so an unfilled hist slot (zeros)
         * contributes exactly nothing in both. */
        float acc = 0.0f;
        for (int tap = 0; tap < k; ++tap) {
            const int age = k - 1 - tap;                          /* 3, 2, 1, 0 for k = 4 */
            if (age == 0) { acc += w[(size_t)c * (size_t)k + (size_t)tap] * xrow[c]; continue; }
            if (!hist3 || age > 3) continue;
            acc += w[(size_t)c * (size_t)k + (size_t)tap] * hist3[(size_t)(3 - age) * (size_t)channels + (size_t)c];
        }
        out[c] = (acc + xrow[c]) * scale;
    }
}

/* Same idea as in2_kv_conv_row, for the full-attention layers: those use in2_shortconv_dynamic, whose tap set is
 * stat + a per-position projection of that position's own input. Only the input row is position-local, so the
 * per-row form needs the three cached history rows and that position's tap row - nothing else carries over. */
void in2_kv_conv_row_dyn(const float *hist3, const float *xrow, int channels, const float *stat,
                         const float *dwrow, int n_head, int k, float *out)
{
    const int hd = channels / n_head;
    const float scale = 1.0f / sqrtf((float)(2 * k + 1));
    for (int h = 0; h < n_head; ++h) {
        for (int d = 0; d < hd; ++d) {
            const int c = h * hd + d;
            /* last tap = current row; tap j has age (k-1-j); dyn taps are ordered newest-first */
            float acc = (stat[(size_t)c * (size_t)k + (size_t)(k - 1)] + dwrow[(size_t)h * (size_t)k]) * xrow[c];
            for (int j = 0; j < k - 1; ++j) {
                const int age = k - 1 - j;
                if (!hist3 || age > 3) continue;
                acc += (stat[(size_t)c * (size_t)k + (size_t)j] + dwrow[(size_t)h * (size_t)k + (size_t)age])
                     * hist3[(size_t)(3 - age) * (size_t)channels + (size_t)c];
            }
            out[c] = (acc + xrow[c]) * scale;
        }
    }
}

/* Independent pushes for the two streams, for callers that fill them at different times (the residual conv and
 * the MoE conv are computed at different points in a layer but share one history record). */
void in2_kv_store_row_k(in2_kv_t *kv, int visit, const float *row)
{
    if (!kv || visit < 0 || visit >= kv->n_visit) return;
    float *kh = kv->kraw + (size_t)visit * 3u * (size_t)kv->dim;
    memmove(kh, kh + kv->dim, (size_t)kv->dim * 2u * sizeof(float));
    memcpy(kh + (size_t)kv->dim * 2u, row, (size_t)kv->dim * sizeof(float));
}
void in2_kv_store_row_v(in2_kv_t *kv, int visit, const float *row)
{
    if (!kv || visit < 0 || visit >= kv->n_visit) return;
    float *vh = kv->vraw + (size_t)visit * 3u * (size_t)kv->dim;
    memmove(vh, vh + kv->dim, (size_t)kv->dim * 2u * sizeof(float));
    memcpy(vh + (size_t)kv->dim * 2u, row, (size_t)kv->dim * sizeof(float));
}

void in2_kv_tick(in2_kv_t *kv)
{
    if (kv) ++kv->n_stored;
}
