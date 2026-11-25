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

#include <stdlib.h>
#include <string.h>
#include <math.h>

static int in2_kv_row_slot(const in2_kv_t *kv, int pos)
{
    if (kv->window > 0) return pos % kv->window;
    return pos;
}

int in2_kv_init(in2_kv_t *kv, int n_visit, int dim, int window, int n_pos_cap)
{
    if (!kv || n_visit <= 0 || dim <= 0 || n_pos_cap <= 0) return -1;
    memset(kv, 0, sizeof *kv);
    kv->n_visit = n_visit;
    kv->dim = dim;
    kv->window = window;
    kv->n_pos_cap = n_pos_cap;

    /* sliding layers only ever hold `window` rows; full layers hold the whole context. Callers pass a window
     * of 0 for a full layer and the same n_pos_cap in both cases, so the cap is authoritative here. */
    const size_t rows = (size_t)n_pos_cap;
    const size_t elems = rows * (size_t)dim;
    kv->k    = (float *)calloc(elems * (size_t)n_visit, sizeof(float));
    kv->v    = (float *)calloc(elems * (size_t)n_visit, sizeof(float));
    kv->kraw = (float *)calloc((size_t)n_visit * 3u * (size_t)dim, sizeof(float));
    kv->vraw = (float *)calloc((size_t)n_visit * 3u * (size_t)dim, sizeof(float));
    if (!kv->k || !kv->v || !kv->kraw || !kv->vraw) { in2_kv_free(kv); return -1; }
    kv->n_stored = 0;
    return 0;
}

void in2_kv_free(in2_kv_t *kv)
{
    if (!kv) return;
    free(kv->k); free(kv->v); free(kv->kraw); free(kv->vraw);
    memset(kv, 0, sizeof *kv);
}

float *in2_kv_k(in2_kv_t *kv, int visit, int pos)
{
    if (!kv || !kv->k || visit < 0 || visit >= kv->n_visit || pos < 0 || pos >= kv->n_pos_cap) return NULL;
    return kv->k + ((size_t)visit * (size_t)kv->n_pos_cap + (size_t)in2_kv_row_slot(kv, pos)) * (size_t)kv->dim;
}

float *in2_kv_v(in2_kv_t *kv, int visit, int pos)
{
    if (!kv || !kv->v || visit < 0 || visit >= kv->n_visit || pos < 0 || pos >= kv->n_pos_cap) return NULL;
    return kv->v + ((size_t)visit * (size_t)kv->n_pos_cap + (size_t)in2_kv_row_slot(kv, pos)) * (size_t)kv->dim;
}

/* A sliding ring slot is only valid while it still holds the position asked for; once the window has moved
 * past it the slot belongs to a newer key. Callers use this to fall back instead of attending to stale rows. */
int in2_kv_resident(const in2_kv_t *kv, int pos)
{
    if (!kv || pos < 0 || pos >= kv->n_stored) return 0;
    if (kv->window > 0 && pos < kv->n_stored - kv->window) return 0;
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
        float acc = w[(size_t)c * (size_t)k] * xrow[c];            /* tap 0 = current */
        for (int tap = 1; tap < k; ++tap) {
            if (!hist3 || tap > 3) continue;
            acc += w[(size_t)c * (size_t)k + (size_t)tap] * hist3[(size_t)(3 - tap) * (size_t)channels + (size_t)c];
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
            float acc = (stat[(size_t)c * (size_t)k] + dwrow[(size_t)h * (size_t)k]) * xrow[c];
            for (int j = 1; j < k; ++j) {
                if (!hist3 || j > 3) continue;
                acc += (stat[(size_t)c * (size_t)k + (size_t)j] + dwrow[(size_t)h * (size_t)k + (size_t)j])
                     * hist3[(size_t)(3 - j) * (size_t)channels + (size_t)c];
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
