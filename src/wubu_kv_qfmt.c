/*
 * wubu_kv_qfmt.c -- LP-11: reconstruct KV directly in compute format.
 * Strict C11, self-contained, opaque context. See include/wubu_kv_qfmt.h.
 *
 * Format
 * ------
 *   - 8 fixed FP16 anchors: {-6,-4,-2,-0.5,0.5,2,4,6}, decoded once to f32.
 *   - One E8M0 scale per block: byte e -> 2^(e-127) (MX style).
 *   - Each element: high nibble = nearest anchor index, low nibble = E2M1
 *     value (magnitudes {0,0.5,1,1.5,2,3,4,6}, sign bit in bit 3).
 *   - Reconstructed value = anchor_f32 + e2m1(nibble) * scale, computed in
 *     f32 and never re-rounded to fp16.
 */
#include <stdlib.h>
#include <string.h>

#include "wubu_kv_qfmt.h"

struct wubu_kv_qfmt {
    uint16_t anchor_bits[WUBU_KV_QFMT_NANCHORS];
    float anchor_f32[WUBU_KV_QFMT_NANCHORS];
};

static const uint16_t k_anchor_bits[WUBU_KV_QFMT_NANCHORS] = {
    0xC600u, /* -6   */
    0xC400u, /* -4   */
    0xC000u, /* -2   */
    0xB800u, /* -0.5 */
    0x3800u, /* +0.5 */
    0x4000u, /* +2   */
    0x4400u, /* +4   */
    0x4600u  /* +6   */
};

/* E2M1 magnitudes indexed by the low 3 bits of a nibble. */
static const float k_fp4_mag[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

/* ---- exact fp16 -> f32 conversion (no host fp16 type required) ---- */
static float half_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (uint32_t)(h & 0x7C00u) >> 10;
    uint32_t mant = (uint32_t)(h & 0x03FFu);
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FFu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float e8m0_to_scale(uint8_t e)
{
    uint32_t bits = (uint32_t)e << 23; /* 2^(e-127) in f32 encoding */
    float s;
    memcpy(&s, &bits, sizeof(s));
    return s;
}

wubu_kv_qfmt_t *wubu_kv_qfmt_create(void)
{
    wubu_kv_qfmt_t *q = (wubu_kv_qfmt_t *)calloc(1, sizeof(*q));
    if (!q) return NULL;
    for (int i = 0; i < WUBU_KV_QFMT_NANCHORS; i++) {
        q->anchor_bits[i] = k_anchor_bits[i];
        q->anchor_f32[i] = half_to_f32(k_anchor_bits[i]);
    }
    return q;
}

void wubu_kv_qfmt_free(wubu_kv_qfmt_t *q) { free(q); }

size_t wubu_kv_qfmt_packed_bytes(int n)
{
    return n < 0 ? 0u : (size_t)n;
}

float wubu_kv_qfmt_anchor(const wubu_kv_qfmt_t *q, int i)
{
    if (!q || i < 0 || i >= WUBU_KV_QFMT_NANCHORS) return 0.0f;
    return q->anchor_f32[i];
}

static int nearest_anchor(const wubu_kv_qfmt_t *q, float x)
{
    int best = 0;
    float bd = x - q->anchor_f32[0];
    if (bd < 0.0f) bd = -bd;
    for (int a = 1; a < WUBU_KV_QFMT_NANCHORS; a++) {
        float d = x - q->anchor_f32[a];
        if (d < 0.0f) d = -d;
        if (d < bd) { bd = d; best = a; }
    }
    return best;
}

static int nearest_mag_index(float v)
{
    int best = 0;
    float be = k_fp4_mag[0] > v ? k_fp4_mag[0] - v : v - k_fp4_mag[0];
    for (int i = 1; i < 8; i++) {
        float e = k_fp4_mag[i] > v ? k_fp4_mag[i] - v : v - k_fp4_mag[i];
        if (e < be) { be = e; best = i; }
    }
    return best;
}

int wubu_kv_qfmt_encode(const wubu_kv_qfmt_t *q, const float *src, int n,
                        uint8_t *packed, uint8_t *mx_exp)
{
    int e = 127, rc = -1;
    float scale = 1.0f, maxdev = 0.0f;
    int *ai;

    if (!q || !src || !packed || !mx_exp || n <= 0) goto done;
    ai = (int *)malloc((size_t)n * sizeof(int));
    if (!ai) goto done;

    /* Per-element nearest FP16 anchor -> small residual -> fine E8M0 scale. */
    for (int i = 0; i < n; i++) {
        float d;
        ai[i] = nearest_anchor(q, src[i]);
        d = src[i] - q->anchor_f32[ai[i]];
        if (d < 0.0f) d = -d;
        if (d > maxdev) maxdev = d;
    }

    /* Smallest E8M0 scale with 6*scale >= maxdev. */
    scale = 1.0f; e = 127;
    if (maxdev > 0.0f) {
        while (6.0f * scale < maxdev && e < 254) { scale *= 2.0f; e++; }
        while (e > 1 && 3.0f * scale >= maxdev) { scale *= 0.5f; e--; }
    }

    for (int i = 0; i < n; i++) {
        float r = (src[i] - q->anchor_f32[ai[i]]) / scale;
        int neg = r < 0.0f;
        float ar = neg ? -r : r;
        int mag = nearest_mag_index(ar);
        uint8_t nib = (uint8_t)((neg ? 8 : 0) | mag);
        packed[i] = (uint8_t)(((ai[i] & 0x0F) << 4) | (nib & 0x0F));
    }
    *mx_exp = (uint8_t)e;
    rc = 0;
    free(ai);

done:
    return rc;
}

static float fp4_from_nibble(uint8_t nib)
{
    float v = k_fp4_mag[nib & 7u];
    return (nib & 8u) ? -v : v;
}

int wubu_kv_qfmt_decode(const wubu_kv_qfmt_t *q, const uint8_t *packed,
                        uint8_t mx_exp, float *out, int n)
{
    float scale;
    if (!q || !packed || !out || n <= 0) return -1;
    if (mx_exp == 0 || mx_exp == 255) return -1;

    /* Validate the whole block first: never half-decode on corrupt input. */
    for (int i = 0; i < n; i++)
        if ((int)(packed[i] >> 4) >= WUBU_KV_QFMT_NANCHORS) return -1;

    scale = e8m0_to_scale(mx_exp);
    for (int i = 0; i < n; i++) {
        int ai = (int)(packed[i] >> 4);
        out[i] = q->anchor_f32[ai] + fp4_from_nibble((uint8_t)(packed[i] & 0x0Fu)) * scale;
    }
    return 0;
}

int wubu_kv_qfmt_dot(const wubu_kv_qfmt_t *q, const uint8_t *packed,
                     uint8_t mx_exp, const float *query, int n, float *out)
{
    float *tmp;
    double acc = 0.0;
    int rc = -1;

    if (!q || !packed || !query || !out || n <= 0) goto done;
    tmp = (float *)malloc((size_t)n * sizeof(float));
    if (!tmp) goto done;
    if (wubu_kv_qfmt_decode(q, packed, mx_exp, tmp, n) != 0) goto cleanup;
    for (int i = 0; i < n; i++) acc += (double)tmp[i] * (double)query[i];
    *out = (float)acc;
    rc = 0;

cleanup:
    free(tmp);
done:
    return rc;
}
