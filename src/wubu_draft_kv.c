/*
 * wubu_draft_kv.c -- EG-43 K4V4 draft KV. Pure C11.
 */
#include "wubu_draft_kv.h"

#include <math.h>

static void put_nib(uint8_t *dst, int i, unsigned v) {
    if ((i & 1) == 0) dst[i/2] = (uint8_t)((dst[i/2] & 0x0Fu) | (v << 4));
    else              dst[i/2] = (uint8_t)((dst[i/2] & 0xF0u) | (v & 0x0Fu));
}

static unsigned get_nib(const uint8_t *src, int i) {
    return (i & 1) == 0 ? (src[i/2] >> 4) : (src[i/2] & 0x0Fu);
}

static unsigned q4(float x, float scale) {
    int v = (int)roundf(x / scale) + 8; /* signed range [-8,7], zero point 8 */
    if (v < 0) v = 0;
    if (v > 15) v = 15;
    return (unsigned)v;
}

int wubu_draft_kv_pack(const float *k, const float *v, int rows, int d,
                       uint8_t *kq, uint8_t *vq,
                       float *k_scales, float *v_scales) {
    if (!k || !v || !kq || !vq || !k_scales || !v_scales ||
        rows <= 0 || d <= 0 || ((rows * d) & 1)) return -1;

    for (int j = 0; j < d; j++) {
        float mx = 0.0f;
        for (int i = 0; i < rows; i++) {
            float a = fabsf(k[i*d + j]);
            if (a > mx) mx = a;
        }
        k_scales[j] = mx > 0.0f ? mx / 7.0f : 1.0f;
    }
    for (int i = 0; i < rows; i++) {
        float mx = 0.0f;
        for (int j = 0; j < d; j++) {
            float a = fabsf(v[i*d + j]);
            if (a > mx) mx = a;
        }
        v_scales[i] = mx > 0.0f ? mx / 7.0f : 1.0f;
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < d; j++) {
            int p = i * d + j;
            put_nib(kq, p, q4(k[p], k_scales[j]));
            put_nib(vq, p, q4(v[p], v_scales[i]));
        }
    }
    return 0;
}

int wubu_draft_kv_unpack_k(const uint8_t *kq, int rows, int d,
                           const float *k_scales, float *out) {
    if (!kq || !k_scales || !out || rows <= 0 || d <= 0) return -1;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < d; j++) {
            int p = i * d + j;
            out[p] = ((float)get_nib(kq, p) - 8.0f) * k_scales[j];
        }
    return 0;
}

int wubu_draft_kv_unpack_v(const uint8_t *vq, int rows, int d,
                           const float *v_scales, float *out) {
    if (!vq || !v_scales || !out || rows <= 0 || d <= 0) return -1;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < d; j++) {
            int p = i * d + j;
            out[p] = ((float)get_nib(vq, p) - 8.0f) * v_scales[i];
        }
    return 0;
}
