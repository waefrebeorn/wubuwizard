/*
 * wubu_fp8_kv.c -- EG-36a OCP FP8 KV storage. Pure C11.
 */
#include "wubu_fp8_kv.h"

#include "wubu_fp8.h"

#include <math.h>

/* E4M3's largest finite magnitude is 448. Scaling by it keeps the whole row
 * in range with headroom; the codec does its own saturating conversion. */
int wubu_fp8_kv_pack(const float *src, int n, uint8_t *out, float *scale) {
    float mx = 0.0f;
    if (!src || !out || !scale || n <= 0) return -1;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > mx) mx = a;
    }
    *scale = mx > 0.0f ? mx / 448.0f : 1.0f;
    for (int i = 0; i < n; i++)
        out[i] = wubu_fp8_e4m3_from_f32(src[i] / *scale);
    return 0;
}

int wubu_fp8_kv_unpack(const uint8_t *packed, int n, float scale, float *out) {
    if (!packed || !out || n <= 0 || !(scale > 0.0f)) return -1;
    for (int i = 0; i < n; i++)
        out[i] = wubu_fp8_e4m3_to_f32(packed[i]) * scale;
    return 0;
}
