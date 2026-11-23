/*
 * wubu_specquant.c -- EG-63 three precision variants from one base. Pure C11.
 */
#include "wubu_specquant.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#define SQ4_BLOCK 8
#define SQ4_SCALES(n) (((n) + SQ4_BLOCK - 1) / SQ4_BLOCK)

size_t wubu_specquant_bytes(int variant, int n) {
    if (n < 0) return 0;
    if (variant == WUBU_SQ_INT4)
        return ((size_t)n + 1u) / 2u + (size_t)SQ4_SCALES(n) * sizeof(float);
    if (variant == WUBU_SQ_FP8) return (size_t)n;
    if (variant == WUBU_SQ_FP16) return (size_t)n * 2u;
    return 0;
}

static uint8_t f32_e4m3(float f) {
    uint32_t u, bits;
    int sign, e;
    float a = fabsf(f), m;
    memcpy(&u, &a, 4);
    memcpy(&bits, &f, 4);
    sign = (int)((bits >> 31) & 1u);
    if (a == 0.0f) return (uint8_t)(sign ? 0x80 : 0x00);
    e = (int)((u >> 23) & 0xffu) - 127;
    m = a / ldexpf(1.0f, e) - 1.0f;
    if (e < -6) return (uint8_t)sign << 7;
    if (e > 8) return (uint8_t)(sign ? 0x7e : 0x7f);
    {
        int mm = (int)(m * 8.0f + 0.5f);
        if (mm >= 8) { mm = 0; e++; }
        if (e > 8) return (uint8_t)(sign ? 0x7e : 0x7f);
        return (uint8_t)((sign << 7) | ((e + 7) << 3) | mm);
    }
}

static float e4m3_f32(uint8_t b) {
    int e = (b >> 3) & 0xfu, m = b & 7u;
    float v;
    if (e == 0) v = ldexpf((float)m / 8.0f, -6);
    else v = ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return (b & 0x80u) ? -v : v;
}

static uint16_t f32_f16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    {
        uint32_t sign = (u >> 16) & 0x8000u;
        int e = (int)((u >> 23) & 0xffu) - 127 + 15;
        uint32_t mant = (u >> 13) & 0x3ffu;
        if (e <= 0) return (uint16_t)sign;
        if (e >= 31) return (uint16_t)(sign | 0x7c00u);
        return (uint16_t)(sign | ((uint32_t)e << 10) | mant);
    }
}

static float f16_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int e = (h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)(h & 0x3ffu) << 13;
    uint32_t u;
    if (e == 0) return 0.0f;
    u = sign | ((uint32_t)(e - 15 + 127) << 23) | mant;
    {
        float f;
        memcpy(&f, &u, 4);
        return f;
    }
}

int wubu_specquant_pack(const float *w, int n, int variant,
                        unsigned char *dst, float *scale) {
    if (!w || !dst || !scale || n <= 0 ||
        wubu_specquant_bytes(variant, n) == 0) return -1;
    if (variant == WUBU_SQ_INT4) {
        size_t nibbles = ((size_t)n + 1u) / 2u;
        *scale = 1.0f;
        memset(dst, 0, wubu_specquant_bytes(variant, n));
        for (int b = 0; b < n; b += SQ4_BLOCK) {
            int end = b + SQ4_BLOCK > n ? n : b + SQ4_BLOCK;
            float absmax = 0.0f, sc;
            for (int i = b; i < end; i++) {
                float a = fabsf(w[i]);
                if (a > absmax) absmax = a;
            }
            sc = absmax > 0.0f ? absmax / 7.0f : 1.0f;
            memcpy(dst + nibbles + (size_t)(b / SQ4_BLOCK) * sizeof(float),
                   &sc, sizeof(float));
            for (int i = b; i < end; i++) {
                float q = w[i] / sc;
                int v = (int)(q >= 0.0f ? floorf(q + 0.5f) : ceilf(q - 0.5f));
                if (v > 7) v = 7;
                if (v < -7) v = -7;
                {
                    unsigned char nib = (unsigned char)(v & 0xf);
                    if ((i & 1) == 0) dst[i / 2] = nib << 4;
                    else dst[i / 2] |= nib;
                }
            }
        }
        return 0;
    }
    if (variant == WUBU_SQ_FP8) {
        *scale = 1.0f;
        for (int i = 0; i < n; i++) dst[i] = f32_e4m3(w[i]);
        return 0;
    }
    *scale = 1.0f;
    for (int i = 0; i < n; i++) {
        uint16_t h = f32_f16(w[i]);
        dst[i * 2] = (unsigned char)(h & 0xffu);
        dst[i * 2 + 1] = (unsigned char)(h >> 8);
    }
    return 0;
}

int wubu_specquant_unpack(const unsigned char *src, int n, int variant,
                          float scale, float *out) {
    if (!src || !out || n <= 0 || !(scale > 0.0f) ||
        wubu_specquant_bytes(variant, n) == 0) return -1;
    if (variant == WUBU_SQ_INT4) {
        size_t nibbles = ((size_t)n + 1u) / 2u;
        for (int i = 0; i < n; i++) {
            float sc;
            int nib = (i & 1) == 0 ? src[i / 2] >> 4 : src[i / 2] & 0xfu;
            memcpy(&sc, src + nibbles + (size_t)(i / SQ4_BLOCK) * sizeof(float),
                   sizeof(float));
            if (nib > 7) nib -= 16;
            out[i] = (float)nib * sc;
        }
        return 0;
    }
    if (variant == WUBU_SQ_FP8) {
        for (int i = 0; i < n; i++) out[i] = e4m3_f32(src[i]);
        return 0;
    }
    for (int i = 0; i < n; i++) {
        uint16_t h = (uint16_t)(src[i * 2] | ((uint16_t)src[i * 2 + 1] << 8));
        out[i] = f16_f32(h);
    }
    return 0;
}

int wubu_specquant_route(const float *q, int dim, float mid, float high) {
    float mean = 0.0f, var = 0.0f;
    if (!q || dim <= 0 || !(mid > 0.0f) || !(high > mid)) return -1;
    for (int i = 0; i < dim; i++) mean += q[i];
    mean /= (float)dim;
    for (int i = 0; i < dim; i++) var += (q[i] - mean) * (q[i] - mean);
    var = sqrtf(var / (float)dim);
    if (var < mid) return WUBU_SQ_INT4;
    if (var < high) return WUBU_SQ_FP8;
    return WUBU_SQ_FP16;
}
