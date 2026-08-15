/* wubu_fp16.h — IEEE 754 half-precision (FP16) and BFloat16 conversion
 *
 * Self-contained header: include in any .c file that needs f16/bf16 conversion.
 * Replaces the 12+ independent definitions scattered across the codebase.
 *
 * Usage: #include "wubu_fp16.h"
 */

#ifndef WUBU_FP16_H
#define WUBU_FP16_H

#include <stdint.h>
#include <string.h>

/* ---- FP16 (IEEE 754 half-precision) ----
 * Format: 1 sign | 5 exponent | 10 mantissa
 * Bias: 15 */

static inline float wubu_f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) {
            f32 = (sign << 31);  /* zero */
        } else {
            /* subnormal: renormalize */
            exp = 1;
            while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FF;
            f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | (0xFF << 23) | (mant << 13);  /* inf/NaN */
    } else {
        f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f32, 4);
    return result;
}

static inline uint16_t wubu_f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp  = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;
    uint16_t h;
    if (exp > 15) {
        h = (uint16_t)((sign << 15) | 0x7C00);  /* inf */
    } else if (exp < -14) {
        h = (uint16_t)(sign << 15);  /* zero (or subnormal loses precision) */
    } else if (exp < -13) {
        /* subnormal */
        mant |= 0x800000;
        int shift = -1 - exp;
        h = (uint16_t)((sign << 15) | (mant >> (13 + shift)));
    } else {
        h = (uint16_t)((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
    }
    return h;
}

/* ---- BFloat16 ----
 * Format: 1 sign | 8 exponent | 7 mantissa (truncated FP32) */

static inline float wubu_bf16_to_f32(uint16_t h) {
    uint32_t f32 = ((uint32_t)h) << 16;
    float result;
    memcpy(&result, &f32, 4);
    return result;
}

static inline uint16_t wubu_f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

#endif /* WUBU_FP16_H */
