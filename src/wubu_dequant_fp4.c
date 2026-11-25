/*
 * wubu_dequant_fp4.c -- C11 dequantization for ggml types 39 (MXFP4), 40 (NVFP4)
 * and 42 (Q2_0). Strict C11, no dependencies beyond libm.
 *
 * Why this file was rewritten: the previous version decoded each nibble through a
 * four-entry magnitude table {0, 0.5, 1, 2}, which is the NVIDIA FP4 *code* set for
 * 2-bit values, not ggml's 16-entry E2M1 table. Any nibble >= 4 read past the end of
 * that table, the nibble order was even-element/high-nibble instead of
 * low-nibble/high-half, and UE4M3 was decoded with a 5-bit exponent and no 0.5
 * factor. Nothing in the GGUF path called it, so no measurement had caught it.
 *
 * Layouts and constants are taken verbatim from ggml (MIT):
 *   kvalues_fp4[16]        = 0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12
 *   ggml_ue4m3_to_fp32()   = 4-bit exponent (bias 7), 3-bit mantissa, x0.5
 *   GGML_E8M0_TO_FP32_HALF = 2^(e - 128)
 */
#include "wubu_dequant_fp4.h"

#include <stdint.h>
#include <string.h>

/* E2M1 magnitudes doubled: exactly ggml's kvalues_fp4. Index by 4-bit code. */
static const int8_t kvalues_fp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

/* fp16 -> fp32 without libm (d is a ggml_half in Q2_0 blocks). */
static float f16_to_f32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) return 0.0f;
        /* subnormal: normalise */
        int e = -1;
        do { ++e; man <<= 1; } while ((man & 0x400u) == 0);
        man &= 0x3FFu;
        bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/* Unsigned E4M3 scale, identical to ggml_ue4m3_to_fp32(). */
static float ue4m3_to_f32(uint8_t x)
{
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = (float)man / 512.0f;                  /* ldexpf(man, -9) */
    } else {
        const int e = exp - 7;
        const float base = 1.0f + (float)man / 8.0f;
        raw = (e >= 0) ? base * (float)(1u << e) : base / (float)(1u << -e);
    }
    return raw * 0.5f;
}

/* E8M0 shared exponent, identical to GGML_E8M0_TO_FP32_HALF. */
static float e8m0_to_f32(uint8_t e)
{
    const int ex = (int)e - 128;
    if (ex >= 0) return (float)(1u << (ex > 30 ? 30 : ex));
    return 1.0f / (float)(1u << (ex < -30 ? 30 : -ex));
}

long wubu_fp4_raw_size(int ggml_type, long n_elems)
{
    if (n_elems <= 0) return -1;
    switch (ggml_type) {
    case 39: return ((n_elems + 31) / 32) * 17;   /* MXFP4: 1 scale + 16 data   */
    case 40: return ((n_elems + 63) / 64) * 36;   /* NVFP4: 4 scales + 32 data  */
    case 42: return ((n_elems + 63) / 64) * 18;   /* Q2_0:  fp16 d + 16 data    */
    default: return -1;
    }
}

void dequantize_row_mxfp4(const unsigned char *data, float *output, long n_elems)
{
    const long nb = (n_elems + 31) / 32;
    for (long b = 0; b < nb; ++b) {
        const unsigned char *blk = data + b * 17;
        const float d = e8m0_to_f32(blk[0]);
        const unsigned char *qs = blk + 1;
        float *y = output + b * 32;
        for (long j = 0; j < 16; ++j) {
            if (b * 32 + j < n_elems) y[j] = (float)kvalues_fp4[qs[j] & 0x0F] * d;
            if (b * 32 + j + 16 < n_elems) y[j + 16] = (float)kvalues_fp4[qs[j] >> 4] * d;
        }
    }
}

void dequantize_row_nvfp4(const unsigned char *data, float *output, long n_elems)
{
    const long nb = (n_elems + 63) / 64;
    for (long b = 0; b < nb; ++b) {
        const unsigned char *blk = data + b * 36;
        const unsigned char *scales = blk;       /* 4 UE4M3, one per 16 elements */
        const unsigned char *qs = blk + 4;       /* 32 bytes = 64 codes          */
        for (int s = 0; s < 4; ++s) {
            const float d = ue4m3_to_f32(scales[s]);
            const unsigned char *sub = qs + s * 8;
            float *y = output + b * 64 + s * 16;
            for (long j = 0; j < 8; ++j) {
                const long i0 = b * 64 + s * 16 + j;
                const long i1 = i0 + 8;
                if (i0 < n_elems) y[j] = (float)kvalues_fp4[sub[j] & 0x0F] * d;
                if (i1 < n_elems) y[j + 8] = (float)kvalues_fp4[sub[j] >> 4] * d;
            }
        }
    }
}

/* Fused Q2_0 dot: never builds the float row.
 *
 * dequantize_row_q2_0 writes 64 floats per block and the caller then reads all 64 back to dot them, so the
 * value never leaves a 256-byte buffer. Here the 2-bit quants are unpacked and multiplied against x directly:
 * value = (q - 1) * d, so a block contributes d * (sum q_j x_j - sum x_j), which is one pass over x and the
 * quant bytes with no intermediate store. Same arithmetic, same order of accumulation per element, so the
 * result is the reference result up to float rounding - which is what the test in tools/dot_test.c checks. */
void dot_row_q2_0(const unsigned char *data, const float *x, long n_elems, float *out)
{
    const long nb = (n_elems + 63) / 64;
    double acc = 0.0;
    for (long b = 0; b < nb; ++b) {
        const unsigned char *blk = data + b * 18;
        uint16_t d_bits = 0;
        memcpy(&d_bits, blk, 2);
        const float d = f16_to_f32(d_bits);
        const unsigned char *qs = blk + 2;
        const float *xb = x + b * 64;
        double sum = 0.0;
        const long lim = (b * 64 + 64 <= n_elems) ? 64 : (n_elems - b * 64);
        for (long j = 0; j < lim; ++j) {
            const unsigned char q = (unsigned char)((qs[j >> 2] >> ((j & 3) * 2)) & 0x03u);
            sum += (double)((int)q - 1) * (double)xb[j];
        }
        acc += (double)d * sum;
    }
    *out = (float)acc;
}

void dequantize_row_q2_0(const unsigned char *data, float *output, long n_elems)
{
    const long nb = (n_elems + 63) / 64;
    for (long b = 0; b < nb; ++b) {
        const unsigned char *blk = data + b * 18;
        uint16_t d_bits = 0;
        memcpy(&d_bits, blk, 2);
        const float d = f16_to_f32(d_bits);
        const unsigned char *qs = blk + 2;
        for (long j = 0; j < 64; ++j) {
            if (b * 64 + j >= n_elems) break;
            const unsigned char q = (unsigned char)((qs[j >> 2] >> ((j & 3) * 2)) & 0x03u);
            output[b * 64 + j] = (float)((int)q - 1) * d;   /* 00=-1 01=0 10=+1 11=+2 */
        }
    }
}
