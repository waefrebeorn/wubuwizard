/* Activation quantisation for the i-quant row kernels, C11 only.
 *
 * Two activation formats are in play and they are NOT interchangeable:
 *   block_q8_K  - 256 elements, one float scale `d`, 256 int8 quants, 16 per-16 sums. Used by the K-quants and
 *                 the IQ2/IQ3/IQ1 families (ggml types 12,13,14,16,17,18,19,21,22,23,29).
 *   block_q8_0  - 32 elements, one FP16 scale `d`, 32 int8 quants. Used by Q8_0, Q2_0 and NVFP4 (types 8,42,40).
 * Feeding the wrong one produces plausible numbers, not a crash, so the kind is explicit in the API.
 *
 * Both mirror ggml's own ref quantisers constant for constant, because the kernels were translated 1:1 from the
 * functions that consume exactly these layouts - any difference here would silently break the bit-exactness the
 * translation was verified against.
 *
 * Quantise ONCE per matvec and pass the same buffer to every row of the expert: the amortisation over 2304 rows
 * is where the speedup comes from, not the kernel alone.
 */
#include "wubu_iq2q8.h"
#include <math.h>
#include <string.h>

/* IEEE-754 binary32 -> binary16, round to nearest even. Needed because block_q8_0 stores its scale as FP16. */
uint16_t in2_f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t expo = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;

    if (expo == 0xFFu) return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));   /* inf / nan */
    int32_t e = (int32_t)expo - 127 + 15;
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u);                             /* overflow -> inf */
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;                                       /* underflow -> zero */
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - e);
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (h & 1u))) h++;                          /* round to even */
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)e << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;                        /* may carry into exponent */
    return (uint16_t)(sign | h);
}

float in2_f16_to_f32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t expo = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t out;
    if (expo == 0) {
        if (mant == 0) { out = sign; }
        else {                                                                     /* subnormal */
            int32_t e = -1;
            do { mant <<= 1; e++; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            out = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (expo == 0x1F) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((expo - 15u + 127u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}

/* block_q8_K: 256 elements, scale convention taken from ggml (iscale = -128/maxv, d = 1/iscale) so a positive
 * extreme maps to negative quants and a negative scale; the product is unchanged but the bit pattern matches. */
void in2_q8k_quantise(const float *x, int64_t n_groups, in2_q8k_blk *out)
{
    for (int64_t b = 0; b < n_groups; ++b) {
        const float *xb = x + b * 256;
        in2_q8k_blk *y = &out[b];
        float amax = 0.0f, maxv = 0.0f;
        for (int j = 0; j < 256; ++j) {
            const float a = fabsf(xb[j]);
            if (a > amax) { amax = a; maxv = xb[j]; }
        }
        if (amax == 0.0f) { y->d = 0.0f; memset(y->qs, 0, 256); memset(y->bsums, 0, 32); continue; }
        const float iscale = -127.0f / maxv;   /* ggml uses 127, not 128: at 128 every |q| runs 128/127 too small
                                                * and d comes out 0.8% off, which the parity test caught */
        y->d = 1.0f / iscale;
        for (int j = 0; j < 256; ++j) {
            const float v = iscale * xb[j];
            int q = (int)(v < 0.0f ? v - 0.5f : v + 0.5f);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            y->qs[j] = (int8_t)q;
        }
        for (int g = 0; g < 16; ++g) {
            int s = 0;
            for (int j = 0; j < 16; ++j) s += y->qs[g * 16 + j];
            y->bsums[g] = (int16_t)s;
        }
    }
}

/* block_q8_0: 32 elements, FP16 scale d = amax/127 (ggml's convention), qs = round(x/d). */
void in2_q8_0_quantise(const float *x, int64_t n_groups, in2_q8_0_blk *out)
{
    for (int64_t b = 0; b < n_groups; ++b) {
        const float *xb = x + b * 32;
        in2_q8_0_blk *y = &out[b];
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) { const float a = fabsf(xb[j]); if (a > amax) amax = a; }
        const float d = amax / 127.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        y->d = in2_f32_to_f16(d);
        for (int j = 0; j < 32; ++j) {
            const float v = xb[j] * id;
            int q = (int)(v < 0.0f ? v - 0.5f : v + 0.5f);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            y->qs[j] = (int8_t)q;
        }
    }
}
