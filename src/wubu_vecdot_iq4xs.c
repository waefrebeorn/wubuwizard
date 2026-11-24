/* wubu_vecdot_iq4xs.c -- fused IQ4_XS x Q8_K dot product. Strict C11.
 *
 * 4 IQ4_XS tensors in the 12B Gemma-4 IQ2 GGUF were costing 4.8 ms per matvec on
 * the generic path (measured with WUBU_GPU_STATS), because that path rebuilds the
 * 4-bit table lookup per block. This kernel does the same math with one shuffle
 * per 32 values plus maddubs/madd, and keeps the per-32 scale from a 6-bit
 * (scales_l nibble + scales_h high bit) pair.
 *
 * Layout: block_iq4_xs = fp16 d(2) + uint16 scales_h + uint8 scales_l[4] + qs[128]
 * = 136 bytes / 256 elements. Reference: ggml_vec_dot_iq4_xs_q8_K(_generic).
 */
#include "wubu_vecdot_iq4xs.h"
#include "gguf_reader.h"

#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define QK_K 256

typedef struct {
    uint16_t d;
    uint16_t scales_h;
    uint8_t  scales_l[4];
    uint8_t  qs[128];
} wubu_block_iq4_xs;

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

int wubu_vec_dot_iq4_xs_q8_k_scalar(int n, float *s, const void *w, const void *q8v)
{
    const wubu_block_iq4_xs *x = (const wubu_block_iq4_xs *)w;
    const block_q8_K         *y = (const block_q8_K *)q8v;
    const int nb = n / QK_K;

    float sumf = 0.0f;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = gguf_f16_to_f32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        const uint8_t *qs = x[ibl].qs;
        const int8_t  *q8 = y[ibl].qs;
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const uint8_t ls1 = (uint8_t)((x[ibl].scales_l[ib / 2] & 0xf) | ((h << 4) & 0x30));
            const uint8_t ls2 = (uint8_t)((x[ibl].scales_l[ib / 2] >> 4) | ((h << 2) & 0x30));
            h >>= 4;
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j + 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j + 16] * kvalues_iq4nl[qs[j] >> 4];
            }
            sumf += d4d8 * (float)(ls1 - 32) * (float)(sumi1 + sumi2);
            qs += 16; q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j + 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j + 16] * kvalues_iq4nl[qs[j] >> 4];
            }
            sumf += d4d8 * (float)(ls2 - 32) * (float)(sumi1 + sumi2);
            qs += 16; q8 += 32;
        }
    }
    *s = sumf;
    return 0;
}

#if defined(__AVX2__)

static inline __m256i iq4_mul_add_epi8(const __m256i a, const __m256i b)
{
    const __m256i ax = _mm256_sign_epi8(a, a);
    const __m256i sy = _mm256_sign_epi8(b, a);
    return _mm256_maddubs_epi16(ax, sy);
}

static inline __m256i iq4_set_m128i(__m128i hi, __m128i lo)
{
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static inline float iq4_hsum8(const __m256 x)
{
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

int wubu_vec_dot_iq4_xs_q8_k_avx2(int n, float *s, const void *w, const void *q8v)
{
    const wubu_block_iq4_xs *x = (const wubu_block_iq4_xs *)w;
    const block_q8_K         *y = (const block_q8_K *)q8v;
    const int nb = n / QK_K;

    const __m128i values128 = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    __m256 accum = _mm256_setzero_ps();

    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t *qs = x[ibl].qs;
        const int8_t  *q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)qs); qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)qs); qs += 16;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q4b_1 = iq4_set_m128i(
                _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
            const __m256i q4b_2 = iq4_set_m128i(
                _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
            const __m256i p16_1 = iq4_mul_add_epi8(q4b_1, q8b_1);
            const __m256i p16_2 = iq4_mul_add_epi8(q4b_2, q8b_2);
            const short ls1 = (short)(((x[ibl].scales_l[ib / 2] & 0xf) | ((sh << 4) & 0x30)) - 32);
            const short ls2 = (short)(((x[ibl].scales_l[ib / 2] >> 4) | ((sh << 2) & 0x30)) - 32);
            sh >>= 4;
            sumi1 = _mm256_add_epi32(_mm256_madd_epi16(p16_1, _mm256_set1_epi16(ls1)), sumi1);
            sumi2 = _mm256_add_epi32(_mm256_madd_epi16(p16_2, _mm256_set1_epi16(ls2)), sumi2);
        }
        const float d = gguf_f16_to_f32(x[ibl].d) * y[ibl].d;
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d),
                                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accum);
    }

    *s = iq4_hsum8(accum);
    return 0;
}

#endif /* __AVX2__ */

int wubu_vec_dot_iq4_xs_q8_k(int n, float *s, const void *w, const void *q8)
{
    if (!s || !w || !q8 || n <= 0 || (n % QK_K) != 0) return -1;
#if defined(__AVX2__)
    return wubu_vec_dot_iq4_xs_q8_k_avx2(n, s, w, q8);
#else
    return wubu_vec_dot_iq4_xs_q8_k_scalar(n, s, w, q8);
#endif
}

void wubu_vec_dot_iq4_xs_q8_k_vd(int n, float *s, size_t bs, const void *vx,
                                 size_t bx, const void *vy, size_t by, int nrc)
{
    (void)bs; (void)bx; (void)by;
    if (nrc != 1) { if (s) *s = 0.0f; return; }
    if (wubu_vec_dot_iq4_xs_q8_k(n, s, vx, vy) != 0 && s) *s = 0.0f;
}
