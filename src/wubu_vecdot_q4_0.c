/* wubu_vecdot_q4_0.c -- fused Q4_0 x Q8_K dot product. Strict C11.
 *
 * Reference layout: ggml block_q4_0 (fp16 d + 16 packed nibbles per 32 elements),
 * ggml_vec_dot_q4_0_q8_0 for the kernel shape. Activations here are this repo's
 * Q8_K blocks (one fp16-free float d per 256, int8 qs, int16 bsums per 16).
 *
 * Math per 32-element sub-block b:
 *   q4 value v_j = nibble_j  (0..15); the block stores (v_j - 8)
 *   dot_b = d4_b * d8 * sum_j (v_j - 8) * q8_j
 *         = d4_b * d8 * ( sum_j v_j*q8_j - 8 * sum_j q8_j )
 * so the 8-bias is removed with the per-16 Q8_K sums instead of widening every
 * nibble before the multiply.
 */
#include "wubu_vecdot_q4_0.h"
#include "gguf_reader.h"

#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define QK_K 256
#define Q4_0_BLOCK_ELEMS 32
#define Q4_0_BLOCK_SIZE  18

typedef struct {
    uint16_t d;
    uint8_t  qs[16];
} wubu_block_q4_0;

int wubu_vec_dot_q4_0_q8_k_scalar(int n, float *s, const void *w, const void *q8v)
{
    const wubu_block_q4_0 *x = (const wubu_block_q4_0 *)w;
    const block_q8_K      *y = (const block_q8_K *)q8v;
    const int nb = n / QK_K;

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const int8_t *q8 = y[i].qs;
        float block_sum = 0.0f;
        for (int b = 0; b < QK_K / Q4_0_BLOCK_ELEMS; ++b) {
            const wubu_block_q4_0 *blk = &x[i * (QK_K / Q4_0_BLOCK_ELEMS) + b];
            const float d4 = gguf_f16_to_f32(blk->d);
            int acc = 0;
            for (int j = 0; j < 16; ++j) {
                const int lo = blk->qs[j] & 0x0f;
                const int hi = blk->qs[j] >> 4;
                /* elements 0..15 are the low nibbles, 16..31 the high nibbles */
                acc += (lo - 8) * q8[b * 32 + j];
                acc += (hi - 8) * q8[b * 32 + 16 + j];
            }
            block_sum += d4 * (float)acc;
        }
        sumf += block_sum * y[i].d;
    }
    *s = sumf;
    return 0;
}

#if defined(__AVX2__)

int wubu_vec_dot_q4_0_q8_k_avx2(int n, float *s, const void *w, const void *q8v)
{
    const wubu_block_q4_0 *x = (const wubu_block_q4_0 *)w;
    const block_q8_K      *y = (const block_q8_K *)q8v;
    const int nb = n / QK_K;

    const __m128i low_mask = _mm_set1_epi8(0x0f);
    const __m256i eight16 = _mm256_set1_epi16(8);

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const int8_t *q8 = y[i].qs;
        __m256 accf = _mm256_setzero_ps();
        for (int b = 0; b < QK_K / Q4_0_BLOCK_ELEMS; ++b) {
            const wubu_block_q4_0 *blk = &x[i * (QK_K / Q4_0_BLOCK_ELEMS) + b];
            const float d4 = gguf_f16_to_f32(blk->d);
            const __m128i packed = _mm_loadu_si128((const __m128i *)blk->qs);
            const __m128i lo = _mm_and_si128(packed, low_mask);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
            /* element order: lo[0..15] then hi[0..15] */
            const __m256i q4v = _mm256_set_m128i(hi, lo);
            const __m256i q8v = _mm256_loadu_si256((const __m256i *)(q8 + b * 32));
            /* widen both to int16 and do exact signed multiply-add pairs */
            const __m256i q4_lo16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(q4v));
            const __m256i q4_hi16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(q4v, 1));
            const __m256i q8_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8v));
            const __m256i q8_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8v, 1));
            const __m256i p_lo = _mm256_madd_epi16(
                _mm256_sub_epi16(q4_lo16, eight16), q8_lo16);
            const __m256i p_hi = _mm256_madd_epi16(
                _mm256_sub_epi16(q4_hi16, eight16), q8_hi16);
            const __m256i partial = _mm256_add_epi32(p_lo, p_hi);
            accf = _mm256_fmadd_ps(_mm256_set1_ps(d4), _mm256_cvtepi32_ps(partial), accf);
        }
        __m128 r = _mm256_extractf128_ps(accf, 1);
        r = _mm_add_ps(r, _mm256_castps256_ps128(accf));
        r = _mm_add_ps(r, _mm_movehl_ps(r, r));
        r = _mm_add_ss(r, _mm_movehdup_ps(r));
        sumf += _mm_cvtss_f32(r) * y[i].d;
    }
    *s = sumf;
    return 0;
}

#endif /* __AVX2__ */

int wubu_vec_dot_q4_0_q8_k(int n, float *s, const void *w, const void *q8)
{
    if (!s || !w || !q8 || n <= 0 || (n % QK_K) != 0) return -1;
#if defined(__AVX2__)
    return wubu_vec_dot_q4_0_q8_k_avx2(n, s, w, q8);
#else
    return wubu_vec_dot_q4_0_q8_k_scalar(n, s, w, q8);
#endif
}

void wubu_vec_dot_q4_0_q8_k_vd(int n, float *s, size_t bs, const void *vx,
                               size_t bx, const void *vy, size_t by, int nrc)
{
    (void)bs; (void)bx; (void)by;
    if (nrc != 1) { if (s) *s = 0.0f; return; }
    if (wubu_vec_dot_q4_0_q8_k(n, s, vx, vy) != 0 && s) *s = 0.0f;
}
