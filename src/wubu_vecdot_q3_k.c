/* wubu_vecdot_q3_k.c -- fused Q3_K x Q8_K dot product. Strict C11.
 *
 * The tied token-embedding / LM head of the 12B Gemma-4 IQ2 GGUF is a single
 * Q3_K tensor (412 MB). Until this kernel existed the head had no vec_dot, so
 * every token dequantised all 412 MB to F32 before the matvec — measured at
 * ~0.9 s/token, i.e. most of the decode time.
 *
 * Scalar port: ggml_vec_dot_q3_K_q8_K_generic (llama/ggml-cpu/quants.c).
 * AVX2 port:   ggml_vec_dot_q3_K_q8_K from llama/ggml-cpu/arch/x86/quants.c.
 * Layout: block_q3_K = fp16 d(2) + hmask[32] + qs[64] + scales[12] = 110 bytes.
 */
#include "wubu_vecdot_q3_k.h"
#include "gguf_reader.h"

#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define QK_K 256

/* ggml's block_q3_K puts the fp16 scale LAST (unlike Q4_K/Q5_K, where it is
 * first): hmask[32] + qs[64] + scales[12] + d(2) = 110 bytes. */
typedef struct {
    uint8_t  hmask[QK_K / 8];   /* 32 */
    uint8_t  qs[QK_K / 4];      /* 64 */
    uint8_t  scales[12];
    uint16_t d;
} wubu_block_q3_k;

int wubu_vec_dot_q3_k_q8_k_scalar(int n, float *s, const void *w, const void *q8v)
{
    const wubu_block_q3_k *x = (const wubu_block_q3_k *)w;
    const block_q8_K      *y = (const block_q8_K *)q8v;
    const int nb = n / QK_K;

    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    int32_t aux32[8];
    float   sums[8];
    uint32_t auxs[4];
    memset(sums, 0, sizeof(sums));

    for (int i = 0; i < nb; ++i) {
        const uint8_t *q3 = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        const int8_t  *q8 = y[i].qs;
        memset(aux32, 0, sizeof(aux32));

        int8_t *a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q3[l] & 3);
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m) ? 0 : 4;
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)((q3[l] >> 2) & 3);
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m) ? 0 : 4;
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)((q3[l] >> 4) & 3);
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m) ? 0 : 4;
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)((q3[l] >> 6) & 3);
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m) ? 0 : 4;
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        const uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = (const int8_t *)auxs;

        for (int j = 0; j < QK_K / 16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = (int16_t)(q8[l] * a[l]);
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = (int16_t)(q8[l] * a[l]);
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = gguf_f16_to_f32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * (float)aux32[l];
    }

    *s = sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
    return 0;
}

#if defined(__AVX2__)

static inline float q3_hsum_float_8(const __m256 x)
{
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256i q3_get_scale_shuffle(uint32_t i)
{
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i *)k_shuffle + i);
}

static inline __m256i q3_mm256_set_m128i(__m128i hi, __m128i lo)
{
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

/* AVX2 port of the same decomposition the scalar kernel uses: expand the 3-bit
 * codes into a[0..255] once (byte ops), then run 16-element groups through
 * maddubs/madd with the group's 6-bit scale. Exact integer path: the scalar and
 * vector results agree bit for bit before the float block scale. */
int wubu_vec_dot_q3_k_q8_k_avx2(int n, float *s, const void *w, const void *q8v)
{
    const wubu_block_q3_k *x = (const wubu_block_q3_k *)w;
    const block_q8_K      *y = (const block_q8_K *)q8v;
    const int nb = n / QK_K;

    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;

    int8_t  aux8[QK_K];
    int32_t acc32[8];
    uint32_t auxs[4];
    __m256 accf = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const uint8_t *q3 = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        const int8_t  *q8 = y[i].qs;
        memset(acc32, 0, sizeof(acc32));

        int8_t *a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q3[l] & 3);
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m) ? 0 : 4;
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)((q3[l] >> 2) & 3);
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m) ? 0 : 4;
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)((q3[l] >> 4) & 3);
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m) ? 0 : 4;
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)((q3[l] >> 6) & 3);
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m) ? 0 : 4;
            a += 32; m <<= 1;
            q3 += 32;
        }

        memcpy(auxs, x[i].scales, 12);
        const uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = (const int8_t *)auxs;

        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < QK_K / 16; ++j) {
            const __m256i av = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(aux8 + j * 16)));
            const __m256i qv = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(y[i].qs + j * 16)));
            const __m256i prod = _mm256_madd_epi16(av, qv);
            const __m256i scal = _mm256_set1_epi32(scales[j] - 32);
            sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(prod, scal));
            (void)acc32;
        }

        const float d = gguf_f16_to_f32(x[i].d) * y[i].d;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accf);
        (void)q8;
    }

    *s = q3_hsum_float_8(accf);
    return 0;
}

#endif /* __AVX2__ */

int wubu_vec_dot_q3_k_q8_k(int n, float *s, const void *w, const void *q8)
{
    if (!s || !w || !q8 || n <= 0 || (n % QK_K) != 0) return -1;
#if defined(__AVX2__)
    return wubu_vec_dot_q3_k_q8_k_avx2(n, s, w, q8);
#else
    return wubu_vec_dot_q3_k_q8_k_scalar(n, s, w, q8);
#endif
}

void wubu_vec_dot_q3_k_q8_k_vd(int n, float *s, size_t bs, const void *vx,
                               size_t bx, const void *vy, size_t by, int nrc)
{
    (void)bs; (void)bx; (void)by;
    if (nrc != 1) { if (s) *s = 0.0f; return; }
    if (wubu_vec_dot_q3_k_q8_k(n, s, vx, vy) != 0 && s) *s = 0.0f;
}
