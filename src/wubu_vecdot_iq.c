/* wubu_vecdot_iq.c -- fused IQ2_S / IQ3_S x Q8_K dot products. Strict C11.
 *
 * Reference: ggml_vec_dot_iq2_s_q8_K_generic / ggml_vec_dot_iq3_s_q8_K_generic
 * in llama/ggml-cpu/quants.c (MIT), with the block layouts from
 * llama/ggml-common.h and the grid tables already used by src/gguf_reader.c.
 *
 * Grid semantics (must match gguf_reader.c dequantisers, which are the
 * independent reference the gate compares against):
 *   iq2s_grid[1024] : uint64, 8 int8 magnitudes per entry
 *   iq3s_grid[512]  : uint32, 4 int8 magnitudes per entry
 */
#include "wubu_vecdot_iq.h"
#include "gguf_reader.h"

#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define QK_K 256

typedef struct {
    uint16_t d;
    uint8_t  qs[QK_K / 4];      /* 64 */
    uint8_t  qh[QK_K / 32];     /* 8  */
    uint8_t  scales[QK_K / 32]; /* 8  */
} wubu_block_iq2_s;

typedef struct {
    uint16_t d;
    uint8_t  qs[QK_K / 4];      /* 64 */
    uint8_t  qh[QK_K / 32];     /* 8  */
    uint8_t  signs[QK_K / 8];   /* 32 */
    uint8_t  scales[QK_K / 64]; /* 4  */
} wubu_block_iq3_s;

static const uint8_t ksign_mask[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static const uint64_t iq2s_grid[1024] = {
#include "iq2s_grid.inc"
};

static const uint32_t iq3s_grid[512] = {
#include "iq3s_grid.inc"
};

int wubu_vec_dot_iq2_s_q8_k_scalar(int n, float *s, const void *w, const void *q8)
{
    const wubu_block_iq2_s *x = (const wubu_block_iq2_s *)w;
    const block_q8_K       *y = (const block_q8_K *)q8;
    const int nb = n / QK_K;

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d = gguf_f16_to_f32(x[i].d) * y[i].d;
        const int8_t  *q8v = y[i].qs;
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = qs + QK_K / 8;

        int bsum = 0;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const int ls1 = 1 + 2 * (x[i].scales[ib32] & 0xf);
            const int ls2 = 1 + 2 * (x[i].scales[ib32] >> 4);
            int sumi1 = 0, sumi2 = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t *grid = (const uint8_t *)(iq2s_grid +
                    (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
                for (int j = 0; j < 8; ++j)
                    sumi1 += q8v[j] * grid[j] * ((signs[l] & ksign_mask[j]) ? -1 : 1);
                q8v += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const uint8_t *grid = (const uint8_t *)(iq2s_grid +
                    (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
                for (int j = 0; j < 8; ++j)
                    sumi2 += q8v[j] * grid[j] * ((signs[l] & ksign_mask[j]) ? -1 : 1);
                q8v += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs += 4;
            signs += 4;
        }
        sumf += d * (float)bsum;
    }
    /* IQ2_S dequant carries a 0.25 factor on (0.5 + scale), i.e. 0.125 * (1 + 2*scale). */
    *s = 0.125f * sumf;
    return 0;
}

int wubu_vec_dot_iq3_s_q8_k_scalar(int n, float *s, const void *w, const void *q8)
{
    const wubu_block_iq3_s *x = (const wubu_block_iq3_s *)w;
    const block_q8_K       *y = (const block_q8_K *)q8;
    const int nb = n / QK_K;

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d = gguf_f16_to_f32(x[i].d) * y[i].d;
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = x[i].signs;
        const int8_t  *q8v = y[i].qs;

        int bsum = 0;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const int ls1 = 2 * (x[i].scales[ib32 / 2] & 0xf) + 1;
            const int ls2 = 2 * (x[i].scales[ib32 / 2] >> 4) + 1;

            int sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = (const uint8_t *)(iq3s_grid +
                    (qs[2 * l + 0] | ((qh[ib32 + 0] << (8 - 2 * l)) & 256)));
                const uint8_t *grid2 = (const uint8_t *)(iq3s_grid +
                    (qs[2 * l + 1] | ((qh[ib32 + 0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8v[j + 0] * ((signs[l] & ksign_mask[j + 0]) ? -1 : 1);
                    sumi += grid2[j] * q8v[j + 4] * ((signs[l] & ksign_mask[j + 4]) ? -1 : 1);
                }
                q8v += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls1;

            sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = (const uint8_t *)(iq3s_grid +
                    (qs[2 * l + 0] | ((qh[ib32 + 1] << (8 - 2 * l)) & 256)));
                const uint8_t *grid2 = (const uint8_t *)(iq3s_grid +
                    (qs[2 * l + 1] | ((qh[ib32 + 1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8v[j + 0] * ((signs[l] & ksign_mask[j + 0]) ? -1 : 1);
                    sumi += grid2[j] * q8v[j + 4] * ((signs[l] & ksign_mask[j + 4]) ? -1 : 1);
                }
                q8v += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls2;
        }
        sumf += d * (float)bsum;
    }
    *s = sumf;
    return 0;
}

/* -------------------------------------------------------------------- AVX2 */

#if defined(__AVX2__)

static inline float iq_hsum_float_8(const __m256 x)
{
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

/* Byte shuffle that broadcasts one 16-bit scale per 32-element group. */
static inline __m256i iq_get_scale_shuffle_k4(int i)
{
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i *)k_shuffle + i);
}

int wubu_vec_dot_iq2_s_q8_k_avx2(int n, float *s, const void *w, const void *q8v)
{
    const wubu_block_iq2_s *x = (const wubu_block_iq2_s *)w;
    const block_q8_K       *y = (const block_q8_K *)q8v;
    const int nb = n / QK_K;

    static const uint8_t k_mask1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };
    static const uint8_t k_mask2[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
    };

    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m128i m1 = _mm_set1_epi8(1);
    const __m256i mask1 = _mm256_loadu_si256((const __m256i *)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i *)k_mask2);

    uint64_t aux64;
    __m256 accumf = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = gguf_f16_to_f32(x[i].d) * y[i].d;
        const uint8_t  *qs = x[i].qs;
        const uint8_t  *qh = x[i].qh;
        const uint16_t *signs = (const uint16_t *)(x[i].qs + QK_K / 8);
        const int8_t   *q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(
            _mm_slli_epi16(_mm_and_si128(
                _mm_set_epi64x((long long)(aux64 >> 4), (long long)aux64), m4), 1), m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8);

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_1 = _mm256_set_epi64x(
                (long long)iq2s_grid[qs[3] | ((qh[ib32 + 0] << 2) & 0x300)],
                (long long)iq2s_grid[qs[2] | ((qh[ib32 + 0] << 4) & 0x300)],
                (long long)iq2s_grid[qs[1] | ((qh[ib32 + 0] << 6) & 0x300)],
                (long long)iq2s_grid[qs[0] | ((qh[ib32 + 0] << 8) & 0x300)]);
            const __m256i q2_2 = _mm256_set_epi64x(
                (long long)iq2s_grid[qs[7] | ((qh[ib32 + 1] << 2) & 0x300)],
                (long long)iq2s_grid[qs[6] | ((qh[ib32 + 1] << 4) & 0x300)],
                (long long)iq2s_grid[qs[5] | ((qh[ib32 + 1] << 6) & 0x300)],
                (long long)iq2s_grid[qs[4] | ((qh[ib32 + 1] << 8) & 0x300)]);
            qs += 8;

            __m256i aux256 = _mm256_set1_epi32((int)(signs[0] | ((uint32_t)signs[1] << 16)));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32((int)(signs[2] | ((uint32_t)signs[3] << 16)));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);

            const __m256i p1 = _mm256_madd_epi16(dot1,
                _mm256_shuffle_epi8(scales16, iq_get_scale_shuffle_k4(ib32 + 0)));
            const __m256i p2 = _mm256_madd_epi16(dot2,
                _mm256_shuffle_epi8(scales16, iq_get_scale_shuffle_k4(ib32 + 1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        const __m256 prod = _mm256_mul_ps(_mm256_set1_ps(d),
                                          _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)));
        accumf = _mm256_add_ps(prod, accumf);
    }

    *s = 0.125f * iq_hsum_float_8(accumf);
    return 0;
}

int wubu_vec_dot_iq3_s_q8_k_avx2(int n, float *s, const void *w, const void *q8v)
{
    const wubu_block_iq3_s *x = (const wubu_block_iq3_s *)w;
    const block_q8_K       *y = (const block_q8_K *)q8v;
    const int nb = n / QK_K;

    static const uint8_t k_mask1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };
    static const uint8_t k_mask2[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
    };

    const __m256i mask1 = _mm256_loadu_si256((const __m256i *)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i *)k_mask2);
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask = _mm256_set1_epi32(256);

    typedef union {
        __m256i  vec[2];
        uint32_t index[16];
    } index_t;

    index_t idx;
    __m256 accumf = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = gguf_f16_to_f32(x[i].d) * y[i].d;
        const uint8_t  *qs = x[i].qs;
        const uint8_t  *qh = x[i].qh;
        const uint16_t *signs = (const uint16_t *)x[i].signs;
        const int8_t   *q8 = y[i].qs;

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs)); qs += 16;

            idx.vec[0] = _mm256_set1_epi32(qh[ib32 + 0]);
            idx.vec[1] = _mm256_set1_epi32(qh[ib32 + 1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));

            const __m256i q2_1 = _mm256_set_epi32(
                (int)iq3s_grid[idx.index[7]], (int)iq3s_grid[idx.index[6]],
                (int)iq3s_grid[idx.index[5]], (int)iq3s_grid[idx.index[4]],
                (int)iq3s_grid[idx.index[3]], (int)iq3s_grid[idx.index[2]],
                (int)iq3s_grid[idx.index[1]], (int)iq3s_grid[idx.index[0]]);
            const __m256i q2_2 = _mm256_set_epi32(
                (int)iq3s_grid[idx.index[15]], (int)iq3s_grid[idx.index[14]],
                (int)iq3s_grid[idx.index[13]], (int)iq3s_grid[idx.index[12]],
                (int)iq3s_grid[idx.index[11]], (int)iq3s_grid[idx.index[10]],
                (int)iq3s_grid[idx.index[ 9]], (int)iq3s_grid[idx.index[ 8]]);

            __m256i aux256 = _mm256_set1_epi32((int)(signs[0] | ((uint32_t)signs[1] << 16)));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32((int)(signs[2] | ((uint32_t)signs[3] << 16)));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32 / 2] & 0x0f;
            const uint16_t ls2 = x[i].scales[ib32 / 2] >> 4;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16((short)(2 * ls1 + 1)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16((short)(2 * ls2 + 1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        const __m256 prod = _mm256_mul_ps(_mm256_set1_ps(d),
                                          _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)));
        accumf = _mm256_add_ps(prod, accumf);
    }

    *s = iq_hsum_float_8(accumf);
    return 0;
}

#endif /* __AVX2__ */

/* ---------------------------------------------------------------- dispatch */

/* Table accessors: the CUDA kernels need the same grids on the device. */
const void *wubu_vecdot_iq2s_grid(void) { return (const void *)iq2s_grid; }
const void *wubu_vecdot_iq3s_grid(void) { return (const void *)iq3s_grid; }

int wubu_vec_dot_iq2_s_q8_k(int n, float *s, const void *w, const void *q8)
{
    if (!s || !w || !q8 || n <= 0 || (n % QK_K) != 0) return -1;
#if defined(__AVX2__)
    return wubu_vec_dot_iq2_s_q8_k_avx2(n, s, w, q8);
#else
    return wubu_vec_dot_iq2_s_q8_k_scalar(n, s, w, q8);
#endif
}

int wubu_vec_dot_iq3_s_q8_k(int n, float *s, const void *w, const void *q8)
{
    if (!s || !w || !q8 || n <= 0 || (n % QK_K) != 0) return -1;
#if defined(__AVX2__)
    return wubu_vec_dot_iq3_s_q8_k_avx2(n, s, w, q8);
#else
    return wubu_vec_dot_iq3_s_q8_k_scalar(n, s, w, q8);
#endif
}

void wubu_vec_dot_iq2_s_q8_k_vd(int n, float *s, size_t bs, const void *vx,
                                size_t bx, const void *vy, size_t by, int nrc)
{
    (void)bs; (void)bx; (void)by;
    if (nrc != 1) { if (s) *s = 0.0f; return; }
    if (wubu_vec_dot_iq2_s_q8_k(n, s, vx, vy) != 0 && s) *s = 0.0f;
}

void wubu_vec_dot_iq3_s_q8_k_vd(int n, float *s, size_t bs, const void *vx,
                                size_t bx, const void *vy, size_t by, int nrc)
{
    (void)bs; (void)bx; (void)by;
    if (nrc != 1) { if (s) *s = 0.0f; return; }
    if (wubu_vec_dot_iq3_s_q8_k(n, s, vx, vy) != 0 && s) *s = 0.0f;
}
