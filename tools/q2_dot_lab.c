/* Q2_0 dot: float-activation path (what the engine runs) vs int8-activation + maddubs.
 * Same data, same order of magnitude of work. Decides whether the rewrite is worth the parity risk.
 * Layout: 18-byte block = fp16 d, 16 bytes = 64 two-bit fields (4 per byte, LSB first). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#define NROW 2304           /* in-features per row */
#define NROW_OUT 512        /* rows so the run takes measurable time */
#define NB (NROW / 64)

static double wall(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__attribute__((target("avx2,fma,f16c"))) static float dot_float(const unsigned char *w, const float *x)
{
    const __m256i sh = _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0);
    const __m256i m3 = _mm256_set1_epi32(3);
    const __m256i one = _mm256_set1_epi32(1);
    float total = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const unsigned char *blk = w + b * 18;
        uint16_t db; memcpy(&db, blk, 2);
        const float d = _cvtsh_ss(db);
        const unsigned char *qs = blk + 2;
        const float *xb = x + b * 64;
        __m256 bacc = _mm256_setzero_ps();
        for (int g = 0; g < 8; ++g) {
            const unsigned u = (unsigned)qs[2 * g] | ((unsigned)qs[2 * g + 1] << 8);
            __m256i q = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)u), sh), m3);
            q = _mm256_sub_epi32(q, one);
            bacc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q), _mm256_loadu_ps(xb + g * 8), bacc);
        }
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(bacc), _mm256_extractf128_ps(bacc, 1));
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        total += d * _mm_cvtss_f32(s);
    }
    return total;
}

/* int8 activations: one scale per 64-element block, matching the q8_0 block granularity the engine already uses
 * for type 8/40/42. */
__attribute__((target("avx2,fma,f16c"))) static float dot_i8(const unsigned char *w, const float *x)
{
    const __m256i sh2 = _mm256_set_epi32(30, 28, 26, 24, 22, 20, 18, 16);
    const __m256i sh0 = _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0);
    const __m256i e3  = _mm256_set1_epi32(3);
    const __m256i e2  = _mm256_set1_epi8(2);
    const __m256i e1  = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i iacc = _mm256_setzero_si256();
    float fsum = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const unsigned char *blk = w + b * 18;
        uint16_t db; memcpy(&db, blk, 2);
        const float d = _cvtsh_ss(db);
        const unsigned char *qs = blk + 2;
        const float *xb = x + b * 64;

        /* quantise the 64 activations to int8 with one scale */
        __m256 vmax = _mm256_setzero_ps();
        for (int g = 0; g < 8; ++g) vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_loadu_ps(xb + g * 8)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        const float amax = _mm_cvtss_f32(m4);
        const float xs = amax > 0.0f ? amax / 127.0f : 1.0f;
        const __m256 ivs = _mm256_set1_ps(1.0f / xs);
        __m256i xq[8];
        for (int g = 0; g < 8; ++g) {
            __m256 v = _mm256_mul_ps(_mm256_loadu_ps(xb + g * 8), ivs);
            xq[g] = _mm256_cvtps_epi32(_mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }

        /* 2-bit fields -> int8 {-1,0,1,2}: two 32-bit words cover 16 fields, sll to the top byte of each dword,
         * then one shuffle with 0x03.. to place four signed bytes per dword. */
        __m256i acc16 = _mm256_setzero_si256();
        for (int h = 0; h < 4; ++h) {
            const unsigned u0 = (unsigned)qs[4 * h] | ((unsigned)qs[4 * h + 1] << 8) |
                                ((unsigned)qs[4 * h + 2] << 16) | ((unsigned)qs[4 * h + 3] << 24);
            const __m256i bcast = _mm256_set1_epi32((int)u0);
            __m256i q0 = _mm256_and_si256(_mm256_srlv_epi32(bcast, sh0), e3);   /* fields 0..7  */
            __m256i q1 = _mm256_and_si256(_mm256_srlv_epi32(bcast, sh2), e3);   /* fields 8..15 */
            __m256i qv = _mm256_packus_epi32(q0, q1);                          /* 16 values in 16-bit lanes */
            qv = _mm256_permute4x64_epi64(qv, 0xD8);
            __m256i qb = _mm256_packus_epi16(qv, qv);                          /* low 16 bytes = 16 values */
            qb = _mm256_sub_epi8(_mm256_and_si256(qb, _mm256_set1_epi8(0x03)), _mm256_set1_epi8(1));
            (void)e2; (void)e1;
            /* activations for these 16 elements are xq[2h], xq[2h+1] packed to int16 then int8 */
            __m256i xa = _mm256_packs_epi32(xq[2 * h], xq[2 * h + 1]);
            xa = _mm256_permute4x64_epi64(xa, 0xD8);
            xa = _mm256_packs_epi16(xa, xa);
            const __m256i prod = _mm256_maddubs_epi16(qb, xa);   /* q is unsigned input in maddubs -> see note */
            (void)prod;
            /* maddubs treats the first operand as unsigned; our q is signed, so use signed-signed instead: */
            const __m256i p2 = _mm256_maddubs_epi16(_mm256_add_epi8(qb, _mm256_set1_epi8(1)), xa);
            const __m256i corr = _mm256_maddubs_epi16(_mm256_set1_epi8(1), xa);
            acc16 = _mm256_add_epi32(acc16, _mm256_madd_epi16(_mm256_sub_epi16(p2, corr), ones16));
        }
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(acc16), _mm256_extracti128_si256(acc16, 1));
        s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
        fsum += d * xs * (float)_mm_cvtsi128_si32(s4);
    }
    return fsum;
}

int main(void)
{
    const int rows = NROW_OUT;
    unsigned char *W = (unsigned char *)malloc((size_t)rows * NB * 18);
    float *X = (float *)malloc((size_t)NROW * sizeof(float));
    float *Y = (float *)malloc((size_t)rows * sizeof(float));
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < NB; ++b) {
            unsigned char *blk = W + ((size_t)r * NB + b) * 18;
            uint16_t d = 0x3C00; /* 1.0h */
            memcpy(blk, &d, 2);
            for (int i = 0; i < 16; ++i) blk[2 + i] = (unsigned char)((r * 31 + b * 17 + i * 7) & 0xFF);
        }
    for (int i = 0; i < NROW; ++i) X[i] = (float)((i % 37) - 18) / 18.0f;

    double t0 = wall();
    for (int r = 0; r < rows; ++r) Y[r] = dot_float(W + (size_t)r * NB * 18, X);
    const double tf = wall() - t0;

    t0 = wall();
    for (int r = 0; r < rows; ++r) Y[r] = dot_i8(W + (size_t)r * NB * 18, X);
    const double ti = wall() - t0;

    const double macs = (double)rows * NROW / 1e9;
    printf("rows=%d in=%d blocks=%d  %.3f GMAC\n", rows, NROW, NB, macs);
    printf("float-activation  %7.3f s   %6.2f GMAC/s\n", tf, macs / tf);
    printf("int8 + maddubs    %7.3f s   %6.2f GMAC/s\n", ti, macs / ti);
    printf("speedup %.2fx   (sink %.4f)\n", tf / ti, Y[0] + Y[rows - 1]);
    return 0;
}
