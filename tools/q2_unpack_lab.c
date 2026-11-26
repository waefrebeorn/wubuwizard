/* How much of the Q2_0 dot is unpacking 2-bit fields into lanes?
 * If expansion alone costs most of the dot, no maddubs rewrite can win much - the unpack is the floor. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#define NIN 2304
#define ROWS 512
#define NB (NIN / 64)

static double wall(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* reference: full float dot */
__attribute__((target("avx2,fma,f16c"))) static float dot_full(const unsigned char *w, const float *x)
{
    const __m256i sh = _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0);
    const __m256i m3 = _mm256_set1_epi32(3), one = _mm256_set1_epi32(1);
    float total = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const unsigned char *qs = w + b * 18 + 2;
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
        total += _mm_cvtss_f32(s);
    }
    return total;
}

/* unpack only: expand every field to a lane and accumulate integers, no multiply, no float */
__attribute__((target("avx2,fma,f16c"))) static int unpack_only(const unsigned char *w)
{
    const __m256i sh = _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0);
    const __m256i m3 = _mm256_set1_epi32(3);
    __m256i acc = _mm256_setzero_si256();
    for (int b = 0; b < NB; ++b) {
        const unsigned char *qs = w + b * 18 + 2;
        for (int g = 0; g < 8; ++g) {
            const unsigned u = (unsigned)qs[2 * g] | ((unsigned)qs[2 * g + 1] << 8);
            acc = _mm256_add_epi32(acc, _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)u), sh), m3));
        }
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

int main(void)
{
    unsigned char *W = (unsigned char *)malloc((size_t)ROWS * NB * 18);
    float *X = (float *)malloc((size_t)NIN * sizeof(float));
    for (int r = 0; r < ROWS; ++r)
        for (int b = 0; b < NB; ++b) {
            unsigned char *blk = W + ((size_t)r * NB + b) * 18;
            uint16_t d = 0x3C00; memcpy(blk, &d, 2);
            for (int i = 0; i < 16; ++i) blk[2 + i] = (unsigned char)((r * 31 + b * 17 + i * 7) & 0xFF);
        }
    for (int i = 0; i < NIN; ++i) X[i] = (float)((i % 37) - 18) / 18.0f;

    double t0 = wall();
    float sink = 0;
    for (int r = 0; r < ROWS; ++r) sink += dot_full(W + (size_t)r * NB * 18, X);
    const double t_full = wall() - t0;

    t0 = wall();
    long isink = 0;
    for (int r = 0; r < ROWS; ++r) isink += unpack_only(W + (size_t)r * NB * 18);
    const double t_un = wall() - t0;

    const double macs = (double)ROWS * NIN / 1e9;
    printf("full dot      %6.3f s  %6.2f GMAC/s\n", t_full, macs / t_full);
    printf("unpack only   %6.3f s   (%.0f%% of the full dot's time)\n", t_un, 100.0 * t_un / t_full);
    printf("=> best case if the multiply+FMA were free: %.2fx\n", t_full / t_un);
    printf("sink %f %ld\n", sink, isink);
    return 0;
}
