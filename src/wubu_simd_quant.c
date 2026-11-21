/*
 * wubu_simd_quant.c -- LP-15: per-shape native SIMD quant decode.
 * Strict C11 + GCC intrinsics, self-contained, opaque context.
 * See include/wubu_simd_quant.h.
 */
#include "wubu_portable_simd.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_simd_quant.h"

#define WUBU_SIMD_Q_SPECIAL_TAG 0xFFFFFFFFu

struct wubu_simd_quant {
    float scale;
    wubu_simd_q_path_t last_path;
};

/* Exponent-all-ones check works even under -ffast-math (no isnan/isinf). */
static int special_bits(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return (u & 0x7F800000u) == 0x7F800000u;
}

static int finite_scale(float s)
{
    uint32_t u;
    if (!(s > 0.0f)) return 0;
    memcpy(&u, &s, sizeof(u));
    return (u & 0x7F800000u) != 0x7F800000u;
}

wubu_simd_quant_t *wubu_simd_quant_create(float scale)
{
    wubu_simd_quant_t *s;
    if (!finite_scale(scale)) return NULL;
    s = (wubu_simd_quant_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->scale = scale;
    s->last_path = WUBU_SIMD_Q_SCALAR;
    return s;
}

void wubu_simd_quant_free(wubu_simd_quant_t *s) { free(s); }

int wubu_simd_quant_has_avx2(void)
{
#if defined(__AVX2__)
    return __builtin_cpu_supports("avx2") ? 1 : 0;
#else
    return 0;
#endif
}

wubu_simd_q_path_t wubu_simd_quant_dispatch(int n)
{
    if (n <= 0) return WUBU_SIMD_Q_SCALAR;
    if (wubu_simd_quant_has_avx2() && (n & 7) == 0) return WUBU_SIMD_Q_AVX2_8;
    if ((n & 1) == 0) return WUBU_SIMD_Q_SSE_2;
    return WUBU_SIMD_Q_SCALAR;
}

int wubu_simd_quant_encode(const wubu_simd_quant_t *s, const float *in,
                           uint64_t *out, int n)
{
    if (!s || !in || !out || n <= 0) return -1;
    for (int i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &in[i], sizeof(bits));
        if (special_bits(in[i])) {
            out[i] = ((uint64_t)WUBU_SIMD_Q_SPECIAL_TAG << 32) | bits;
        } else {
            int c = (int)lrintf(in[i] / s->scale);
            if (c > 127) c = 127;
            if (c < -127) c = -127;
            out[i] = (uint64_t)(uint32_t)(uint8_t)(int8_t)c; /* tag 0 */
        }
    }
    return 0;
}

/* Scalar tail (also the whole scalar kernel). */
static void decode_one(uint64_t t, float scale, float *out)
{
    uint32_t payload = (uint32_t)t;
    uint32_t tag = (uint32_t)(t >> 32);
    if (tag == WUBU_SIMD_Q_SPECIAL_TAG) {
        memcpy(out, &payload, sizeof(payload)); /* exact NaN/Inf bit identity */
    } else {
        int8_t c = (int8_t)(uint8_t)payload;
        *out = (float)c * scale;
    }
}

static void decode_scalar(const uint64_t *in, float *out, int n, float scale)
{
    for (int i = 0; i < n; i++) decode_one(in[i], scale, out + i);
}

#if defined(__SSE4_1__)
/* Decode two 64-bit tokens held in one 128-bit register. The low 32-bit
 * halves are compacted to lanes 0,1 (tag halves would otherwise decode as
 * extra codes and overwrite neighbours); the integer blend restores raw
 * NaN/Inf bits exactly. */
static inline void decode_pair_sse(__m128i t, float scale, float *dst)
{
    const __m128i ones = _mm_set1_epi64x(0x00000000FFFFFFFFLL);
    const __m128i lowmask = _mm_set1_epi64x(0x00000000FFFFFFFFLL);
    const __m128i bytemask = _mm_set1_epi32(0xFF);
    const __m128i x80 = _mm_set1_epi32(0x80);
    __m128i lo = _mm_and_si128(t, lowmask);
    __m128i hi = _mm_srli_epi64(t, 32);
    __m128i m = _mm_cmpeq_epi64(hi, ones);
    lo = _mm_shuffle_epi32(lo, _MM_SHUFFLE(0, 0, 2, 0)); /* lanes 0,2 -> 0,1 */
    m = _mm_shuffle_epi32(m, _MM_SHUFFLE(0, 0, 2, 0));
    __m128i code = _mm_and_si128(lo, bytemask);
    __m128i sg = _mm_sub_epi32(_mm_xor_si128(code, x80), x80);
    __m128 f = _mm_mul_ps(_mm_cvtepi32_ps(sg), _mm_set1_ps(scale));
    __m128i bits = _mm_blendv_epi8(_mm_castps_si128(f), lo, m);
    uint64_t pair = (uint64_t)_mm_cvtsi128_si64(bits);
    memcpy(dst, &pair, sizeof(pair));
}

static void decode_sse(const uint64_t *in, float *out, int n, float scale)
{
    int i = 0;
    for (; i + 2 <= n; i += 2)
        decode_pair_sse(_mm_loadu_si128((const __m128i *)(in + i)), scale, out + i);
    for (; i < n; i++) decode_one(in[i], scale, out + i);
}

#endif /* __SSE4_1__ */

#if defined(__AVX2__)
/* 8-wide AVX2 kernel: two 256-bit loads per block, each split into its two
 * 128-bit halves (a 256-bit shuffle cannot cross the 128-bit boundary). */
static void decode_avx2(const uint64_t *in, float *out, int n, float scale)
{
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        for (int b = 0; b < 2; b++) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(in + i + b * 4));
            decode_pair_sse(_mm256_castsi256_si128(v), scale, out + i + b * 4);
            decode_pair_sse(_mm256_extracti128_si256(v, 1), scale, out + i + b * 4 + 2);
        }
    }
    for (; i < n; i++) decode_one(in[i], scale, out + i);
}

#endif /* __AVX2__ */

int wubu_simd_quant_decode_path(wubu_simd_quant_t *s, wubu_simd_q_path_t path,
                                const uint64_t *in, float *out, int n)
{
    if (!s || !in || !out || n <= 0) return -1;
    if (path == WUBU_SIMD_Q_AVX2_8 && !wubu_simd_quant_has_avx2()) return -1;
    switch (path) {
        case WUBU_SIMD_Q_SCALAR: decode_scalar(in, out, n, s->scale); break;
        case WUBU_SIMD_Q_SSE_2:
#if defined(__SSE4_1__)
            decode_sse(in, out, n, s->scale); break;
#else
            decode_scalar(in, out, n, s->scale); break;
#endif
        case WUBU_SIMD_Q_AVX2_8:
#if defined(__AVX2__)
            decode_avx2(in, out, n, s->scale); break;
#else
            decode_scalar(in, out, n, s->scale); break;
#endif
        default: return -1;
    }
    s->last_path = path;
    return 0;
}

int wubu_simd_quant_decode(wubu_simd_quant_t *s, const uint64_t *in,
                           float *out, int n)
{
    if (!s || !in || !out || n <= 0) return -1;
    return wubu_simd_quant_decode_path(s, wubu_simd_quant_dispatch(n), in, out, n);
}

wubu_simd_q_path_t wubu_simd_quant_last_path(const wubu_simd_quant_t *s)
{
    return s ? s->last_path : WUBU_SIMD_Q_SCALAR;
}
