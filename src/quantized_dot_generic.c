// src/quantized_dot_generic.c
// Self-contained generic + SIMD implementations of quantized dot products.
// All vec_dot types self-hosted — no libggml-cpu.so dependency.

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "gguf_reader.h"

#define GGML_RESTRICT restrict
#define GGML_CPU_FP16_TO_FP32(x) fp16_to_fp32(x)
#define UNUSED(x) (void)(x)

#define QK_K 256
#define K_SCALE_SIZE 12

static float fp16_to_fp32(uint16_t v) {
    int sign = (v >> 15) & 1;
    int exp  = (v >> 10) & 0x1F;
    int mant =  v        & 0x3FF;
    if (exp == 0) return ldexpf((float)mant / 1024.0f, -14) * (sign ? -1.0f : 1.0f);
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    return ldexpf(1.0f + (float)mant / 1024.0f, exp - 15) * (sign ? -1.0f : 1.0f);
}

#pragma pack(push, 1)
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } block_q4_K;
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; } block_q5_K;
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; uint16_t d; } block_q6_K;
#pragma pack(pop)

void quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k) {
    if (k % QK_K != 0) {
        /* Non-256-divisible activation dim (e.g. WuBu-35M d_model=448).
         * Q8_K block layout can't represent a partial block — the caller
         * must not use the Q8_K dot path for this row. Zero the buffer so
         * a misrouted dot degrades to zeros instead of reading garbage. */
        memset(y, 0, (size_t)((k + QK_K - 1) / QK_K) * sizeof(block_q8_K));
        return;
    }
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f, max_val = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            float ax = fabsf(x[j]);
            if (ax > amax) { amax = ax; max_val = x[j]; }
        }
        if (amax == 0.0f) {
            y[i].d = 0.0f;
            memset(y[i].qs, 0, QK_K);
            memset(y[i].bsums, 0, QK_K/16 * sizeof(int16_t));
            x += QK_K; continue;
        }
        const float iscale = -127.0f / max_val;
        for (int j = 0; j < QK_K; j++) {
            int v = (int)(iscale * x[j] + (x[j] >= 0 ? 0.5f : -0.5f));
            y[i].qs[j] = v > 127 ? 127 : (v < -128 ? -128 : v);
        }
        for (int j = 0; j < QK_K/16; j++) {
            int sum = 0;
            for (int ii = 0; ii < 16; ii++) sum += y[i].qs[j*16 + ii];
            y[i].bsums[j] = (int16_t)sum;
        }
        y[i].d = -max_val / 127.0f;
        x += QK_K;
    }
}

static void decode_scales(const uint8_t scales[12], uint32_t utmp[4]) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    memcpy(utmp, scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

// ========================================================================
// SSE2/SSSE3-accelerated Q4_K × Q8_K vec_dot
// Uses _mm_maddubs_epi16 (SSSE3) for unsigned×signed byte multiply-accumulate
// Ryzen 7950X supports SSSE3+, so this is always available with -march=native
// ========================================================================
#ifdef __SSSE3__
#include <tmmintrin.h>

void ggml_vec_dot_q4_K_q8_K_sse(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0); assert(nrc == 1); UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;
    const int nb = n / QK_K;
    uint32_t utmp[4];
    int8_t  aux8[QK_K];
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        // Dequant Q4_K to int8 aux8 (values 0-15 unsigned)
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32;
            q4 += 32;
        }
        // Decode scales
        decode_scales(x[i].scales, utmp);
        const uint8_t * scales = (const uint8_t*)&utmp[0];
        const uint8_t * mins   = (const uint8_t*)&utmp[2];
        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        // SSE dot product
        a = aux8;
        int is = 0;
        __m128i acc4 = _mm_setzero_si128();  // 4×int32 accumulator
        for (int j = 0; j < QK_K/32; ++j) {
            const int32_t scale = scales[is++];
            // Broadcast scale as 16-bit values for pmaddwd
            const __m128i scale16 = _mm_set1_epi16(scale);
            // Process 32 elements: 2 loads of 16 bytes each
            const __m128i a0 = _mm_loadu_si128((const __m128i*)(a));
            const __m128i q0 = _mm_loadu_si128((const __m128i*)(q8));
            const __m128i a1 = _mm_loadu_si128((const __m128i*)(a + 16));
            const __m128i q1 = _mm_loadu_si128((const __m128i*)(q8 + 16));
            // pmaddubsw: unsigned(a) * signed(q8) → pairwise add to 8×int16
            const __m128i p0 = _mm_maddubs_epi16(a0, q0);
            const __m128i p1 = _mm_maddubs_epi16(a1, q1);
            // pmaddwd: multiply each int16 pair by scale, sum to int32
            acc4 = _mm_add_epi32(acc4, _mm_madd_epi16(p0, scale16));
            acc4 = _mm_add_epi32(acc4, _mm_madd_epi16(p1, scale16));
            q8 += 32; a += 32;
        }
        // Horizontal sum of acc4 (4×int32 → 1×int32)
        __m128i tmp = _mm_hadd_epi32(acc4, acc4);
        tmp = _mm_hadd_epi32(tmp, tmp);
        int32_t sum_val = _mm_cvtsi128_si32(tmp);
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        sumf += d * sum_val;
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    *s = sumf;
}
#endif // __SSSE3__

// ========================================================================
// AVX2-accelerated Q4_K × Q8_K vec_dot (256-bit, 2× SSE throughput)
// ========================================================================
#ifdef __AVX2__
void ggml_vec_dot_q4_K_q8_K_avx2(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0); assert(nrc == 1); UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;
    const int nb = n / QK_K;
    uint32_t utmp[4];
    int8_t  aux8[QK_K];
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        // Prefetch next blocks into L2 while processing current
        if (i + 4 < nb) {
            _mm_prefetch(&x[i+4], _MM_HINT_T1);
            _mm_prefetch(&y[i+4], _MM_HINT_T1);
        }
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32;
            q4 += 32;
        }
        decode_scales(x[i].scales, utmp);
        const uint8_t * scales = (const uint8_t*)&utmp[0];
        const uint8_t * mins   = (const uint8_t*)&utmp[2];
        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        __m256i acc8 = _mm256_setzero_si256();
        for (int j = 0; j < QK_K/64; ++j) {
            const int32_t scale0 = scales[is++];
            const int32_t scale1 = scales[is++];
            __m256i a0 = _mm256_loadu_si256((const __m256i*)(a));
            __m256i q0 = _mm256_loadu_si256((const __m256i*)(q8));
            __m256i a1 = _mm256_loadu_si256((const __m256i*)(a + 32));
            __m256i q1 = _mm256_loadu_si256((const __m256i*)(q8 + 32));
            __m256i p0 = _mm256_maddubs_epi16(a0, q0);
            __m256i p1 = _mm256_maddubs_epi16(a1, q1);
            acc8 = _mm256_add_epi32(acc8, _mm256_madd_epi16(p0, _mm256_set1_epi16((int16_t)scale0)));
            acc8 = _mm256_add_epi32(acc8, _mm256_madd_epi16(p1, _mm256_set1_epi16((int16_t)scale1)));
            q8 += 64; a += 64;
        }
        __m128i lo = _mm256_castsi256_si128(acc8);
        __m128i hi = _mm256_extracti128_si256(acc8, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        int32_t sum_val = _mm_cvtsi128_si32(sum128);
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        sumf += d * sum_val;
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    *s = sumf;
}

// ========================================================================
// AVX2-accelerated Q5_K × Q8_K vec_dot
// ========================================================================
void ggml_vec_dot_q5_K_q8_K_avx2(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0); assert(nrc == 1); UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_q5_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];
    int8_t  aux8[QK_K];
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        __m256i acc8 = _mm256_setzero_si256();
        for (int j = 0; j < QK_K/64; ++j) {
            const int32_t scale0 = scales[is++];
            const int32_t scale1 = scales[is++];
            __m256i a0 = _mm256_loadu_si256((const __m256i*)(a));
            __m256i q0 = _mm256_loadu_si256((const __m256i*)(q8));
            __m256i a1 = _mm256_loadu_si256((const __m256i*)(a + 32));
            __m256i q1 = _mm256_loadu_si256((const __m256i*)(q8 + 32));
            __m256i p0 = _mm256_maddubs_epi16(a0, q0);
            __m256i p1 = _mm256_maddubs_epi16(a1, q1);
            acc8 = _mm256_add_epi32(acc8, _mm256_madd_epi16(p0, _mm256_set1_epi16((int16_t)scale0)));
            acc8 = _mm256_add_epi32(acc8, _mm256_madd_epi16(p1, _mm256_set1_epi16((int16_t)scale1)));
            q8 += 64; a += 64;
        }
        __m128i lo = _mm256_castsi256_si128(acc8);
        __m128i hi = _mm256_extracti128_si256(acc8, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        int32_t sum_val = _mm_cvtsi128_si32(sum128);
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        sumf += d * sum_val;
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    *s = sumf;
}

// ========================================================================
// AVX2-accelerated Q6_K × Q8_K vec_dot (signed × signed, 256-bit)
// ========================================================================
void ggml_vec_dot_q6_K_q8_K_avx2(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0); assert(nrc == 1); UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;
    const int nb = n / QK_K;
    int8_t aux8[QK_K];
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l+  0] = (int8_t)((q4[l+ 0] & 0xF) | (((qh[l]>>0)&3)<<4)) - 32;
                a[l+ 32] = (int8_t)((q4[l+32] & 0xF) | (((qh[l]>>2)&3)<<4)) - 32;
                a[l+ 64] = (int8_t)((q4[l+ 0] >> 4) | (((qh[l]>>4)&3)<<4)) - 32;
                a[l+ 96] = (int8_t)((q4[l+32] >> 4) | (((qh[l]>>6)&3)<<4)) - 32;
            }
            a += 128; q4 += 64; qh += 32;
        }
        a = aux8;
        __m256i acc8 = _mm256_setzero_si256();
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            const int32_t scale = x[i].scales[is++];
            __m256i a8_0 = _mm256_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(a)));
            __m256i q8_0 = _mm256_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(q8)));
            acc8 = _mm256_add_epi32(acc8, _mm256_madd_epi16(_mm256_mullo_epi16(a8_0, q8_0),
                                 _mm256_set1_epi16((int16_t)scale)));
            a += 8; q8 += 8;
            __m256i a8_1 = _mm256_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(a)));
            __m256i q8_1 = _mm256_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(q8)));
            acc8 = _mm256_add_epi32(acc8, _mm256_madd_epi16(_mm256_mullo_epi16(a8_1, q8_1),
                                 _mm256_set1_epi16((int16_t)scale)));
            a += 8; q8 += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        __m128i lo = _mm256_castsi256_si128(acc8);
        __m128i hi = _mm256_extracti128_si256(acc8, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sumf += d * (float)_mm_cvtsi128_si32(sum128);
    }
    *s = sumf;
}
#endif // __AVX2__

// ========================================================================
// SSE2/SSSE3-accelerated Q5_K × Q8_K vec_dot
// Same approach as Q4_K, but Q5_K has extra high-bit (qh) in dequant
// ========================================================================
#ifdef __SSSE3__
void ggml_vec_dot_q5_K_q8_K_sse(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0); assert(nrc == 1); UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_q5_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];
    int8_t  aux8[QK_K];
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        // Dequant Q5_K to int8 aux8 (values 0-31 unsigned)
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        // SSE dot product (same as Q4_K)
        a = aux8;
        int is = 0;
        __m128i acc4 = _mm_setzero_si128();
        for (int j = 0; j < QK_K/32; ++j) {
            const int32_t scale = scales[is++];
            const __m128i scale16 = _mm_set1_epi16(scale);
            const __m128i a0 = _mm_loadu_si128((const __m128i*)(a));
            const __m128i q0 = _mm_loadu_si128((const __m128i*)(q8));
            const __m128i a1 = _mm_loadu_si128((const __m128i*)(a + 16));
            const __m128i q1 = _mm_loadu_si128((const __m128i*)(q8 + 16));
            const __m128i p0 = _mm_maddubs_epi16(a0, q0);
            const __m128i p1 = _mm_maddubs_epi16(a1, q1);
            acc4 = _mm_add_epi32(acc4, _mm_madd_epi16(p0, scale16));
            acc4 = _mm_add_epi32(acc4, _mm_madd_epi16(p1, scale16));
            q8 += 32; a += 32;
        }
        __m128i tmp = _mm_hadd_epi32(acc4, acc4);
        tmp = _mm_hadd_epi32(tmp, tmp);
        int32_t sum_val = _mm_cvtsi128_si32(tmp);
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        sumf += d * sum_val;
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    *s = sumf;
}
#endif // __SSSE3__

// ========================================================================
// Q4_K × Q8_K generic vec_dot (fallback)
// ========================================================================
void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0); assert(nrc == 1); UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;
    const int nb = n / QK_K;
    uint32_t utmp[4];
    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];
    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8]; memset(sums, 0, 8*sizeof(float));
    int32_t aux32[8];
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0f) | (((utmp[1] >> 6) & 0x03030303) << 4);
        const uint32_t uaux = utmp[1] & 0x3f3f3f3f;
        utmp[1] = (utmp[2] & 0x0f0f0f0f) | (((utmp[0] >> 6) & 0x03030303) << 4);
        utmp[2] = uaux;
        utmp[0] &= 0x3f3f3f3f;
        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

// ========================================================================
// Q5_K × Q8_K generic vec_dot
// ========================================================================
void ggml_vec_dot_q5_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0); assert(nrc == 1); UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_q5_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];
    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8]; memset(sums, 0, 8*sizeof(float));
    int32_t aux32[8];
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

// ========================================================================
// Q6_K × Q8_K generic vec_dot
// ========================================================================
void ggml_vec_dot_q6_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0); assert(nrc == 1); UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;
    const int nb = n / QK_K;
    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8]; memset(sums, 0, 8*sizeof(float));
    int32_t aux32[8];
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

// ========================================================================
// Wrapper functions — auto-select SSE or generic
// ========================================================================
void q4_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
#ifdef __AVX2__
    ggml_vec_dot_q4_K_q8_K_avx2(n, s, bs, vx, bx, vy, by, nrc);
#elif defined(__SSSE3__)
    ggml_vec_dot_q4_K_q8_K_sse(n, s, bs, vx, bx, vy, by, nrc);
#else
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

// ========================================================================
// Q8_0 × Q8_0 vec_dot — the Qwen3.5 toolset models are Q8_0-quantized;
// the dispatch previously had NO Q8_0 case, so every Qwen3.5 matmul
// no-oped and the Q/K/V buffers stayed uninitialized (NaN forward).
// Block: [d: fp16][qs: 32 × int8] = 34 bytes.
// ========================================================================
void q8_0_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    const int nb = n / 32;
    const uint8_t *w = (const uint8_t *)vx;   /* Q8_0 weights */
    const uint8_t *x = (const uint8_t *)vy;   /* Q8_0 activations */
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        float d0 = fp16_to_fp32(*(const uint16_t *)(w + i * 34));
        float d1 = fp16_to_fp32(*(const uint16_t *)(x + i * 34));
        __m256i x_i = _mm256_loadu_si256((const __m256i *)(w + i * 34 + 2));
        __m256i y_i = _mm256_loadu_si256((const __m256i *)(x + i * 34 + 2));
        __m256i xs = _mm256_sign_epi8(x_i, x_i);          /* |x| (u8) */
        __m256i ys = _mm256_sign_epi8(y_i, x_i);          /* y·sign(x) (s8) */
        __m256i p16 = _mm256_maddubs_epi16(xs, ys);       /* x·y pairs → s16 */
        __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
        __m256i s32 = _mm256_hadd_epi32(p32, p32);
        s32 = _mm256_hadd_epi32(s32, s32);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d0 * d1), _mm256_cvtepi32_ps(s32), acc);
    }
    __m128 acc0 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    acc0 = _mm_add_ps(acc0, _mm_movehl_ps(acc0, acc0));
    acc0 = _mm_add_ss(acc0, _mm_shuffle_ps(acc0, acc0, 1));
    *s += _mm_cvtss_f32(acc0);
#else
    float acc = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d0 = fp16_to_fp32(*(const uint16_t *)(w + i * 34));
        float d1 = fp16_to_fp32(*(const uint16_t *)(x + i * 34));
        const int8_t *qw = (const int8_t *)(w + i * 34 + 2);
        const int8_t *qx = (const int8_t *)(x + i * 34 + 2);
        int sum = 0;
        for (int j = 0; j < 32; j++) sum += (int)qw[j] * (int)qx[j];
        acc += d0 * d1 * (float)sum;
    }
    *s += acc;
#endif
}
void q5_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
#ifdef __AVX2__
    ggml_vec_dot_q5_K_q8_K_avx2(n, s, bs, vx, bx, vy, by, nrc);
#elif defined(__SSSE3__)
    ggml_vec_dot_q5_K_q8_K_sse(n, s, bs, vx, bx, vy, by, nrc);
#else
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
void q6_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
#ifdef __AVX2__
    ggml_vec_dot_q6_K_q8_K_avx2(n, s, bs, vx, bx, vy, by, nrc);
#elif defined(__SSE4_1__)
    ggml_vec_dot_q6_K_q8_K_sse(n, s, bs, vx, bx, vy, by, nrc);
#else
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}


// IQ block structs (uint16_t = ggml_half replacement)
#pragma pack(push, 1)
typedef struct { uint16_t d; uint16_t qs[QK_K/8]; } block_iq2_xxs;
typedef struct { uint16_t d; uint8_t qs[3*QK_K/8]; } block_iq3_xxs;
typedef struct { uint16_t d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; } block_iq4_xs;
#pragma pack(pop)
// GGML_TABLE_BEGIN macro creates static const arrays
#undef GGML_TABLE_BEGIN
#undef GGML_TABLE_END
#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };

GGML_TABLE_BEGIN(uint8_t, kmask_iq2xs, 8)
    1, 2, 4, 8, 16, 32, 64, 128
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint8_t, ksigns_iq2xs, 128)
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint64_t, iq2xxs_grid, 256)
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint32_t, iq3xxs_grid, 256)
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
GGML_TABLE_END()

GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16)
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
GGML_TABLE_END()

// ========================================================================
// AVX2-accelerated IQ2_XXS × Q8_K vec_dot (256-bit grid lookup)
// Ported from llama.cpp ggml-cpu/arch/x86/quants.c
// ========================================================================
#if defined(__AVX2__) || defined(__AVX__)
// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static const int8_t keven_signs_q2xs[1024] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
};

void ggml_vec_dot_iq2_xxs_q8_K_avx2(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0); assert(nrc == 1); UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_iq2_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;
    const int nb = n / QK_K;
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;
    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *)aux32;
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[ 3]], iq2xxs_grid[aux8[ 2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = 0.125f * hsum_float_8(accumf);
}

// ========================================================================
// AVX2-accelerated IQ3_XXS × Q8_K vec_dot (port from llama.cpp)
// 256-element block: 64 bytes grid indices + 32 bytes gas (signs+scale)
// Each outer iteration processes 2× Q32 blocks = 64 elements
// Uses keven_signs_q2xs (not ksigns_iq2xs — different lookup!)
// ========================================================================
void ggml_vec_dot_iq3_xxs_q8_K_avx2(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0); assert(nrc == 1); UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_iq3_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;
    const int nb = n / QK_K;
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;
    uint32_t aux32[2];
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            const __m256i q2_2 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            memcpy(aux32, gas, 8); gas += 8;
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                   signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = aux32[0] >> 28;
            const uint16_t ls2 = aux32[1] >> 28;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = 0.25f * hsum_float_8(accumf);
}
#endif // defined(__AVX2__) || defined(__AVX__)

// ========== IQ Vec-dot Functions ==========
void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, q2, 2*sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq3_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    uint32_t aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, gas, sizeof(uint32_t)); gas += sizeof(uint32_t);
            const uint32_t ls = 2*(aux32 >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*l+0]);
                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*l+1]);
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            q3 += 8;
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.25f * sumf;
}


void ggml_vec_dot_iq4_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_K == 0);

    const block_iq4_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_CPU_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const uint8_t ls1 = (x[ibl].scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
            const uint8_t ls2 = (x[ibl].scales_l[ib/2] >>  4) | ((h << 2) & 0x30);
            h >>= 4;
            const float d1 = d4d8*(ls1 - 32);
            const float d2 = d4d8*(ls2 - 32);
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d2 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    *s = sumf;
}


// IQ wrapper functions
void iq2_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
#ifdef __AVX2__
    ggml_vec_dot_iq2_xxs_q8_K_avx2(n, s, bs, vx, bx, vy, by, nrc);
#else
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
void iq3_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
#ifdef __AVX2__
    ggml_vec_dot_iq3_xxs_q8_K_avx2(n, s, bs, vx, bx, vy, by, nrc);
#else
    ggml_vec_dot_iq3_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
void iq4_xs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
    /* IQ4_XS has a different block layout (block_iq4_xs: d, scales_h,
     * scales_l, qs) from Q4_K (block_q4_K: d, dmin, scales, qs), so it cannot
     * share the Q4_K SIMD path. The generic path applies the scale ONCE
     * at the reduction boundary (int32 accumulator, single dequant) — no
     * per-element F32 materialization. With the Theory/08 aligned geometry
     * (input dim always div-256), the generic path is still correct and
     * branch-free; a dedicated IQ4_XS AVX2 kernel is future work. */
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
}

// ========================================================================
// SSE4.1 Q6_K vec_dot (signed × signed)
// ========================================================================
#if defined(__SSE4_1__) && !defined(AVOID_SIMD_Q6)
#include <smmintrin.h>
void ggml_vec_dot_q6_K_q8_K_sse(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0); assert(nrc == 1); UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;
    const int nb = n / QK_K;
    int8_t aux8[QK_K];
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l+  0] = (int8_t)((q4[l+ 0] & 0xF) | (((qh[l]>>0)&3)<<4)) - 32;
                a[l+ 32] = (int8_t)((q4[l+32] & 0xF) | (((qh[l]>>2)&3)<<4)) - 32;
                a[l+ 64] = (int8_t)((q4[l+ 0] >> 4) | (((qh[l]>>4)&3)<<4)) - 32;
                a[l+ 96] = (int8_t)((q4[l+32] >> 4) | (((qh[l]>>6)&3)<<4)) - 32;
            }
            a += 128; q4 += 64; qh += 32;
        }
        a = aux8;
        __m128i acc4 = _mm_setzero_si128();
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            const int32_t scale = x[i].scales[is++];
            __m128i a8 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(a)));
            __m128i q8v = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(q8)));
            acc4 = _mm_add_epi32(acc4, _mm_madd_epi16(_mm_mullo_epi16(a8, q8v),
                                 _mm_set1_epi16((int16_t)scale)));
            a += 8; q8 += 8;
            a8 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(a)));
            q8v = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(q8)));
            acc4 = _mm_add_epi32(acc4, _mm_madd_epi16(_mm_mullo_epi16(a8, q8v),
                                 _mm_set1_epi16((int16_t)scale)));
            a += 8; q8 += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        __m128i tmp = _mm_hadd_epi32(acc4, acc4);
        tmp = _mm_hadd_epi32(tmp, tmp);
        sumf += d * (float)_mm_cvtsi128_si32(tmp);
    }
    *s = sumf;
}
#endif // __SSE4_1__
