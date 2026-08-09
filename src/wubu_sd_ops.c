/* wubu_sd_ops.c -- 2D conv / groupnorm / sample ops for the WuBu SD engine.
 * C11, self-contained. Conv is the direct algorithm (fine for the
 * 64x64 latent: 320-1280 channels at 8x8-64x64 spatial is small).
 */
#include "wubu_sd_ops.h"
#include "gguf_reader.h" /* gguf_f16_to_f32 — MUST be declared (implicit int return truncates) */
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#if defined(__AVX2__) || defined(__FMA__)
#include <immintrin.h>
#endif

/* y[M,N] = x[M,K] @ W^T[K,N]  (nn.Linear convention, W row-major [N,K]).
 * OpenMP-parallel over M; the workhorse linear for CLIP/UNet.
 * SIMD: AVX2 FMA, 8 output columns in flight per k-vector — the x row is
 * read ONCE per 8 columns (not per column), cutting the GEMM's x memory
 * traffic ~N/8x. For N=320 conv GEMMs that is 15GB -> 1.9GB per call. */
void wubu_sd_matmul_nt(const float *x, const float *W, int M, int K, int N,
                       float *y)
{
#if defined(__AVX512F__) && defined(__FMA__)
    /* AVX-512: 2 rows x 8 cols, 16 zmm accs (16-wide). */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M - 1; i += 2) {
        const float *x0 = x + (size_t)i * K;
        const float *x1 = x + (size_t)(i + 1) * K;
        float *y0 = y + (size_t)i * N;
        float *y1 = y + (size_t)(i + 1) * N;
        int j0 = 0;
        for (; j0 + 8 <= N; j0 += 8) {
            __m512 acc[16];
            for (int a = 0; a < 16; a++) acc[a] = _mm512_setzero_ps();
            const float *wr[8];
            for (int a = 0; a < 8; a++) wr[a] = W + (size_t)(j0 + a) * K;
            int k = 0;
            for (; k + 16 <= K; k += 16) {
                __m512 xv0 = _mm512_loadu_ps(x0 + k);
                __m512 xv1 = _mm512_loadu_ps(x1 + k);
                #pragma unroll(8)
                for (int a = 0; a < 8; a++) {
                    __m512 wv = _mm512_loadu_ps(wr[a] + k);
                    acc[a]     = _mm512_fmadd_ps(xv0, wv, acc[a]);
                    acc[8 + a] = _mm512_fmadd_ps(xv1, wv, acc[8 + a]);
                }
            }
            float s[16];
            for (int a = 0; a < 16; a++) {
                float t[16];
                _mm512_storeu_ps(t, acc[a]);
                s[a] = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7] +
                       t[8] + t[9] + t[10] + t[11] + t[12] + t[13] + t[14] + t[15];
            }
            for (; k < K; k++) {
                float xv0 = x0[k], xv1 = x1[k];
                for (int a = 0; a < 8; a++) {
                    float wv = wr[a][k];
                    s[a] += xv0 * wv;
                    s[8 + a] += xv1 * wv;
                }
            }
            for (int a = 0; a < 8; a++) { y0[j0 + a] = s[a]; y1[j0 + a] = s[8 + a]; }
        }
        for (; j0 < N; j0++) {
            const float *wr = W + (size_t)j0 * K;
            float s0 = 0.0f, s1 = 0.0f;
            for (int k = 0; k < K; k++) { s0 += x0[k] * wr[k]; s1 += x1[k] * wr[k]; }
            y0[j0] = s0; y1[j0] = s1;
        }
    }
    for (int i = M - (M & 1); i < M; i++) {   /* odd last row */
        const float *xr = x + (size_t)i * K;
        float *yr = y + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            const float *wr = W + (size_t)j * K;
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += xr[k] * wr[k];
            yr[j] = s;
        }
    }
#elif defined(__AVX2__) && defined(__FMA__)
    /* Register-blocked: 2 output rows x 8 output cols in flight (16 ymm
     * accs — no spills). The W row is loaded once per k-vector and reused
     * for BOTH rows (halves W re-reads vs 1-row streaming), and the x row
     * is read once per 8 columns. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M - 1; i += 2) {
        const float *x0 = x + (size_t)i * K;
        const float *x1 = x + (size_t)(i + 1) * K;
        float *y0 = y + (size_t)i * N;
        float *y1 = y + (size_t)(i + 1) * N;
        int j0 = 0;
        for (; j0 + 8 <= N; j0 += 8) {
            __m256 acc[16];
            for (int a = 0; a < 16; a++) acc[a] = _mm256_setzero_ps();
            const float *wr[8];
            for (int a = 0; a < 8; a++) wr[a] = W + (size_t)(j0 + a) * K;
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 xv0 = _mm256_loadu_ps(x0 + k);
                __m256 xv1 = _mm256_loadu_ps(x1 + k);
                #pragma unroll(8)
                for (int a = 0; a < 8; a++) {
                    __m256 wv = _mm256_loadu_ps(wr[a] + k);
                    acc[a]     = _mm256_fmadd_ps(xv0, wv, acc[a]);
                    acc[8 + a] = _mm256_fmadd_ps(xv1, wv, acc[8 + a]);
                }
            }
            float s[16];
            for (int a = 0; a < 16; a++) {
                float t[8];
                _mm256_storeu_ps(t, acc[a]);
                s[a] = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
            }
            for (; k < K; k++) {
                float xv0 = x0[k], xv1 = x1[k];
                for (int a = 0; a < 8; a++) {
                    float wv = wr[a][k];
                    s[a] += xv0 * wv;
                    s[8 + a] += xv1 * wv;
                }
            }
            for (int a = 0; a < 8; a++) { y0[j0 + a] = s[a]; y1[j0 + a] = s[8 + a]; }
        }
        for (; j0 < N; j0++) {
            const float *wr = W + (size_t)j0 * K;
            float s0 = 0.0f, s1 = 0.0f;
            for (int k = 0; k < K; k++) { s0 += x0[k] * wr[k]; s1 += x1[k] * wr[k]; }
            y0[j0] = s0; y1[j0] = s1;
        }
    }
    for (int i = M - (M & 1); i < M; i++) {   /* odd last row */
        const float *xr = x + (size_t)i * K;
        float *yr = y + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            const float *wr = W + (size_t)j * K;
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += xr[k] * wr[k];
            yr[j] = s;
        }
    }
#else
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const float *xr = x + (size_t)i * K;
        float *yr = y + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            const float *wr = W + (size_t)j * K;
            for (int k = 0; k < K; k++) s += xr[k] * wr[k];
            yr[j] = s;
        }
    }
#endif
}

void wubu_sd_conv2d(const float *x, int N, int C_in, int H, int W,
                    const float *w, const float *b,
                    int C_out, int KH, int KW, int stride,
                    int pad_h, int pad_w,
                    float *y, int *H_out, int *W_out)
{
    const int H_out_ = (H + 2 * pad_h - KH) / stride + 1;
    const int W_out_ = (W + 2 * pad_w - KW) / stride + 1;
    if (H_out) *H_out = H_out_;
    if (W_out) *W_out = W_out_;

    /* All kernels (1x1 and 3x3) go through im2col + GEMM — the 1x1
     * pixel-outer path has strided reads that die at 512x512x512ch.
     * xcol is [M][K] ROW-major (M = T*W_out pixels, K = C_in*KH*KW),
     * k = ci*KH*KW + kh*KW + kw (kw innermost) matching w[co][k]. */
    const int K = C_in * KH * KW;
    const int T = 8; /* output rows per tile; keeps T*W_out*K*4 in L3 */
    float *xcol = (float *)malloc((size_t)T * W_out_ * K * sizeof(float));
    float *yt   = (float *)malloc((size_t)T * W_out_ * C_out * sizeof(float));
    if (!xcol || !yt) { free(xcol); free(yt); return; }
    for (int n = 0; n < N; n++) {
        const float *xn = x + (size_t)n * C_in * H * W;
        float *yn = y + (size_t)n * C_out * H_out_ * W_out_;
        for (int oh0 = 0; oh0 < H_out_; oh0 += T) {
            int T_eff = (oh0 + T < H_out_) ? T : (H_out_ - oh0);
            /* build im2col tile: xcol[m][k], m = t*W_out + ow */
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < K; k++) {
                int ci = k / (KH * KW);
                int kh = (k / KW) % KH;
                int kw = k % KW;
                const float *xr = xn + (size_t)ci * H * W;
                for (int t = 0; t < T_eff; t++) {
                    int ih = (oh0 + t) * stride - pad_h + kh;
                    float *xcolm = xcol + ((size_t)t * W_out_) * K + k;
                    if (ih < 0 || ih >= H) {
                        /* zero padding — MUST be explicit (no garbage) */
                        for (int ow = 0; ow < W_out_; ow++)
                            xcolm[(size_t)ow * K] = 0.0f;
                        continue;
                    }
                    const float *xrow = xr + (size_t)ih * W;
                    int iw0 = -pad_w + kw;
                    if (iw0 >= 0 && iw0 + W_out_ <= W) {
                        for (int ow = 0; ow < W_out_; ow++)
                            xcolm[(size_t)ow * K] = xrow[iw0 + ow];
                    } else {
                        for (int ow = 0; ow < W_out_; ow++) {
                            int iw = iw0 + ow;
                            xcolm[(size_t)ow * K] = (iw >= 0 && iw < W) ? xrow[iw] : 0.0f;
                        }
                    }
                }
            }
            /* yt[m][co] = sum_k xcol[m][k] * w[co][k]:
             * matmul_nt(xcol, w, M=T*W_out, K, N=C_out, yt) */
            wubu_sd_matmul_nt(xcol, w, T_eff * W_out_, K, C_out, yt);
            /* add bias + store (yt is [M][C_out] row-major) */
            #pragma omp parallel for schedule(static)
            for (int co = 0; co < C_out; co++) {
                float bias = b ? b[co] : 0.0f;
                float *yrow = yn + (size_t)co * H_out_ * W_out_ + (size_t)oh0 * W_out_;
                for (int p = 0; p < T_eff * W_out_; p++)
                    yrow[p] = yt[(size_t)p * C_out + co] + bias;
            }
        }
    }
    free(xcol); free(yt);
}

void wubu_sd_groupnorm(const float *x, int N, int C, int H, int W,
                       int G, float eps,
                       const float *g, const float *b, float *y)
{
    const int ch_per_g = C / G;
    const int hw = H * W;
    for (int n = 0; n < N; n++) {
        const float *xn = x + (size_t)n * C * hw;
        float *yn = y + (size_t)n * C * hw;
        for (int gg = 0; gg < G; gg++) {
            double sum = 0.0, sumsq = 0.0;
            const int count = ch_per_g * hw;
            for (int c = 0; c < ch_per_g; c++) {
                const float *xr = xn + (size_t)(gg * ch_per_g + c) * hw;
                for (int p = 0; p < hw; p++) {
                    sum += xr[p];
                    sumsq += (double)xr[p] * xr[p];
                }
            }
            const float mean = (float)(sum / count);
            const float var = (float)(sumsq / count - (double)mean * mean);
            const float inv = 1.0f / sqrtf(var + eps);
            for (int c = 0; c < ch_per_g; c++) {
                const int cc = gg * ch_per_g + c;
                const float *xr = xn + (size_t)cc * hw;
                float *yr = yn + (size_t)cc * hw;
                const float gs = g ? g[cc] : 1.0f;
                const float bs = b ? b[cc] : 0.0f;
                for (int p = 0; p < hw; p++)
                    yr[p] = (xr[p] - mean) * inv * gs + bs;
            }
        }
    }
}

void wubu_sd_upsample2x(const float *x, int N, int C, int H, int W, float *y)
{
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++) {
            const float *xr = x + (size_t)(n * C + c) * H * W;
            float *yr = y + (size_t)(n * C + c) * (2 * H) * (2 * W);
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    const float v = xr[(size_t)h * W + w];
                    yr[(size_t)(2 * h) * (2 * W) + 2 * w] = v;
                    yr[(size_t)(2 * h) * (2 * W) + 2 * w + 1] = v;
                    yr[(size_t)(2 * h + 1) * (2 * W) + 2 * w] = v;
                    yr[(size_t)(2 * h + 1) * (2 * W) + 2 * w + 1] = v;
                }
        }
}

void wubu_sd_silu(float *x, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* ---- quantized linear (weights stay in the mmap'd blob) ----
 * GGUF stores weights in ggml convention: dims[0] is the innermost
 * (contraction K) dim. A [N][K] Linear weight is laid out as K contiguous
 * per column n: element (k,n) at offset n*K + k.
 *
 * F16:  2 bytes/elem, half -> float via gguf_f16_to_f32.
 * Q4_0: blocks of 32 elements: [f16 d][16 x uint8 nibbles] = 18 bytes.
 *       value = (nibble - 8) * d. ggml layout: elements 0-15 = LOW nibbles
 *       of bytes 0-15, elements 16-31 = HIGH nibbles of bytes 0-15
 *       (NOT interleaved — verified against sd.cpp's dequantize_row_q4_0
 *       and its dequantized gn weights, 2026-08-09).
 */
/* Dequantize a [K][N] column-major weight (F16 or Q4_0) to F32 scratch.
 * Read ONCE per call (bandwidth-optimal), then the GEMM is pure FMA —
 * the old loop dequantized each column M times (M=4096 attn rows = 4096x
 * redundant work). Scratch is per-call, freed immediately — never a
 * persistent F32 cache (the user's law: weights stay in the blob). */
static float *sd_dequant_w(const void *W, int type, int K, int N) {
    float *f32 = (float *)malloc((size_t)K * N * sizeof(float));
    if (!f32) return NULL;
    if (type == 1) { /* F16: bit-math convert, column-major [K][N] */
        const uint16_t *w = (const uint16_t *)W;
        #pragma omp parallel for
        for (int j = 0; j < N; j++) {
            const uint16_t *wj = w + (size_t)j * K;
            float *fj = f32 + (size_t)j * K;
            int k = 0;
#if defined(__AVX512F__) && defined(__FMA__)
            for (; k + 16 <= K; k += 16) {
                __m512 hv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(wj + k)));
                _mm512_storeu_ps(fj + k, hv);
            }
#endif
#if defined(__AVX2__) && defined(__FMA__)
            for (; k + 8 <= K; k += 8) {
                __m256 hv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(wj + k)));
                _mm256_storeu_ps(fj + k, hv);
            }
#endif
            for (; k < K; k++) {
                uint32_t h = wj[k];
                uint32_t u = ((h & 0x8000u) << 16)
                           | ((((h >> 10) & 0x1Fu) + 112u) << 23)
                           | ((h & 0x3FFu) << 13);
                float v;
                memcpy(&v, &u, 4);
                fj[k] = v;
            }
        }
        } else if (type == 2) { /* Q4_0: 32-elem blocks, 18 bytes */
        const uint8_t *w = (const uint8_t *)W;
        const int nblk = (K + 31) / 32;
        #pragma omp parallel for
        for (int j = 0; j < N; j++) {
            const uint8_t *wj = w + (size_t)j * nblk * 18;
            float *fj = f32 + (size_t)j * K;
            int b = 0;
#if defined(__AVX512F__) && defined(__FMA__)
            const __m512i zlo_mask = _mm512_set1_epi8(0x0F);
            const __m512 zeight = _mm512_set1_ps(8.0f);
            for (; b + 1 <= nblk; b++) {
                const uint8_t *blk = wj + (size_t)b * 18;
                float d = wubu_sd_f16_to_f32(*(const uint16_t *)blk);
                const uint8_t *qs = blk + 2;
                __m512 dv = _mm512_set1_ps(d);
                __m128i q0 = _mm_loadu_si128((const __m128i *)qs);
                __m128i lo = _mm_and_si128(q0, _mm512_castsi512_si128(zlo_mask));
                __m128i hi = _mm_and_si128(_mm_srli_epi16(q0, 4), _mm512_castsi512_si128(zlo_mask));
                __m512 f0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo));
                __m512 f1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi));
                _mm512_storeu_ps(fj + (size_t)b * 32 + 0,  _mm512_mul_ps(_mm512_sub_ps(f0, zeight), dv));
                _mm512_storeu_ps(fj + (size_t)b * 32 + 16, _mm512_mul_ps(_mm512_sub_ps(f1, zeight), dv));
            }
#endif
#if defined(__AVX2__) && defined(__FMA__)
            const __m128i lo_mask = _mm_set1_epi8(0x0F);
            const __m256 eight = _mm256_set1_ps(8.0f);
            for (; b + 1 <= nblk; b++) {
                const uint8_t *blk = wj + (size_t)b * 18;
                float d = wubu_sd_f16_to_f32(*(const uint16_t *)blk);
                const uint8_t *qs = blk + 2;
                __m256 dv = _mm256_set1_ps(d);
                /* ggml Q4_0: elements 0-15 = LOW nibbles of qs[0..15],
                 * elements 16-31 = HIGH nibbles of qs[0..15]. */
                __m128i q0 = _mm_loadu_si128((const __m128i *)qs);
                __m128i lo = _mm_and_si128(q0, lo_mask);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(q0, 4), lo_mask);
                __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
                __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)));
                __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
                __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)));
                _mm256_storeu_ps(fj + (size_t)b * 32 + 0,  _mm256_mul_ps(_mm256_sub_ps(f0, eight), dv));
                _mm256_storeu_ps(fj + (size_t)b * 32 + 8,  _mm256_mul_ps(_mm256_sub_ps(f1, eight), dv));
                _mm256_storeu_ps(fj + (size_t)b * 32 + 16, _mm256_mul_ps(_mm256_sub_ps(f2, eight), dv));
                _mm256_storeu_ps(fj + (size_t)b * 32 + 24, _mm256_mul_ps(_mm256_sub_ps(f3, eight), dv));
            }
#endif
            for (; b < nblk; b++) {   /* partial tail block (K % 32) */
                const uint8_t *blk = wj + (size_t)b * 18;
                float d = wubu_sd_f16_to_f32(*(const uint16_t *)blk);
                const uint8_t *qs = blk + 2;
                int base = b * 32;
                int rem = K - base; if (rem > 32) rem = 32;
                for (int l = 0; l < rem; l++) {
                    /* ggml Q4_0 layout: elements 0-15 = LOW nibbles of the 16
                     * bytes, elements 16-31 = HIGH nibbles (NOT interleaved). */
                    int8_t q = (l < 16) ? (int8_t)(qs[l] & 0x0F) - 8
                                        : (int8_t)(qs[l - 16] >> 4) - 8;
                    fj[base + l] = (float)q * d;
                }
            }
        }
    } else {
        free(f32);
        return NULL;
    }
    return f32;
}

/* ---- startup: flush denormals (FTZ+DAZ). Denormal FP ops trigger
 * microcode assists on AMD/Intel — for neural activations they are pure
 * noise, so flush-to-zero is free speed (wuburvc 50-fixes #36). */
__attribute__((constructor)) static void sd_mxcsr_ftz(void) {
    unsigned int mxcsr = 0;
    __asm__ __volatile__("stmxcsr %0" : "=m"(mxcsr));
    mxcsr |= 0x8040u;  /* FTZ (bit 15) | DAZ (bit 6) */
    __asm__ __volatile__("ldmxcsr %0" : : "m"(mxcsr));
}

/* f32 -> f16 (round-to-nearest-even), for F16 xcol/scratch paths. */
static inline uint16_t wubu_sd_f32_to_f16(float f) {
    union { float f; uint32_t u; } u = { f };
    uint32_t x = u.u;
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFFu;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);          /* inf */
    if (e <= 0) {                                            /* subnormal/zero */
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u;
        m >>= (14 - e);
        return (uint16_t)(sign | (m >> 13));
    }
    uint32_t half = sign | ((uint32_t)e << 10) | (m >> 13);
    if (m & 0x1000u) half++;                                 /* round */
    return (uint16_t)half;
}

/* ---- Quantized GEMM: weights stay in the RAW mmap'd blob, never dequantized
 * to a scratch. F16 weights are read as f16 (cvtph on load); Q4_0 blocks are
 * dequantized IN REGISTERS (vec_dot pattern) and reused across the 2x8
 * register block. This is both the user law ("never dequantize — quantized
 * in-kernel only") and the speedup: raw Q4_0 is 0.5625 B/elem vs 2 (f16
 * scratch) / 4 (f32 scratch) of memory traffic, plus zero per-call dequant
 * pass. x is F32 (linears) or F16 (conv xcol) — xf16 selects the load. */

static inline void sd_dequant_block_q40(const uint8_t *blk, __m256 out[4]) {
    float d = wubu_sd_f16_to_f32(*(const uint16_t *)blk);
    const uint8_t *qs = blk + 2;
    __m256 dv = _mm256_set1_ps(d);
    __m256 eight = _mm256_set1_ps(8.0f);
    __m128i q0 = _mm_loadu_si128((const __m128i *)qs);
    __m128i lo = _mm_and_si128(q0, _mm_set1_epi8(0x0F));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(q0, 4), _mm_set1_epi8(0x0F));
    out[0] = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)), eight), dv);
    out[1] = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8))), eight), dv);
    out[2] = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)), eight), dv);
    out[3] = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8))), eight), dv);
}

/* Scalar dot of x-row (F32 or F16) with one W column (F16 or Q4_0 raw) —
 * used for the j/row tails only. */
static inline float sd_dot_col(const char *xr, int xf16, int K, const void *W,
                               int wtype, int col, int nblk) {
    float s = 0.0f;
    if (wtype == 1) {
        const uint16_t *wr = (const uint16_t *)W + (size_t)col * K;
        for (int k = 0; k < K; k++) {
            float xv = xf16 ? wubu_sd_f16_to_f32(((const uint16_t *)xr)[k])
                            : ((const float *)xr)[k];
            s += xv * wubu_sd_f16_to_f32(wr[k]);
        }
    } else {
        const uint8_t *wq = (const uint8_t *)W + (size_t)col * nblk * 18;
        for (int k = 0; k < K; k++) {
            float xv = xf16 ? wubu_sd_f16_to_f32(((const uint16_t *)xr)[k])
                            : ((const float *)xr)[k];
            int kb = k / 32, l = k % 32;
            const uint8_t *blk = wq + (size_t)kb * 18;
            float d = wubu_sd_f16_to_f32(*(const uint16_t *)blk);
            const uint8_t *qs = blk + 2;
            int8_t q = (l < 16) ? (int8_t)(qs[l] & 0x0F) - 8
                                : (int8_t)(qs[l - 16] >> 4) - 8;
            s += xv * ((float)q * d);
        }
    }
    return s;
}

void wubu_sd_matmul_q(const void *x, int xf16, const void *W, int wtype,
                      int M, int K, int N, float *y)
{
    const int esz = xf16 ? 2 : 4;
    const int nblk = (K + 31) / 32;
    const int nblk_full = K / 32;
#if defined(__AVX2__) && defined(__FMA__)
    if (N <= 8) {
        /* small-N path (e.g. VAE conv_out C->3): vectorize over k, one ymm
         * acc per output column — the j-block loop never fires for N<8 and
         * the scalar tail would cost ~20x more. 2 rows in flight. */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M - 1; i += 2) {
            const char *x0c = (const char *)x + (size_t)i * K * esz;
            const char *x1c = (const char *)x + (size_t)(i + 1) * K * esz;
            float *y0 = y + (size_t)i * N;
            float *y1 = y + (size_t)(i + 1) * N;
            __m256 acc[16];
            for (int a = 0; a < 16; a++) acc[a] = _mm256_setzero_ps();
            if (wtype == 1) {  /* F16 */
                const uint16_t *wr[8];
                for (int j = 0; j < N; j++) wr[j] = (const uint16_t *)W + (size_t)j * K;
                int k = 0;
                for (; k + 8 <= K; k += 8) {
                    __m256 xv0 = xf16
                        ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x0c + (size_t)k * 2)))
                        : _mm256_loadu_ps((const float *)(x0c + (size_t)k * 4));
                    __m256 xv1 = xf16
                        ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x1c + (size_t)k * 2)))
                        : _mm256_loadu_ps((const float *)(x1c + (size_t)k * 4));
                    for (int j = 0; j < N; j++) {
                        __m256 wv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(wr[j] + k)));
                        acc[j]     = _mm256_fmadd_ps(xv0, wv, acc[j]);
                        acc[8 + j] = _mm256_fmadd_ps(xv1, wv, acc[8 + j]);
                    }
                }
                for (; k < K; k++) {
                    float xv0 = xf16 ? wubu_sd_f16_to_f32(((const uint16_t *)x0c)[k])
                                     : ((const float *)x0c)[k];
                    float xv1 = xf16 ? wubu_sd_f16_to_f32(((const uint16_t *)x1c)[k])
                                     : ((const float *)x1c)[k];
                    for (int j = 0; j < N; j++) {
                        float wv = wubu_sd_f16_to_f32(wr[j][k]);
                        acc[j] = _mm256_add_ps(acc[j], _mm256_set1_ps(xv0 * wv));
                        acc[8 + j] = _mm256_add_ps(acc[8 + j], _mm256_set1_ps(xv1 * wv));
                    }
                }
            } else {  /* Q4_0 */
                const uint8_t *wq[8];
                for (int j = 0; j < N; j++) wq[j] = (const uint8_t *)W + (size_t)j * nblk * 18;
                for (int kb = 0; kb < nblk_full; kb++) {
                    for (int j = 0; j < N; j++) {
                        __m256 w[4];
                        sd_dequant_block_q40(wq[j] + (size_t)kb * 18, w);
                        for (int s = 0; s < 4; s++) {
                            __m256 xv0 = xf16
                                ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x0c + (size_t)(kb * 32 + s * 8) * 2)))
                                : _mm256_loadu_ps((const float *)(x0c + (size_t)(kb * 32 + s * 8) * 4));
                            __m256 xv1 = xf16
                                ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x1c + (size_t)(kb * 32 + s * 8) * 2)))
                                : _mm256_loadu_ps((const float *)(x1c + (size_t)(kb * 32 + s * 8) * 4));
                            acc[j]     = _mm256_fmadd_ps(xv0, w[s], acc[j]);
                            acc[8 + j] = _mm256_fmadd_ps(xv1, w[s], acc[8 + j]);
                        }
                    }
                }
                for (int k = nblk_full * 32; k < K; k++) {
                    float xv0 = xf16 ? wubu_sd_f16_to_f32(((const uint16_t *)x0c)[k])
                                     : ((const float *)x0c)[k];
                    float xv1 = xf16 ? wubu_sd_f16_to_f32(((const uint16_t *)x1c)[k])
                                     : ((const float *)x1c)[k];
                    for (int j = 0; j < N; j++) {
                        int kb = k / 32, l = k % 32;
                        const uint8_t *blk = wq[j] + (size_t)kb * 18;
                        float d = wubu_sd_f16_to_f32(*(const uint16_t *)blk);
                        const uint8_t *qs = blk + 2;
                        int8_t q = (l < 16) ? (int8_t)(qs[l] & 0x0F) - 8
                                            : (int8_t)(qs[l - 16] >> 4) - 8;
                        acc[j] = _mm256_add_ps(acc[j], _mm256_set1_ps(xv0 * ((float)q * d)));
                        acc[8 + j] = _mm256_add_ps(acc[8 + j], _mm256_set1_ps(xv1 * ((float)q * d)));
                    }
                }
            }
            for (int j = 0; j < N; j++) {
                float t0[8], t1[8];
                _mm256_storeu_ps(t0, acc[j]);
                _mm256_storeu_ps(t1, acc[8 + j]);
                y0[j] = t0[0] + t0[1] + t0[2] + t0[3] + t0[4] + t0[5] + t0[6] + t0[7];
                y1[j] = t1[0] + t1[1] + t1[2] + t1[3] + t1[4] + t1[5] + t1[6] + t1[7];
            }
        }
        for (int i = M - (M & 1); i < M; i++) {
            const char *xr = (const char *)x + (size_t)i * K * esz;
            float *yr = y + (size_t)i * N;
            for (int j = 0; j < N; j++)
                yr[j] = sd_dot_col(xr, xf16, K, W, wtype, j, nblk);
        }
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M - 1; i += 2) {
        const char *x0c = (const char *)x + (size_t)i * K * esz;
        const char *x1c = (const char *)x + (size_t)(i + 1) * K * esz;
        float *y0 = y + (size_t)i * N;
        float *y1 = y + (size_t)(i + 1) * N;
        int j0 = 0;
        for (; j0 + 8 <= N; j0 += 8) {
            __m256 acc[16];
            for (int a = 0; a < 16; a++) acc[a] = _mm256_setzero_ps();
            float s0[8] = {0}, s1[8] = {0};
            if (wtype == 1) {  /* F16: raw blob, cvtph on load */
                const uint16_t *wr[8];
                for (int a = 0; a < 8; a++) wr[a] = (const uint16_t *)W + (size_t)(j0 + a) * K;
                int k = 0;
                for (; k + 8 <= K; k += 8) {
                    __m256 xv0 = xf16
                        ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x0c + (size_t)k * 2)))
                        : _mm256_loadu_ps((const float *)(x0c + (size_t)k * 4));
                    __m256 xv1 = xf16
                        ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x1c + (size_t)k * 2)))
                        : _mm256_loadu_ps((const float *)(x1c + (size_t)k * 4));
                    #pragma unroll(8)
                    for (int a = 0; a < 8; a++) {
                        __m256 wv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(wr[a] + k)));
                        acc[a]     = _mm256_fmadd_ps(xv0, wv, acc[a]);
                        acc[8 + a] = _mm256_fmadd_ps(xv1, wv, acc[8 + a]);
                    }
                }
                for (; k < K; k++) {
                    float xv0 = xf16 ? wubu_sd_f16_to_f32(((const uint16_t *)x0c)[k])
                                     : ((const float *)x0c)[k];
                    float xv1 = xf16 ? wubu_sd_f16_to_f32(((const uint16_t *)x1c)[k])
                                     : ((const float *)x1c)[k];
                    for (int a = 0; a < 8; a++) {
                        float wv = wubu_sd_f16_to_f32(wr[a][k]);
                        s0[a] += xv0 * wv;
                        s1[a] += xv1 * wv;
                    }
                }
            } else {  /* Q4_0: raw blob, per-block dequant in registers */
                const uint8_t *wq[8];
                for (int a = 0; a < 8; a++) wq[a] = (const uint8_t *)W + (size_t)(j0 + a) * nblk * 18;
                for (int kb = 0; kb < nblk_full; kb++) {
                    for (int a = 0; a < 8; a++) {
                        __m256 w[4];
                        sd_dequant_block_q40(wq[a] + (size_t)kb * 18, w);
                        for (int s = 0; s < 4; s++) {
                            __m256 xv0 = xf16
                                ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x0c + (size_t)(kb * 32 + s * 8) * 2)))
                                : _mm256_loadu_ps((const float *)(x0c + (size_t)(kb * 32 + s * 8) * 4));
                            __m256 xv1 = xf16
                                ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x1c + (size_t)(kb * 32 + s * 8) * 2)))
                                : _mm256_loadu_ps((const float *)(x1c + (size_t)(kb * 32 + s * 8) * 4));
                            acc[a]     = _mm256_fmadd_ps(xv0, w[s], acc[a]);
                            acc[8 + a] = _mm256_fmadd_ps(xv1, w[s], acc[8 + a]);
                        }
                    }
                }
                /* scalar tail: k in [nblk_full*32, K) — partial last block */
                for (int k = nblk_full * 32; k < K; k++) {
                    float xv0 = xf16 ? wubu_sd_f16_to_f32(((const uint16_t *)x0c)[k])
                                     : ((const float *)x0c)[k];
                    float xv1 = xf16 ? wubu_sd_f16_to_f32(((const uint16_t *)x1c)[k])
                                     : ((const float *)x1c)[k];
                    for (int a = 0; a < 8; a++) {
                        int kb = k / 32, l = k % 32;
                        const uint8_t *blk = wq[a] + (size_t)kb * 18;
                        float d = wubu_sd_f16_to_f32(*(const uint16_t *)blk);
                        const uint8_t *qs = blk + 2;
                        int8_t q = (l < 16) ? (int8_t)(qs[l] & 0x0F) - 8
                                            : (int8_t)(qs[l - 16] >> 4) - 8;
                        s0[a] += xv0 * ((float)q * d);
                        s1[a] += xv1 * ((float)q * d);
                    }
                }
            }
            for (int a = 0; a < 8; a++) {
                float t0[8], t1[8];
                _mm256_storeu_ps(t0, acc[a]);
                _mm256_storeu_ps(t1, acc[8 + a]);
                float h0 = t0[0] + t0[1] + t0[2] + t0[3] + t0[4] + t0[5] + t0[6] + t0[7];
                float h1 = t1[0] + t1[1] + t1[2] + t1[3] + t1[4] + t1[5] + t1[6] + t1[7];
                y0[j0 + a] = h0 + s0[a];
                y1[j0 + a] = h1 + s1[a];
            }
        }
        for (; j0 < N; j0++) {
            y0[j0] = sd_dot_col(x0c, xf16, K, W, wtype, j0, nblk);
            y1[j0] = sd_dot_col(x1c, xf16, K, W, wtype, j0, nblk);
        }
    }
    for (int i = M - (M & 1); i < M; i++) {
        const char *xr = (const char *)x + (size_t)i * K * esz;
        float *yr = y + (size_t)i * N;
        for (int j = 0; j < N; j++)
            yr[j] = sd_dot_col(xr, xf16, K, W, wtype, j, nblk);
    }
#else
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            y[(size_t)i * N + j] = sd_dot_col((const char *)x + (size_t)i * K * esz,
                                              xf16, K, W, wtype, j, nblk);
#endif
}

void wubu_sd_linear_q(const float *x, const void *W, int type,
                      int M, int K, int N, float *y) {
    if (type == 0) { /* GGML_TYPE_F32 (0): plain matmul */
        wubu_sd_matmul_nt(x, (const float *)W, M, K, N, y);
        return;
    }
    /* F16 / Q4_0: quantized GEMM on the RAW blob — no dequant scratch.
     * x stays F32 (linears). */
    wubu_sd_matmul_q(x, 0, W, type, M, K, N, y);
}

void wubu_sd_linear_qb(const float *x, const void *W, int type,
                       const float *b, int M, int K, int N, float *y) {
    wubu_sd_linear_q(x, W, type, M, K, N, y);
    if (b) {
        #pragma omp parallel for
        for (int i = 0; i < M * N; i++) y[i] += b[i % N];
    }
}

void wubu_sd_conv2d_q(const float *x, int N, int C_in, int H, int W,
                      const void *w, int wtype, const float *b,
                      int C_out, int KH, int KW, int stride,
                      int pad_h, int pad_w,
                      float *y, int *H_out, int *W_out) {
    const int H_out_ = (H + 2 * pad_h - KH) / stride + 1;
    const int W_out_ = (W + 2 * pad_w - KW) / stride + 1;
    if (H_out) *H_out = H_out_;
    if (W_out) *W_out = W_out_;
    /* im2col in the WEIGHT's k-order (ggml 4D column-major: element
     * (kh,kw,ci,co) at offset kh + KW*kw + KH*KW*ci + K*co — kh is
     * INNERMOST (stride 1), kw stride KW, ci stride KH*KW. So the
     * per-column K order is k = ci*KH*KW + kw*KH + kh. Then
     * y[co][p] = sum_k W[co][k] * xcol[k][p] is exactly wubu_sd_linear_q
     * with x = xcol^T [T*W_out][K], W = raw [K][C_out]. */
    const int K = C_in * KH * KW;
    /* Adaptive tile rows: keep the F16 xcol tile (T*W_out*K*2B) under ~12MB
     * so the GEMM's x re-reads stay in L3 (16MB here). The 512x512 VAE
     * tiles at T=8 were 18.9MB (DRAM re-reads); small convs keep T=8 for
     * GEMM efficiency (larger M per call). */
    int T = 8;
    if ((size_t)T * W_out_ * K * 2 > (12u << 20)) T = 4;
    /* Dequant the conv weight ONCE for all tiles (F16 scratch, per-call).
     * The im2col's K-order matches the weight's raw [K][C_out] layout:
     * k = ci*KH*KW + kw*KH + kh (kh innermost). */
    /* xcol is [T*W_out][K] ROW-major in F16 — matmul_q(xf16=1) expects
     * x[M][K] with x[m*K + k]; a [K][M] column-major layout would
     * misalign. F16 halves the xcol memory traffic (the big 512x512
     * VAE convs build >1GB of xcol). Weights stay RAW — the GEMM
     * dequants F16/Q4_0 in registers. */
    uint16_t *xcol = NULL;
    float *yt = NULL;
    if (posix_memalign((void **)&xcol, 64, (size_t)T * W_out_ * K * sizeof(uint16_t)) != 0) xcol = NULL;
    if (posix_memalign((void **)&yt, 64, (size_t)C_out * T * W_out_ * sizeof(float)) != 0) yt = NULL;
    if (!xcol || !yt) { free(xcol); free(yt); return; }
    for (int n = 0; n < N; n++) {
        const float *xn = x + (size_t)n * C_in * H * W;
        float *yn = y + (size_t)n * C_out * H_out_ * W_out_;
        for (int oh0 = 0; oh0 < H_out_; oh0 += T) {
            int T_eff = (oh0 + T < H_out_) ? T : (H_out_ - oh0);
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < K; k++) {
                int ci = k / (KH * KW);
                int kw = (k / KH) % KW;
                int kh = k % KH;
                const float *xr = xn + (size_t)ci * H * W;
                /* xcol[m][k], m = t*W_out + ow */
                for (int t = 0; t < T_eff; t++) {
                    int ih = (oh0 + t) * stride - pad_h + kh;
                    uint16_t *xcolm = xcol + ((size_t)t * W_out_) * K + k;
                    if (ih < 0 || ih >= H) {
                        for (int ow = 0; ow < W_out_; ow++)
                            xcolm[(size_t)ow * K] = 0;
                        continue;
                    }
                    const float *xrow = xr + (size_t)ih * W;
                    int iw0 = -pad_w + kw;
                    if (iw0 >= 0 && iw0 + W_out_ <= W) {
                        for (int ow = 0; ow < W_out_; ow++)
                            xcolm[(size_t)ow * K] = wubu_sd_f32_to_f16(xrow[iw0 + ow]);
                    } else {
                        for (int ow = 0; ow < W_out_; ow++) {
                            int iw = iw0 + ow;
                            xcolm[(size_t)ow * K] = (iw >= 0 && iw < W)
                                ? wubu_sd_f32_to_f16(xrow[iw]) : 0;
                        }
                    }
                }
            }
            /* yt[co][p] = sum_k W[co][k] * xcol[k][p]:
             * matmul_q(x=xcol^T [T*W_out][K] F16, W=raw [K][C_out])
             * NOTE: output is [M][N] row-major = yt[p*C_out + co] */
            wubu_sd_matmul_q(xcol, 1, w, wtype, T_eff * W_out_, K, C_out, yt);
            #pragma omp parallel for schedule(static)
            for (int co = 0; co < C_out; co++) {
                float bias = b ? b[co] : 0.0f;
                float *yrow = yn + (size_t)co * H_out_ * W_out_ + (size_t)oh0 * W_out_;
                for (int p = 0; p < T_eff * W_out_; p++)
                    yrow[p] = yt[(size_t)p * C_out + co] + bias;
            }
        }
    }
    free(xcol); free(yt);
}

void wubu_sd_downsample2x(const float *x, int N, int C, int H, int W, float *y)
{
    const int H2 = H / 2, W2 = W / 2;
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++) {
            const float *xr = x + (size_t)(n * C + c) * H * W;
            float *yr = y + (size_t)(n * C + c) * H2 * W2;
            for (int h = 0; h < H2; h++)
                for (int w = 0; w < W2; w++) {
                    float mx = -INFINITY;
                    for (int dh = 0; dh < 2; dh++)
                        for (int dw = 0; dw < 2; dw++) {
                            float v = xr[(size_t)(2 * h + dh) * W + 2 * w + dw];
                            if (v > mx) mx = v;
                        }
                    yr[(size_t)h * W2 + w] = mx;
                }
        }
}
