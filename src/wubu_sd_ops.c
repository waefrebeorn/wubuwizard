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
#include <time.h>
#if defined(__AVX2__) || defined(__FMA__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
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
#elif defined(__ARM_NEON)
    /* NEON 4x4 register-blocked kernel (Cortex-A72/A76):
     * 4 output rows x 4 output cols in flight = 16 float32x4_t
     * accumulators (v0-v23, no spills). Per k-iteration (4 wide):
     *   4x vld1q_f32  x rows
     *   4x vld1q_f32  W cols
     *   16x fmla      element-wise: acc[r][c] += xr[r][i] * wr[c][i]
     *   2x prfm       pldl1keep dual-issued into FMA slots (free)
     * Reduction: 16x faddp (pairwise) at the end of the k-block.
     * This is the ARM mirror of the AVX2 2x8 kernel — same register
     * blocking philosophy, same x-row reuse across 4 cols. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M - 3; i += 4) {
        const float *xr[4] = {
            x + (size_t)i * K, x + (size_t)(i + 1) * K,
            x + (size_t)(i + 2) * K, x + (size_t)(i + 3) * K};
        float *yr[4] = {
            y + (size_t)i * N, y + (size_t)(i + 1) * N,
            y + (size_t)(i + 2) * N, y + (size_t)(i + 3) * N};
        int j0 = 0;
        for (; j0 + 4 <= N; j0 += 4) {
            const float *wr[4];
            for (int a = 0; a < 4; a++) wr[a] = W + (size_t)(j0 + a) * K;
            float32x4_t acc[16];
            for (int a = 0; a < 16; a++) acc[a] = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 4 <= K; k += 4) {
                float32x4_t xv0 = vld1q_f32(xr[0] + k);
                float32x4_t xv1 = vld1q_f32(xr[1] + k);
                float32x4_t xv2 = vld1q_f32(xr[2] + k);
                float32x4_t xv3 = vld1q_f32(xr[3] + k);
                float32x4_t wv0 = vld1q_f32(wr[0] + k);
                float32x4_t wv1 = vld1q_f32(wr[1] + k);
                float32x4_t wv2 = vld1q_f32(wr[2] + k);
                float32x4_t wv3 = vld1q_f32(wr[3] + k);
                /* prefetch 64 bytes ahead of next k-iter (2 cache lines) */
                __builtin_prefetch(xr[0] + k + 16, 0, 3);
                __builtin_prefetch(wr[0] + k + 16, 0, 3);
                acc[0]  = vfmaq_f32(acc[0],  xv0, wv0);
                acc[1]  = vfmaq_f32(acc[1],  xv0, wv1);
                acc[2]  = vfmaq_f32(acc[2],  xv0, wv2);
                acc[3]  = vfmaq_f32(acc[3],  xv0, wv3);
                acc[4]  = vfmaq_f32(acc[4],  xv1, wv0);
                acc[5]  = vfmaq_f32(acc[5],  xv1, wv1);
                acc[6]  = vfmaq_f32(acc[6],  xv1, wv2);
                acc[7]  = vfmaq_f32(acc[7],  xv1, wv3);
                acc[8]  = vfmaq_f32(acc[8],  xv2, wv0);
                acc[9]  = vfmaq_f32(acc[9],  xv2, wv1);
                acc[10] = vfmaq_f32(acc[10], xv2, wv2);
                acc[11] = vfmaq_f32(acc[11], xv2, wv3);
                acc[12] = vfmaq_f32(acc[12], xv3, wv0);
                acc[13] = vfmaq_f32(acc[13], xv3, wv1);
                acc[14] = vfmaq_f32(acc[14], xv3, wv2);
                acc[15] = vfmaq_f32(acc[15], xv3, wv3);
            }
            float s[16];
            for (int a = 0; a < 16; a++) s[a] = vaddvq_f32(acc[a]);
            for (; k < K; k++) {
                float xv0 = xr[0][k], xv1 = xr[1][k];
                float xv2 = xr[2][k], xv3 = xr[3][k];
                for (int a = 0; a < 4; a++) {
                    float wv = wr[a][k];
                    s[a]      += xv0 * wv;
                    s[4 + a]  += xv1 * wv;
                    s[8 + a]  += xv2 * wv;
                    s[12 + a] += xv3 * wv;
                }
            }
            for (int a = 0; a < 4; a++) {
                yr[0][j0 + a] = s[a];
                yr[1][j0 + a] = s[4 + a];
                yr[2][j0 + a] = s[8 + a];
                yr[3][j0 + a] = s[12 + a];
            }
        }
        for (; j0 < N; j0++) {
            const float *wr = W + (size_t)j0 * K;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int k = 0; k < K; k++) {
                s0 += xr[0][k] * wr[k];
                s1 += xr[1][k] * wr[k];
                s2 += xr[2][k] * wr[k];
                s3 += xr[3][k] * wr[k];
            }
            yr[0][j0] = s0; yr[1][j0] = s1;
            yr[2][j0] = s2; yr[3][j0] = s3;
        }
    }
    for (int i = M - (M & 3); i < M; i++) {   /* odd last rows */
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

/* y[M,N] = x16[M,K] @ W^T[K,N]  with x in F16 (uint16_t halves) and W in
 * F32 — the conv2d_q xcol path. NEON converts F16->F32 in-register
 * (vcvt_f32_f16) so the GEMM reads HALF the x bytes vs the F32-x variant
 * (2 B/elem vs 4). Simple i-outer 4x4 kernel: W stays L2-hot and is
 * SHARED across all 4 OpenMP threads (they stride the same W stream),
 * so W DRAM traffic is ~once per call. x rows (4*K*2B <= 18KB) stay in
 * L1 across the j-loop. This beat a collapse(2)+dynamic L2-blocked
 * variant (v14: -9% conv regression) — the A72's shared 1MB L2 + 4
 * threads make block scheduling overhead dominate. Bit-identical to
 * the v12 verified kernel. ARM-only. */
#if defined(__ARM_NEON)
void wubu_sd_matmul_nt_f16(const uint16_t *x, const float *W, int M, int K,
                           int N, float *y)
{
    /* j-CHUNKED (serial outer over W chunks, i-parallel inner):
     * the i-outer kernel sweeps ALL of W per i-block — for the 512x512
     * VAE upsample conv (N=256, W=K*N*4=2.4MB > 1MB L2) that re-reads
     * W from DRAM M/4=512x per tile (~1.2GB). Chunking N into blocks
     * whose W slice fits L2 (~590KB) and keeping the chunk SERIAL means
     * all 4 threads share ONE W-chunk resident in L2 across their
     * i-ranges — W DRAM drops to ~once per tile. v14's
     * collapse(2)+dynamic failed because threads ran on DIFFERENT
     * j-chunks simultaneously (4×590KB > 1MB L2, thrash); serial-j
     * fixes that. Same 4x4 FMA kernel per (i,j) block — bit-identical
     * accumulation order. */
    int JC = 4 * ((590 * 1024) / (4 * K * 4));  /* ~590KB W slice */
    if (JC < 4) JC = 4;
    JC = (JC / 4) * 4;
    for (int jc = 0; jc < N; jc += JC) {
        int jlim = (jc + JC < N) ? jc + JC : N;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M - 3; i += 4) {
            const uint16_t *xr[4] = {
                x + (size_t)i * K, x + (size_t)(i + 1) * K,
                x + (size_t)(i + 2) * K, x + (size_t)(i + 3) * K};
            float *yr[4] = {
                y + (size_t)i * N, y + (size_t)(i + 1) * N,
                y + (size_t)(i + 2) * N, y + (size_t)(i + 3) * N};
            int j0 = jc;
            for (; j0 + 4 <= jlim; j0 += 4) {
                const float *wr[4];
                for (int a = 0; a < 4; a++) wr[a] = W + (size_t)(j0 + a) * K;
                float32x4_t acc[16];
                for (int a = 0; a < 16; a++) acc[a] = vdupq_n_f32(0.0f);
                int k = 0;
                for (; k + 4 <= K; k += 4) {
                    /* x is F16: load 4 halves -> 4 floats per row */
                    float32x4_t xv0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(xr[0] + k)));
                    float32x4_t xv1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(xr[1] + k)));
                    float32x4_t xv2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(xr[2] + k)));
                    float32x4_t xv3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(xr[3] + k)));
                    float32x4_t wv0 = vld1q_f32(wr[0] + k);
                    float32x4_t wv1 = vld1q_f32(wr[1] + k);
                    float32x4_t wv2 = vld1q_f32(wr[2] + k);
                    float32x4_t wv3 = vld1q_f32(wr[3] + k);
                    __builtin_prefetch(xr[0] + k + 16, 0, 3);
                    __builtin_prefetch(wr[0] + k + 16, 0, 3);
                    acc[0]  = vfmaq_f32(acc[0],  xv0, wv0);
                    acc[1]  = vfmaq_f32(acc[1],  xv0, wv1);
                    acc[2]  = vfmaq_f32(acc[2],  xv0, wv2);
                    acc[3]  = vfmaq_f32(acc[3],  xv0, wv3);
                    acc[4]  = vfmaq_f32(acc[4],  xv1, wv0);
                    acc[5]  = vfmaq_f32(acc[5],  xv1, wv1);
                    acc[6]  = vfmaq_f32(acc[6],  xv1, wv2);
                    acc[7]  = vfmaq_f32(acc[7],  xv1, wv3);
                    acc[8]  = vfmaq_f32(acc[8],  xv2, wv0);
                    acc[9]  = vfmaq_f32(acc[9],  xv2, wv1);
                    acc[10] = vfmaq_f32(acc[10], xv2, wv2);
                    acc[11] = vfmaq_f32(acc[11], xv2, wv3);
                    acc[12] = vfmaq_f32(acc[12], xv3, wv0);
                    acc[13] = vfmaq_f32(acc[13], xv3, wv1);
                    acc[14] = vfmaq_f32(acc[14], xv3, wv2);
                    acc[15] = vfmaq_f32(acc[15], xv3, wv3);
                }
                float s[16];
                for (int a = 0; a < 16; a++) s[a] = vaddvq_f32(acc[a]);
                for (; k < K; k++) {
                    float xv0 = wubu_sd_f16_to_f32(xr[0][k]);
                    float xv1 = wubu_sd_f16_to_f32(xr[1][k]);
                    float xv2 = wubu_sd_f16_to_f32(xr[2][k]);
                    float xv3 = wubu_sd_f16_to_f32(xr[3][k]);
                    for (int a = 0; a < 4; a++) {
                        float wv = wr[a][k];
                        s[a]      += xv0 * wv;
                        s[4 + a]  += xv1 * wv;
                        s[8 + a]  += xv2 * wv;
                        s[12 + a] += xv3 * wv;
                    }
                }
                for (int a = 0; a < 4; a++) {
                    yr[0][j0 + a] = s[a];
                    yr[1][j0 + a] = s[4 + a];
                    yr[2][j0 + a] = s[8 + a];
                    yr[3][j0 + a] = s[12 + a];
                }
            }
            for (; j0 < jlim; j0++) {
                const float *wr = W + (size_t)j0 * K;
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (int k = 0; k < K; k++) {
                    s0 += wubu_sd_f16_to_f32(xr[0][k]) * wr[k];
                    s1 += wubu_sd_f16_to_f32(xr[1][k]) * wr[k];
                    s2 += wubu_sd_f16_to_f32(xr[2][k]) * wr[k];
                    s3 += wubu_sd_f16_to_f32(xr[3][k]) * wr[k];
                }
                yr[0][j0] = s0; yr[1][j0] = s1;
                yr[2][j0] = s2; yr[3][j0] = s3;
            }
        }
        for (int i = M - (M & 3); i < M; i++) {   /* odd last rows */
            const uint16_t *xr = x + (size_t)i * K;
            float *yr = y + (size_t)i * N;
            for (int j = jc; j < jlim; j++) {
                const float *wr = W + (size_t)j * K;
                float s = 0.0f;
                for (int k = 0; k < K; k++) s += wubu_sd_f16_to_f32(xr[k]) * wr[k];
                yr[j] = s;
            }
        }
    }
}
#endif

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

#if defined(__ARM_NEON) && defined(__aarch64__)
/* NEON exp2: 2^x = 2^i * 2^f — exponent insertion (exact) + degree-8
 * Horner for e^(f*ln2) (sub-ULP). FEXPA is SVE-only (ARMv8.2+), A72
 * has no SVE — this poly is the A72 fast path (~10 vector ops/4 elems
 * vs a libm exp2f call per element). llama.cpp PR 7154 pattern. */
static inline float32x4_t exp2q_f32(float32x4_t x) {
    float32x4_t i = vrndnq_f32(x);
    float32x4_t f = vsubq_f32(x, i);
    int32x4_t ie = vcvtq_s32_f32(i);
    ie = vshlq_n_s32(vaddq_s32(ie, vdupq_n_s32(127)), 23);
    float32x4_t pow2i = vreinterpretq_f32_s32(ie);
    float32x4_t u = vmulq_f32(f, vdupq_n_f32(0.6931471805599453f));
    float32x4_t p = vdupq_n_f32(2.4801587301587302e-05f);
    p = vfmaq_f32(vdupq_n_f32(1.984126984126984e-04f), p, u);
    p = vfmaq_f32(vdupq_n_f32(1.3888888888888888e-03f), p, u);
    p = vfmaq_f32(vdupq_n_f32(8.3333333333333332e-03f), p, u);
    p = vfmaq_f32(vdupq_n_f32(4.1666666666666666e-02f), p, u);
    p = vfmaq_f32(vdupq_n_f32(1.6666666666666666e-01f), p, u);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, u);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, u);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, u);
    return vmulq_f32(pow2i, p);
}
#endif /* __ARM_NEON && __aarch64__ */

void wubu_sd_silu(float *x, int n) {
    /* exp2(x*log2e) — NEON polynomial on AArch64, 2x faster than
     * software expf (llama.cpp PR 7154, sub-ULP accuracy). */
    const float log2e = 1.4426950408889634f;
#if defined(__ARM_NEON) && defined(__aarch64__)
    const float32x4_t le = vdupq_n_f32(log2e), one = vdupq_n_f32(1.0f);
    int i = 0;
    #pragma omp parallel for lastprivate(i)
    for (i = 0; i <= n - 8; i += 8) {
        float32x4_t s0 = vld1q_f32(x + i), s1 = vld1q_f32(x + i + 4);
        float32x4_t e0 = exp2q_f32(vmulq_f32(vnegq_f32(s0), le));
        float32x4_t e1 = exp2q_f32(vmulq_f32(vnegq_f32(s1), le));
        vst1q_f32(x + i,     vdivq_f32(s0, vaddq_f32(one, e0)));
        vst1q_f32(x + i + 4, vdivq_f32(s1, vaddq_f32(one, e1)));
    }
    #pragma omp parallel for
    for (int j = i; j < n; j++) {
        float s = x[j];
        x[j] = s / (1.0f + exp2f(-s * log2e));
    }
#else
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float s = x[i];
        x[i] = s / (1.0f + exp2f(-s * log2e));
    }
#endif
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
#if defined(__ARM_NEON)
static inline void sd_dequant_block_q40_neon(const uint8_t *blk,
                                             float32x4_t out[8]);
#endif
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
#if defined(__ARM_NEON)
            /* NEON F16->F32: 8 halves per iteration (2x vld1_u16 + cvt) */
            for (; k + 8 <= K; k += 8) {
                float16x4_t h0 = vreinterpret_f16_u16(vld1_u16(wj + k));
                float16x4_t h1 = vreinterpret_f16_u16(vld1_u16(wj + k + 4));
                vst1q_f32(fj + k,     vcvt_f32_f16(h0));
                vst1q_f32(fj + k + 4, vcvt_f32_f16(h1));
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
#if defined(__ARM_NEON)
            /* NEON Q4_0: 32 elems/block -> 8x float32x4_t, store 8 at a time.
             * Same math as AVX2 path (verified bit-identical 2026-08-10). */
            for (; b + 1 <= nblk; b++) {
                float32x4_t out[8];
                sd_dequant_block_q40_neon(wj + (size_t)b * 18, out);
                float *fb = fj + (size_t)b * 32;
                for (int s = 0; s < 8; s++)
                    vst1q_f32(fb + (size_t)s * 4, out[s]);
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
 * noise, so flush-to-zero is free speed (wuburvc 50-fixes #36).
 * x86_64 only: ARM has no MXCSR (AArch64 FPCR handles this via
 * -ffast-math-adjacent settings; the NEON path doesn't hit denormals
 * as hard and the fpcr write is not worth it here). */
#if defined(__x86_64__)
__attribute__((constructor)) static void sd_mxcsr_ftz(void) {
    unsigned int mxcsr = 0;
    __asm__ __volatile__("stmxcsr %0" : "=m"(mxcsr));
    mxcsr |= 0x8040u;  /* FTZ (bit 15) | DAZ (bit 6) */
    __asm__ __volatile__("ldmxcsr %0" : : "m"(mxcsr));
}
#endif

/* ---- Quantized GEMM: weights stay in the RAW mmap'd blob, never dequantized
 * to a scratch. F16 weights are read as f16 (cvtph on load); Q4_0 blocks are
 * dequantized IN REGISTERS (vec_dot pattern) and reused across the 2x8
 * register block. This is both the user law ("never dequantize — quantized
 * in-kernel only") and the speedup: raw Q4_0 is 0.5625 B/elem vs 2 (f16
 * scratch) / 4 (f32 scratch) of memory traffic, plus zero per-call dequant
 * pass. x is F32 (linears) or F16 (conv xcol) — xf16 selects the load. */

#if defined(__AVX2__) && defined(__FMA__)
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
#endif

/* NEON Q4_0 block dequant: one 18-byte block -> 32 dequantized F32 in
 * 8 float32x4_t vectors. ggml layout: elements 0-15 = LOW nibbles of
 * qs[0..15], elements 16-31 = HIGH nibbles. Value = (nibble - 8) * d.
 * This is the ARM equivalent of sd_dequant_block_q40 (AVX2) — same math,
 * same ordering, verified bit-identical vs x86 dequant (2026-08-10). */
#if defined(__ARM_NEON)
static inline void sd_dequant_block_q40_neon(const uint8_t *blk,
                                             float32x4_t out[8]) {
    float d = wubu_sd_f16_to_f32(*(const uint16_t *)blk);
    const uint8_t *qs = blk + 2;
    float32x4_t dv = vdupq_n_f32(d);
    uint8x16_t q0 = vld1q_u8(qs);
    uint8x16_t lo = vandq_u8(q0, vdupq_n_u8(0x0F));
    uint8x16_t hi = vandq_u8(vshrq_n_u8(q0, 4), vdupq_n_u8(0x0F));
    /* widen u8 -> u16 -> u32 -> f32, subtract 8, scale by d */
    int16x8_t d0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(lo))), vdupq_n_s16(8));
    int16x8_t d1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(lo))), vdupq_n_s16(8));
    int16x8_t d2 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(hi))), vdupq_n_s16(8));
    int16x8_t d3 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(hi))), vdupq_n_s16(8));
    out[0] = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0))), dv);
    out[1] = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0))), dv);
    out[2] = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1))), dv);
    out[3] = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1))), dv);
    out[4] = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d2))), dv);
    out[5] = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d2))), dv);
    out[6] = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d3))), dv);
    out[7] = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d3))), dv);
}
#endif

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
#elif defined(__ARM_NEON)
    /* ARM NEON quantized GEMM: dequant W ONCE into F32 scratch (NEON
     * dequant, sd_dequant_w), convert x to F32 if F16, then run the
     * NEON 4x4 FMA kernel (wubu_sd_matmul_nt). This is the ARM mirror
     * of the x86 in-register path — same math, verified bit-identical
     * (2026-08-10, txt2img max diff=1). The scratch is per-call and
     * freed immediately. */
    {
        float *wf = sd_dequant_w(W, wtype, K, N);
        if (wf) {
            const float *xf = (const float *)x;
            float *xf_buf = NULL;
            if (xf16) {
                xf_buf = (float *)malloc((size_t)M * K * sizeof(float));
                if (xf_buf) {
                    const uint16_t *xh = (const uint16_t *)x;
                    int mk = M * K;
                    for (int i = 0; i + 8 <= mk; i += 8) {
                        float16x4_t h0 = vreinterpret_f16_u16(vld1_u16(xh + i));
                        float16x4_t h1 = vreinterpret_f16_u16(vld1_u16(xh + i + 4));
                        vst1q_f32(xf_buf + i,     vcvt_f32_f16(h0));
                        vst1q_f32(xf_buf + i + 4, vcvt_f32_f16(h1));
                    }
                    for (int i = mk - (mk % 8); i < mk; i++)
                        xf_buf[i] = wubu_sd_f16_to_f32(xh[i]);
                    xf = xf_buf;
                }
            }
            if (xf) wubu_sd_matmul_nt(xf, wf, M, K, N, y);
            free(xf_buf);
            free(wf);
        }
        if (!wf) {
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    y[(size_t)i * N + j] = sd_dot_col((const char *)x + (size_t)i * K * esz,
                                                      xf16, K, W, wtype, j, nblk);
        }
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

/* ================= Winograd F(2,3) 3x3 conv (experimental) =============
 * 2x2 output tile from a 4x4 input tile: 16 mults vs 36 direct (2.25x
 * fewer MACs). Transforms (Lavin F(2,3), exact F32 adds — the parity
 * RISK is accumulation-order change vs direct im2col GEMM):
 *   U = B^T d B     (input tile d 4x4 -> U 4x4, 64 adds)
 *   V = G g G^T     (filter g 3x3 -> V 4x4, precomputed per conv)
 *   M = sum_ci U ⊙ V   (16 MACs per (co,ci,tile))
 *   Y = A^T M A     (4x4 -> 2x2 output, 24 adds)
 * B^T = [1 0 -1 0; 0 1 1 0; 0 -1 1 0; 0 1 0 -1]
 * G    = [1 0 0; .5 .5 .5; .5 -.5 .5; 0 0 1]
 * A^T  = [1 1 1 0; 0 1 -1 -1]
 * Blocking (A72 1MB shared L2): serial co-chunk OUTER (V-chunk
 * L2-resident, shared by all threads — the j-chunk lesson), tiles
 * PARALLEL inner (U per tile stays L1). Enabled via SD_WINOGRAD=1.
 * Handles us>0 (fused nearest upsample: bounds on upscaled grid,
 * read source at coord>>us_sh). */
#define WINO_BT { { 1, 0, -1, 0 }, { 0, 1, 1, 0 }, { 0, -1, 1, 0 }, { 0, 1, 0, -1 } }
#define WINO_AT { { 1, 1, 1, 0 }, { 0, 1, -1, -1 } }

static void wino_filter_transform(const float g[3][3], float V[4][4]) {
    /* V = G g G^T; G rows */
    const float G[4][3] = { {1,0,0}, {.5f,.5f,.5f}, {.5f,-.5f,.5f}, {0,0,1} };
    float Gg[4][3];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++) {
            float s = 0;
            for (int k = 0; k < 3; k++) s += G[i][k] * g[k][j];
            Gg[i][j] = s;
        }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float s = 0;
            for (int k = 0; k < 3; k++) s += Gg[i][k] * G[j][k];
            V[i][j] = s;
        }
}

/* Y = A^T M A for one 4x4 M -> 2x2 Y */
static inline void wino_output(const float M[4][4], float y[2][2]) {
    float P[4][2];
    for (int l = 0; l < 4; l++) {
        P[l][0] = M[l][0] + M[l][1] + M[l][2];
        P[l][1] = M[l][1] - M[l][2] - M[l][3];
    }
    y[0][0] = P[0][0] + P[1][0] + P[2][0];
    y[0][1] = P[0][1] + P[1][1] + P[2][1];
    y[1][0] = P[1][0] - P[2][0] - P[3][0];
    y[1][1] = P[1][1] - P[2][1] - P[3][1];
}

/* U = B^T d B for one 4x4 input tile */
static inline void wino_input(const float d[4][4], float U[4][4]) {
    float T[4][4];
    for (int j = 0; j < 4; j++) {
        T[0][j] = d[0][j] - d[2][j];
        T[1][j] = d[1][j] + d[2][j];
        T[2][j] = d[2][j] - d[1][j];
        T[3][j] = d[1][j] - d[3][j];
    }
    for (int i = 0; i < 4; i++) {
        U[i][0] = T[i][0] - T[i][2];
        U[i][1] = T[i][1] + T[i][2];
        U[i][2] = T[i][2] - T[i][1];
        U[i][3] = T[i][1] - T[i][3];
    }
}

/* 3x3 stride-1 conv (with optional fused us upsample) via Winograd.
 * x [N][C_in][H][W]; y [N][C_out][H_out][W_out]. Same call shape as
 * wubu_sd_conv2d_q for KH=KW=3, stride=1. */
static void wubu_sd_conv2d_q_wino(const float *x, int N, int C_in, int H, int W,
                                  const void *w, int wtype, const float *b,
                                  int C_out, int pad_h, int pad_w,
                                  float *y, int us) {
    const int Hs = (us > 0) ? H * us : H, Ws = (us > 0) ? W * us : W;
    const int H_out_ = Hs, W_out_ = Ws;   /* 3x3 s1 p1: out == in */
    const int us_sh = (us == 2) ? 1 : (us == 4) ? 2 : 0;
    const int KH = 3, KW = 3;
    /* dequant weights -> w_f[co][k] F32 ([N][K] row-major) */
    const int K = C_in * KH * KW;
    float *w_f = sd_dequant_w(w, wtype, K, C_out);
    if (!w_f) return;
    /* precompute V[co][ci][4][4] */
    float *V = (float *)malloc((size_t)C_out * C_in * 16 * sizeof(float));
    if (!V) { free(w_f); return; }
    for (int co = 0; co < C_out; co++)
        for (int ci = 0; ci < C_in; ci++) {
            float g[3][3];
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    g[kh][kw] = w_f[co * K + ci * 9 + kw * 3 + kh];
            float Vv[4][4];
            wino_filter_transform(g, Vv);
            memcpy(V + (co * C_in + ci) * 16, Vv, 64);
        }
    free(w_f);
    /* tiles: output 2x2 blocks */
    const int TH = H_out_ / 2, TW = W_out_ / 2;
    /* co-chunk: V slice 32co x C_in x 16 x 4B <= 512KB for C_in<=256 */
    int COB = 32;
    if ((size_t)COB * C_in * 16 * 4 > (512u << 10)) COB = 16;
    if ((size_t)COB * C_in * 16 * 4 > (512u << 10)) COB = 8;
    for (int n = 0; n < N; n++) {
        const float *xn = x + (size_t)n * C_in * H * W;
        float *yn = y + (size_t)n * C_out * H_out_ * W_out_;
        for (int cob = 0; cob < C_out; cob += COB) {
            int coe = (cob + COB < C_out) ? cob + COB : C_out;
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < TH * TW; t++) {
                int oh0 = (t / TW) * 2, ow0 = (t % TW) * 2;
                float U[C_in][4][4];   /* 16KB @ C_in=256 — L1 */
                for (int ci = 0; ci < C_in; ci++) {
                    float d[4][4];
                    const float *xci = xn + (size_t)ci * H * W;
                    for (int r = 0; r < 4; r++)
                        for (int c = 0; c < 4; c++) {
                            int ih = oh0 - pad_h + r, iw = ow0 - pad_w + c;
                            if (ih < 0 || ih >= Hs || iw < 0 || iw >= Ws)
                                d[r][c] = 0.0f;
                            else
                                d[r][c] = xci[(size_t)(ih >> us_sh) * W + (iw >> us_sh)];
                        }
                    wino_input(d, U[ci]);
                }
                for (int co = cob; co < coe; co++) {
                    float M[4][4] = {{0}};
                    const float *Vc = V + (size_t)co * C_in * 16;
                    for (int ci = 0; ci < C_in; ci++) {
                        const float *Vv = Vc + (size_t)ci * 16;
                        const float (*Uv)[4] = U[ci];
                        for (int r = 0; r < 4; r++)
                            for (int c = 0; c < 4; c++)
                                M[r][c] += Uv[r][c] * Vv[r * 4 + c];
                    }
                    float y2[2][2];
                    wino_output(M, y2);
                    float bias = b ? b[co] : 0.0f;
                    float *yrow = yn + (size_t)co * H_out_ * W_out_;
                    yrow[(size_t)oh0 * W_out_ + ow0]     = y2[0][0] + bias;
                    yrow[(size_t)oh0 * W_out_ + ow0 + 1] = y2[0][1] + bias;
                    yrow[(size_t)(oh0 + 1) * W_out_ + ow0]     = y2[1][0] + bias;
                    yrow[(size_t)(oh0 + 1) * W_out_ + ow0 + 1] = y2[1][1] + bias;
                }
            }
        }
    }
    free(V);
}

/* ============ Decoupled pixel-shuffle upsample (experimental) ===========
 * us=2, 3x3 s1 p1 conv on the upscaled grid, decomposed into 4 PHASES:
 * output (2i+di, 2j+dj) for phase (di,dj) reads only the 2x2 source
 * window {i-1+di, i+di} x {j-1+dj, j+dj} — 4 taps instead of 9, with
 * PHASE-SPECIFIC pre-summed weights (Turbo-VAED's "decoupled pixel
 * shuffle": conv on the SMALL grid, not the upsampled one).
 *   MACs: 4 phases x 4 taps = 16 per 2x2 block vs 36 direct = 2.25x fewer.
 *   xcol: 4*C_in per pixel vs 9*C_in = 2.25x smaller build AND GEMM traffic.
 * Parity RISK (like Winograd): weight pre-summing changes the FMA
 * accumulation order — sum-of-products vs product-of-sums. Test against
 * max diff=1. Enabled via SD_PIXELSHUFFLE=1.
 * kh/kw -> (dr,dc) mapping per phase (derived from (2i+di+kh-1)>>1):
 *   di=0: kh {0}->dr0, {1,2}->dr1      di=1: kh {0,1}->dr0, {2}->dr1
 *   dj=0: kw {0}->dc0, {1,2}->dc1      dj=1: kw {0,1}->dc0, {2}->dc1 */
static void wubu_sd_conv2d_q_pixshuf(const float *x, int N, int C_in, int H, int W,
                                     const void *w, int wtype, const float *b,
                                     int C_out, float *y) {
    const int H_out_ = 2 * H, W_out_ = 2 * W;
    const int W_half = W, H_half = H;            /* phase grid = source grid */
    const int Kp = C_in * 4;                     /* 2x2 taps per phase */
    float *w_f32 = sd_dequant_w(w, wtype, C_in * 9, C_out);
    if (!w_f32) return;
    /* phase weights W'[phase][co][ci*4 + dr*2 + dc] */
    float *Wp = (float *)malloc((size_t)4 * C_out * Kp * sizeof(float));
    if (!Wp) { free(w_f32); return; }
    for (int co = 0; co < C_out; co++)
        for (int ci = 0; ci < C_in; ci++) {
            const float *W = w_f32 + (size_t)co * C_in * 9 + ci * 9;
            /* W[kh*3+kw] */
            float *w00 = Wp + (0 * C_out + co) * Kp + ci * 4;
            float *w01 = Wp + (1 * C_out + co) * Kp + ci * 4;
            float *w10 = Wp + (2 * C_out + co) * Kp + ci * 4;
            float *w11 = Wp + (3 * C_out + co) * Kp + ci * 4;
            w00[0] = W[0];             w00[1] = W[1] + W[2];
            w00[2] = W[3] + W[6];      w00[3] = W[4] + W[5] + W[7] + W[8];
            w01[0] = W[0] + W[1];      w01[1] = W[2];
            w01[2] = W[3] + W[4] + W[6] + W[7];  w01[3] = W[5] + W[8];
            w10[0] = W[0] + W[3];      w10[1] = W[1] + W[2] + W[4] + W[5];
            w10[2] = W[6];             w10[3] = W[7] + W[8];
            w11[0] = W[0] + W[1] + W[3] + W[4];  w11[1] = W[2] + W[5];
            w11[2] = W[6] + W[7];      w11[3] = W[8];
        }
    free(w_f32);
    /* tiles over phase rows (each tile = T output rows = T/2 source rows) */
    int T = 8;
    if ((size_t)(T / 2 + 1) * W * Kp * 2 > (12u << 20)) T = 4;
    uint16_t *xcol = NULL;
    float *yt = NULL;
    size_t M_tile = ((size_t)T / 2 + 1) * W;     /* phase pixels per tile */
    if (posix_memalign((void **)&xcol, 64, M_tile * Kp * sizeof(uint16_t)) != 0) xcol = NULL;
    if (posix_memalign((void **)&yt, 64, 4 * M_tile * C_out * sizeof(float)) != 0) yt = NULL;
    if (!xcol || !yt) { free(xcol); free(yt); free(Wp); return; }
    for (int n = 0; n < N; n++) {
        const float *xn = x + (size_t)n * C_in * H * W;
        float *yn = y + (size_t)n * C_out * H_out_ * W_out_;
        for (int oh0 = 0; oh0 < H_out_; oh0 += T) {
            int T_eff = (oh0 + T < H_out_) ? T : (H_out_ - oh0);
            int Ni = (T_eff + 1) / 2;            /* phase rows in this tile */
            int i0 = oh0 / 2;
            for (int ph = 0; ph < 4; ph++) {
                int di = ph >> 1, dj = ph & 1;
                int Mp = Ni * W;
                /* im2col for this phase: xcol[p][ci*4+dr*2+dc] */
                #pragma omp parallel for collapse(2) schedule(static)
                for (int ii = 0; ii < Ni; ii++)
                    for (int j = 0; j < W; j++) {
                        uint16_t *xm = xcol + ((size_t)ii * W + j) * Kp;
                        int srow = i0 + ii - 1 + di;   /* src base row */
                        int scol = j - 1 + dj;         /* src base col */
                        for (int ci = 0; ci < C_in; ci++) {
                            const float *xci = xn + (size_t)ci * H * W;
                            uint16_t *xk = xm + ci * 4;
                            for (int dr = 0; dr < 2; dr++) {
                                int r = srow + dr;
                                for (int dc = 0; dc < 2; dc++) {
                                    int c = scol + dc;
                                    float v = (r >= 0 && r < H && c >= 0 && c < W)
                                            ? xci[(size_t)r * W + c] : 0.0f;
                                    xk[dr * 2 + dc] = wubu_sd_f32_to_f16(v);
                                }
                            }
                        }
                    }
#if defined(__ARM_NEON) && defined(__aarch64__)
                wubu_sd_matmul_nt_f16(xcol, Wp + (size_t)ph * C_out * Kp,
                                      Mp, Kp, C_out, yt + (size_t)ph * M_tile * C_out);
#else
                {
                    /* x86: convert F16 xcol -> F32 scratch, F32 matmul */
                    size_t mk = (size_t)Mp * Kp;
                    float *xf_buf = (float *)malloc(mk * sizeof(float));
                    if (!xf_buf) { free(xcol); free(yt); free(Wp); return; }
                    for (size_t k = 0; k < mk; k++)
                        xf_buf[k] = wubu_sd_f16_to_f32(xcol[k]);
                    wubu_sd_matmul_nt(xf_buf, Wp + (size_t)ph * C_out * Kp,
                                      Mp, Kp, C_out, yt + (size_t)ph * M_tile * C_out);
                    free(xf_buf);
                }
#endif
            }
            /* merge writeback: y[co][(2i+di)*W_out + 2j+dj] */
            #pragma omp parallel for schedule(static)
            for (int co = 0; co < C_out; co++) {
                float bias = b ? b[co] : 0.0f;
                float *yrow = yn + (size_t)co * H_out_ * W_out_;
                for (int ii = 0; ii < Ni; ii++) {
                    int oh_e = (i0 + ii) * 2;          /* even output row */
                    int oh_o = oh_e + 1;
                    float *re = yrow + (size_t)oh_e * W_out_;
                    float *ro = (oh_o < H_out_) ? yrow + (size_t)oh_o * W_out_ : NULL;
                    for (int j = 0; j < W; j++) {
                        int p = ii * W + j;
                        const float *y00 = yt + (0 * M_tile + p) * C_out + co;
                        const float *y01 = yt + (1 * M_tile + p) * C_out + co;
                        const float *y10 = yt + (2 * M_tile + p) * C_out + co;
                        const float *y11 = yt + (3 * M_tile + p) * C_out + co;
                        re[2 * j]     = *y00 + bias;
                        re[2 * j + 1] = *y01 + bias;
                        if (ro) {
                            ro[2 * j]     = *y10 + bias;
                            ro[2 * j + 1] = *y11 + bias;
                        }
                    }
                }
            }
        }
    }
    free(xcol); free(yt); free(Wp);
}

void wubu_sd_conv2d_q(const float *x, int N, int C_in, int H, int W,
                      const void *w, int wtype, const float *b,
                      int C_out, int KH, int KW, int stride,
                      int pad_h, int pad_w,
                      float *y, int *H_out, int *W_out,
                      int us) {
    /* EXPERIMENTAL Winograd F(2,3) path (SD_WINOGRAD=1): 2.25x fewer
     * MACs on 3x3 stride-1 convs, but the accumulation order differs
     * from direct im2col GEMM — parity risk (may break max diff=1). */
    static int wino_env = -1;
    if (wino_env < 0) wino_env = getenv("SD_WINOGRAD") != NULL;
    if (wino_env && KH == 3 && KW == 3 && stride == 1) {
        wubu_sd_conv2d_q_wino(x, N, C_in, H, W, w, wtype, b, C_out,
                              pad_h, pad_w, y, us);
        if (H_out) *H_out = (us > 0 ? H * us : H);
        if (W_out) *W_out = (us > 0 ? W * us : W);
        return;
    }
    /* EXPERIMENTAL decoupled pixel-shuffle upsample (SD_PIXELSHUFFLE=1):
     * us=2 3x3 s1 p1 only. 2.25x fewer MACs on the upsample convs (the
     * biggest VAE cost), phase-specific pre-summed weights. Parity risk
     * like Winograd — test against max diff=1 before enabling by default. */
    static int ps_env = -1;
    if (ps_env < 0) ps_env = getenv("SD_PIXELSHUFFLE") != NULL;
    if (ps_env && us == 2 && KH == 3 && KW == 3 && stride == 1 &&
        pad_h == 1 && pad_w == 1) {
        wubu_sd_conv2d_q_pixshuf(x, N, C_in, H, W, w, wtype, b, C_out, y);
        if (H_out) *H_out = 2 * H;
        if (W_out) *W_out = 2 * W;
        return;
    }
    /* us = nearest upsample factor applied BEFORE the conv, fused into
     * im2col: the conv logically runs on a (H*us)x(W*us) input where
     * every us x us block repeats one source pixel (x[i][j]). We never
     * materialize that buffer — the im2col bounds-checks on the
     * upscaled grid and reads the SOURCE at (coord/us). For the VAE
     * upsample convs (up.1 132.6s + up.2 125.6s = 258s of the ~580s
     * VAE) this kills the 4x-buffer write+read AND each 2x2 output
     * block reuses one source value for 4 (or 9 with 3x3 taps)
     * outputs — the source stays L2/L1-hot instead of streaming a
     * 268MB upsampled copy. */
    const int Hs = (us > 0) ? H * us : H, Ws = (us > 0) ? W * us : W;
    const int H_out_ = (Hs + 2 * pad_h - KH) / stride + 1;
    const int W_out_ = (Ws + 2 * pad_w - KW) / stride + 1;
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
    /* ARM: dequant the conv weight ONCE for all tiles (F16/Q4_0 -> F32
     * NEON dequant), then the GEMM is pure NEON 4x4 FMA. Avoids the
     * per-tile dequant that wubu_sd_matmul_q would do (H_out/T redundant
     * passes over the same weights). x86 keeps the RAW blob and dequants
     * in registers per tile (its AVX2 path is cache-efficient). */
    float *w_f32 = NULL;
#if defined(__ARM_NEON) && !defined(__x86_64__)
    if (wtype != 0) {
        w_f32 = sd_dequant_w(w, wtype, K, C_out);
    }
#endif
    /* xcol is [T*W_out][K] ROW-major in F16 — matmul_q(xf16=1) expects
     * x[M][K] with x[m*K + k]; a [K][M] column-major layout would
     * misalign. F16 halves the xcol memory traffic (the big 512x512
     * VAE convs build >1GB of xcol). Weights stay RAW — the GEMM
     * dequants F16/Q4_0 in registers. */
    uint16_t *xcol = NULL;
    float *yt = NULL;
    if (posix_memalign((void **)&xcol, 64, (size_t)T * W_out_ * K * sizeof(uint16_t)) != 0) xcol = NULL;
    if (posix_memalign((void **)&yt, 64, (size_t)C_out * T * W_out_ * sizeof(float)) != 0) yt = NULL;
    if (!xcol || !yt) { free(xcol); free(yt); free(w_f32); return; }
    double ph_build = 0, ph_gemm = 0, ph_wb = 0;
    const int ph_on = getenv("SD_PHASE_TIMING") != NULL;
    for (int n = 0; n < N; n++) {
        const float *xn = x + (size_t)n * C_in * H * W;
        float *yn = y + (size_t)n * C_out * H_out_ * W_out_;
        for (int oh0 = 0; oh0 < H_out_; oh0 += T) {
            int T_eff = (oh0 + T < H_out_) ? T : (H_out_ - oh0);
            double tA = ph_on ? (double)clock() / CLOCKS_PER_SEC : 0;
            /* im2col: xcol[(t*W_out+ow)][k] row-major F16.
             * Parallel over OUTPUT pixels (t,ow); the k-loop is INNER so
             * writes are sequential (2-byte stores fill 64B lines instead
             * of striding K*2 bytes → ~32x write-allocate amplification
             * on the old k-outer layout). For a 512x512 VAE conv this is
             * 38GB of line traffic saved per conv (2026-08-10).
             * With us>0: bounds-check on the UPSCALED grid (Hs x Ws),
             * read the source at /us — fused nearest upsample. */
            #pragma omp parallel for collapse(2) schedule(static)
            for (int t = 0; t < T_eff; t++)
                for (int ow = 0; ow < W_out_; ow++) {
                    uint16_t *xcolm = xcol + ((size_t)t * W_out_ + ow) * K;
                    int ih = (oh0 + t) * stride - pad_h;
                    /* Row fully out of range only when EVERY kh is out
                     * (ih+kh in [0,Hs) for kh in 0..KH-1). Partial rows
                     * (pad edges) fall through to the per-k check below. */
                    if (ih + (KH - 1) < 0 || ih >= Hs) {
                        memset(xcolm, 0, (size_t)K * sizeof(uint16_t));
                        continue;
                    }
                    int iw0 = -pad_w + ow;
                    /* Incremental tap walk: k = ci*KH*KW + kw*KH + kh with
                     * kh innermost. Tracks (ci,kh,kw) with add/compare
                     * instead of 3 integer divisions per element — the
                     * old k/(KH*KW), (k/KH)%KW, k%KH did ~1.8 G divs per
                     * 512x512 conv (runtime divisors = real divs). */
                    int us_sh = (us == 2) ? 1 : (us == 4) ? 2 : 0;
                    int ci = 0, kw = 0, kh = 0;
                    for (int k = 0; k < K; k++) {
                        int ihk = ih + kh;
                        int iw = iw0 + kw;
                        if (ihk < 0 || ihk >= Hs || iw < 0 || iw >= Ws) {
                            xcolm[k] = 0;
                        } else {
                            xcolm[k] = wubu_sd_f32_to_f16(
                                xn[(size_t)ci * H * W +
                                   (size_t)(ihk >> us_sh) * W +
                                   (iw >> us_sh)]);
                        }
                        if (++kh == KH) { kh = 0; if (++kw == KW) { kw = 0; ++ci; } }
                    }
                }
            if (ph_on) ph_build += (double)clock() / CLOCKS_PER_SEC - tA;
            double tB = ph_on ? (double)clock() / CLOCKS_PER_SEC : 0;
            /* yt[co][p] = sum_k W[co][k] * xcol[k][p]:
             * matmul_q(x=xcol^T [T*W_out][K] F16, W=raw [K][C_out])
             * NOTE: output is [M][N] row-major = yt[p*C_out + co] */
            if (w_f32) {
                /* ARM: W already F32 (dequantized once above). Run the
                 * NEON 4x4 FMA kernel directly on the F16 xcol — the
                 * kernel converts F16->F32 in-register, so we skip the
                 * separate F16->F32 scratch pass AND halve x memory
                 * traffic (x rows re-read N/4 times). */
#if defined(__ARM_NEON)
                wubu_sd_matmul_nt_f16(xcol, w_f32, T_eff * W_out_, K, C_out, yt);
#else
                int mk = T_eff * W_out_ * K;
                float *xf_buf = (float *)malloc((size_t)mk * sizeof(float));
                const float *xf = xf_buf ? xf_buf : (const float *)xcol;
                if (xf_buf) {
                    const uint16_t *xh = (const uint16_t *)xcol;
                    for (int k = 0; k < mk; k++) xf_buf[k] = wubu_sd_f16_to_f32(xh[k]);
                }
                wubu_sd_matmul_nt(xf, w_f32, T_eff * W_out_, K, C_out, yt);
                free(xf_buf);
#endif
            } else {
                wubu_sd_matmul_q(xcol, 1, w, wtype, T_eff * W_out_, K, C_out, yt);
            }
            if (ph_on) ph_gemm += (double)clock() / CLOCKS_PER_SEC - tB;
            double tC = ph_on ? (double)clock() / CLOCKS_PER_SEC : 0;
            #pragma omp parallel for schedule(static)
            for (int co = 0; co < C_out; co++) {
                float bias = b ? b[co] : 0.0f;
                float *yrow = yn + (size_t)co * H_out_ * W_out_ + (size_t)oh0 * W_out_;
                for (int p = 0; p < T_eff * W_out_; p++)
                    yrow[p] = yt[(size_t)p * C_out + co] + bias;
            }
            if (ph_on) ph_wb += (double)clock() / CLOCKS_PER_SEC - tC;
        }
    }
    if (ph_on)
        fprintf(stderr, "  [conv-phases] build=%.2fs gemm=%.2fs wb=%.2fs\n",
                ph_build, ph_gemm, ph_wb);
    free(xcol); free(yt);
    free(w_f32);
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
