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
 * SIMD: AVX2 FMA over K when available (W is [N][K] row-major so each
 * output row streams W contiguously — cache-friendly, 8-wide FMA). */
void wubu_sd_matmul_nt(const float *x, const float *W, int M, int K, int N,
                       float *y)
{
#if defined(__AVX2__) && defined(__FMA__)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const float *xr = x + (size_t)i * K;
        float *yr = y + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            const float *wr = W + (size_t)j * K;
            int k = 0;
            __m256 acc = _mm256_setzero_ps();
            for (; k + 8 <= K; k += 8) {
                __m256 xv = _mm256_loadu_ps(xr + k);
                __m256 wv = _mm256_loadu_ps(wr + k);
                acc = _mm256_fmadd_ps(xv, wv, acc);
            }
            float t[8];
            _mm256_storeu_ps(t, acc);
            float s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
            for (; k < K; k++) s += xr[k] * wr[k];
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
 *       value = (nibble - 8) * d, nibble = qs[i/2] >> (4*(i%2)) & 0xF
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
            for (int b = 0; b < nblk; b++) {
                const uint8_t *blk = wj + (size_t)b * 18;
                float d = wubu_sd_f16_to_f32(*(const uint16_t *)blk);
                const uint8_t *qs = blk + 2;
                int base = b * 32;
                int rem = K - base; if (rem > 32) rem = 32;
                for (int l = 0; l < rem; l++) {
                    int8_t q = (int8_t)((qs[l >> 1] >> (4 * (l & 1))) & 0xF) - 8;
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

void wubu_sd_linear_q(const float *x, const void *W, int type,
                      int M, int K, int N, float *y) {
    if (type == 0) { /* GGML_TYPE_F32 (0): plain matmul */
        wubu_sd_matmul_nt(x, (const float *)W, M, K, N, y);
        return;
    }
    /* F16 / Q4_0: dequant the whole weight ONCE, then AVX2 FMA GEMM.
     * This is the wizard's quantized_matmul pattern (dequant to scratch,
     * SIMD GEMM) — the scratch is per-call, never cached. */
    float *f32 = sd_dequant_w(W, type, K, N);
    if (!f32) { fprintf(stderr, "linear_q: dequant alloc failed\n"); return; }
    wubu_sd_matmul_nt(x, f32, M, K, N, y);
    free(f32);
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
    const int T = 8;
    /* Dequant the conv weight ONCE for all tiles (F32 scratch, per-call).
     * The im2col's K-order matches the weight's raw [K][C_out] layout:
     * k = ci*KH*KW + kw*KH + kh (kh innermost). */
    float *wf = sd_dequant_w(w, wtype, K, C_out);
    if (!wf) { fprintf(stderr, "conv2d_q: dequant failed\n"); return; }
    /* xcol is [T*W_out][K] ROW-major — linear_q expects x[M][K] with
     * x[m*K + k]; a [K][M] column-major layout would misalign. */
    float *xcol = (float *)malloc((size_t)T * W_out_ * K * sizeof(float));
    float *yt   = (float *)malloc((size_t)C_out * T * W_out_ * sizeof(float));
    if (!xcol || !yt) { free(xcol); free(yt); free(wf); return; }
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
                    float *xcolm = xcol + ((size_t)t * W_out_) * K + k;
                    if (ih < 0 || ih >= H) {
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
            /* yt[co][p] = sum_k W[co][k] * xcol[k][p]:
             * linear_q(x=xcol^T [T*W_out][K], W=wf F32 [K][C_out])
             * NOTE: linear_q output is [M][N] row-major = yt[p*C_out + co] */
            wubu_sd_linear_q(xcol, wf, 0, T_eff * W_out_, K, C_out, yt);
            #pragma omp parallel for schedule(static)
            for (int co = 0; co < C_out; co++) {
                float bias = b ? b[co] : 0.0f;
                float *yrow = yn + (size_t)co * H_out_ * W_out_ + (size_t)oh0 * W_out_;
                for (int p = 0; p < T_eff * W_out_; p++)
                    yrow[p] = yt[(size_t)p * C_out + co] + bias;
            }
        }
    }
    free(xcol); free(yt); free(wf);
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
