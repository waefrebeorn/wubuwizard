/* wubu_sd_ops.h -- 2D convolution / groupnorm / sample ops for the
 * WuBu stable-diffusion engine (SD 1.5 LDM family). C11, opaque-free
 * (plain functions), self-contained.
 */
#ifndef WUBU_SD_OPS_H
#define WUBU_SD_OPS_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* float16 -> float32 (standard bit reinterpretation; self-contained so
 * the SD modules don't depend on the GGUF reader's static helper). */
static inline float wubu_sd_f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) {
        if (mant == 0) return (sign ? -0.0f : 0.0f);
        float v = (float)mant * 5.960464477539063e-8f;
        return sign ? -v : v;
    }
    if (exp == 31) {
        uint32_t bits = (sign << 31) | 0x7F800000u | (mant << 13);
        float v; memcpy(&v, &bits, 4);
        return v;
    }
    uint32_t bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float v; memcpy(&v, &bits, 4);
    return v;
}

/* float32 -> float16 (round-to-nearest, same bit-exact behavior as the
 * original conv2d xcol converter — DO NOT "improve" the rounding; the
 * F16 xcol feeds the GEMM and any ULP change risks the max diff=1 bar). */
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

/* y[M,N] = x[M,K] @ W^T[K,N] (nn.Linear convention, W row-major [N,K]).
 * OpenMP-parallel over M; the workhorse linear for CLIP/UNet. */
void wubu_sd_matmul_nt(const float *x, const float *W, int M, int K, int N,
                       float *y);
/* F16-x variant: x is uint16_t halves, W is F32 (pre-dequantized).
 * NEON-only; conv2d_q xcol path reads half the x bytes. */
#if defined(__ARM_NEON)
void wubu_sd_matmul_nt_f16(const uint16_t *x, const float *W, int M, int K,
                           int N, float *y);
#endif
void wubu_sd_matmul_q(const void *x, int xf16, const void *W, int wtype,
                      int M, int K, int N, float *y);

/* y[n][c][h][w] = sum over k,kh,kw of x[n][k][h*stride+kh][w*stride+kw]
 *                  * w[c][k][kh][kw] + b[c]
 * x:      [N][C_in][H][W]
 * w:      [C_out][C_in][KH][KW]
 * y:      [N][C_out][H_out][W_out], H_out=(H-KH)/stride+1
 * All row-major (innermost dim = W). w is F32 weights.
 */
void wubu_sd_conv2d(const float *x, int N, int C_in, int H, int W,
                    const float *w, const float *b,
                    int C_out, int KH, int KW, int stride,
                    int pad_h, int pad_w,
                    float *y, int *H_out, int *W_out);

/* y[n][c][h][w] = (x[n][c][h][w] - mean_c) / sqrt(var_c + eps) * g[c] + b[c]
 * groupnorm: channels split into G groups, each normalized over
 * (C/G)*H*W per sample. */
void wubu_sd_groupnorm(const float *x, int N, int C, int H, int W,
                       int G, float eps,
                       const float *g, const float *b, float *y);

/* nearest-neighbor upsample 2x: [N][C][H][W] -> [N][C][2H][2W] */
void wubu_sd_upsample2x(const float *x, int N, int C, int H, int W, float *y);

/* elementwise SiLU (swish) in place */
void wubu_sd_silu(float *x, int n);

/* ---- quantized (never-dequantize-to-F32) linear layer --------------------
 * y[M][N] = x[M][K] @ W^T[K][N], where W is the RAW mmap'd GGUF weight
 * (F16 or legacy Q4_0). Weights are read directly from the blob and
 * dequantized per-block INSIDE the inner loops (Roofline: memory-bound).
 *
 *   x:     [M][K] F32 activations
 *   W:     raw weight bytes (blob + tensor->data_offset), column-major
 *          like ggml: element (k, n) at W[(n*K + k) * elem_size] (F16),
 *          or Q4_0 blocks per column.
 *   type:  GGML_TYPE_F16 or GGML_TYPE_Q4_0 (as in gguf_reader.h)
 *   K,N:   contraction / output dims
 *   y:     [M][N] output
 */
void wubu_sd_linear_q(const float *x, const void *W, int type,
                      int M, int K, int N, float *y);

/* same, but bias add: y = xW^T + b */
void wubu_sd_linear_qb(const float *x, const void *W, int type,
                       const float *b, int M, int K, int N, float *y);

/* ---- quantized conv2d (weights stay in the mmap'd blob) -------------------
 * Same math as wubu_sd_conv2d but W is the RAW GGUF weight (F16/Q4_0),
 * stored in ggml 4D layout: dims = (kh, kw, cin, cout) with kh innermost,
 * i.e. element (kh,kw,ci,co) at offset ((kh*KW + kw)*C_in + ci) + co*K.
 * im2col is built in that k-order so the GEMM aligns. */
void wubu_sd_conv2d_q(const float *x, int N, int C_in, int H, int W,
                      const void *w, int wtype, const float *b,
                      int C_out, int KH, int KW, int stride,
                      int pad_h, int pad_w,
                      float *y, int *H_out, int *W_out,
                      int us);

/* 2x2 maxpool stride 2: [N][C][H][W] -> [N][C][H/2][W/2] */
void wubu_sd_downsample2x(const float *x, int N, int C, int H, int W, float *y);

#endif /* WUBU_SD_OPS_H */
