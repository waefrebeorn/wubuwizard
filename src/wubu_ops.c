/*
 * wubu_ops.c — Low-level numerical operators (activations, norm, conv).
 *
 * Extracted from wubu_ssm.c (Strangler Fig, research 066-A1/E2).
 * These pure numerical primitives are used by SSM, GQA, MoE, and backward
 * passes. They live in their own compilation unit so that changing one
 * activation does not force recompilation of the 3263-line wubu_ssm.c.
 *
 * See wubu_ops.h for the interface.
 */
#include "wubu_ops.h"
#include "wubu_ssm.h"   /* for g_ssm_l2_eps, SSM_D_STATE, etc. */
#include "wubu_dims.h"   /* for VALUE_DIM */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>  // AVX2/FMA intrinsics (l2_norm, rms_norm, conv1d)

/* ---- Global shared with wubu_ssm.c ---- */
float g_ssm_l2_eps = 1e-6f;
/* g_tensor_naming is defined in wubu_ops.c — it governs tensor naming
 * convention, read/written by wubu_model.c (set) and used by
 * wubu_moe.c + wubu_is_ssm_layer(). Moved here with wubu_is_ssm_layer
 * during the Strangler Fig extraction (ADR-002). */
int g_tensor_naming = 0;

/* ---- Utility ---- */
int wubu_is_ssm_layer(int layer_idx) {
    extern int g_tensor_naming;
    if (g_tensor_naming == 2) return 0;  /* pure GQA flag */
    return (layer_idx + 1) % 4 != 0;
}

/* ============================================================
 * Activations (elementwise)
 * ============================================================ */
void wubu_softplus(int n, const float *x, float *out) {
    #pragma omp parallel for if(n > 100000)
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v > 80.0f) out[i] = v;          // linear region
        else if (v < -80.0f) out[i] = 0.0f; // zero region
        else out[i] = logf(1.0f + expf(v));
    }
}

void wubu_silu(int n, const float *x, float *out) {
    #pragma omp parallel for if(n > 100000)
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v < -80.0f) out[i] = 0.0f;
        else out[i] = v / (1.0f + expf(-v));
    }
}

void wubu_sigmoid(int n, const float *x, float *out) {
    #pragma omp parallel for if(n > 100000)
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v < -80.0f) out[i] = 0.0f;
        else if (v > 80.0f) out[i] = 1.0f;
        else out[i] = 1.0f / (1.0f + expf(-v));
    }
}

void wubu_silu_backward(int n, const float *x, const float *y,
                        const float *dy, float *dx) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float sig = 1.0f / (1.0f + expf(-v));
        float silu = y[i];
        float silu_grad = silu + sig * (1.0f - silu);
        dx[i] += dy[i] * silu_grad;
    }
}

/* ============================================================
 * Normalization
 * ============================================================ */
void wubu_l2_norm(int B, int T, int n_heads, int d,
                  const float *x, float eps, float *out) {
    // x: [B, T, n_heads, d]
    // out: [B, T, n_heads, d]
    int seq_len = B * T;
    #pragma omp parallel for collapse(2) if(seq_len * n_heads > 100)
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_heads; h++) {
            const float *inp = x + (s * n_heads + h) * d;
            float *oup = out + (s * n_heads + h) * d;
            float sum_sq = 0.0f;
#ifdef __AVX2__
            __m256 acc = _mm256_setzero_ps();
            int i;
            for (i = 0; i <= d - 8; i += 8) {
                __m256 v = _mm256_loadu_ps(inp + i);
                acc = _mm256_fmadd_ps(v, v, acc);
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            sum_sq = _mm_cvtss_f32(lo);
            for (; i < d; i++) sum_sq += inp[i] * inp[i];
#else
            for (int i = 0; i < d; i++) sum_sq += inp[i] * inp[i];
#endif
            float scale = 1.0f / sqrtf(sum_sq + eps);
#ifdef __AVX2__
            __m256 v_scale = _mm256_set1_ps(scale);
            for (int i = 0; i <= d - 8; i += 8)
                _mm256_storeu_ps(oup + i, _mm256_mul_ps(_mm256_loadu_ps(inp + i), v_scale));
            for (int i = (d / 8) * 8; i < d; i++) oup[i] = inp[i] * scale;
#else
            for (int i = 0; i < d; i++) oup[i] = inp[i] * scale;
#endif
        }
    }
}

void wubu_rms_norm(int B, int T, int d,
                   const float *x, const float *weight,
                   float eps, float *out) {
    // x: [B, T, d]
    // weight: [d]
    // out: [B, T, d]
    int seq_len = B * T;
    #pragma omp parallel for if(seq_len > 10)
    for (int s = 0; s < seq_len; s++) {
        const float *inp = x + s * d;
        float *oup = out + s * d;
        float sum_sq = 0.0f;
#ifdef __AVX2__
        __m256 acc = _mm256_setzero_ps();
        int i;
        for (i = 0; i <= d - 8; i += 8) {
            __m256 v = _mm256_loadu_ps(inp + i);
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sum_sq = _mm_cvtss_f32(lo);
        for (; i < d; i++) sum_sq += inp[i] * inp[i];
#else
        for (int i = 0; i < d; i++) sum_sq += inp[i] * inp[i];
#endif
        float rms = sqrtf(sum_sq / d + eps);
        float scale = 1.0f / rms;
#ifdef __AVX2__
        __m256 v_scale = _mm256_set1_ps(scale);
        for (int i = 0; i <= d - 8; i += 8)
            _mm256_storeu_ps(oup + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(inp + i), v_scale), _mm256_loadu_ps(weight + i)));
        for (int i = (d / 8) * 8; i < d; i++) oup[i] = inp[i] * scale * weight[i];
#else
        for (int i = 0; i < d; i++) oup[i] = inp[i] * scale * weight[i];
#endif
    }
}

void wubu_l2_norm_backward(int B, int T, int n_heads, int d,
                           const float *x, float eps,
                           const float *d_out, float *d_x) {
    const int N = B * T;
    for (int s = 0; s < N; s++) {
        for (int h = 0; h < n_heads; h++) {
            const float *inp = x + (s * n_heads + h) * d;
            const float *do_h = d_out + (s * n_heads + h) * d;
            float *dx = d_x + (s * n_heads + h) * d;
            
            double sum_sq = 0.0;
            for (int i = 0; i < d; i++) sum_sq += (double)inp[i] * (double)inp[i];
            float norm = sqrtf(sum_sq + eps);
            float n3 = norm * norm * norm;
            
            // d_i = (do_i / norm) - (x_i / n³) * sum_j (do_j * x_j)
            double dot = 0.0;
            for (int j = 0; j < d; j++) dot += (double)do_h[j] * (double)inp[j];
            
            for (int i = 0; i < d; i++) {
                dx[i] += (float)((double)do_h[i] / norm - (double)inp[i] * dot / n3);
            }
        }
    }
}

void wubu_rms_norm_backward(int B, int T, int d,
                            const float *x, const float *weight, float eps,
                            const float *d_out, float *d_x) {
    const int N = B * T;
    for (int s = 0; s < N; s++) {
        const float *inp = x + s * d;
        const float *do_h = d_out + s * d;
        float *dx = d_x + s * d;
        double sum_sq = 0.0;
        for (int i = 0; i < d; i++) sum_sq += (double)inp[i] * (double)inp[i];
        float rms = sqrtf((float)(sum_sq / d) + eps);
        float r = 1.0f / rms;
        float r3 = r * r * r;
        double inner = 0.0;
        for (int j = 0; j < d; j++)
            inner += (double)do_h[j] * (double)weight[j] * (double)inp[j];
        for (int i = 0; i < d; i++)
            dx[i] += do_h[i] * weight[i] * r - (r3 / d) * inp[i] * (float)inner;
    }
}


/* ============================================================
 * 1D Convolution (depthwise, causal)
 * ============================================================ */
void wubu_conv1d(int B, int T, int C, int k,
                 const float *input, const float *kernel,
                 float *output) {
    if (B <= 0 || T <= 0 || C <= 0 || k <= 0) return;
    // input: [B, T+k-1, C] — already padded with k-1 zeros at start
    // kernel: [k, C]
    // output: [B, T, C]
    #pragma omp parallel for collapse(2) if(B * T * C * k > 100000)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // AVX2 conv1d disabled: kernel[ki + c*k] is per-channel.
            // Broadcasting one channel's kernel to all 8 vector channels
            // produces wrong output. Sequential path always used.
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int ki = 0; ki < k; ki++) {
                    int t_in = t + ki;
                    sum += input[(b * (T + k - 1) + t_in) * C + c] *
                           kernel[ki + c * k];
                }
                output[(b * T + t) * C + c] = sum;
            }
        }
    }
}
