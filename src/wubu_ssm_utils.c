/*
 * wubu_ssm_utils.c — shared SSM/GQA helper functions.
 *
 * Extracted from wubu_ssm.c to keep module boundaries clean.
 * Contains: proj_matmul, matmul_nt, tgt_wrap, tgt_safe_expf,
 *           backward_matmul_nt, backward_conv1d.
 */

#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

#ifndef QK_K
#define QK_K 256
#endif

/* TGT (Toroidal Gradient Transformation) safe wrapping using π odometer */
#define TGT_PI       3.14159265358979323846f
#define TGT_BOUNDARY (2.0f * TGT_PI)

float tgt_wrap(float x) {
    return fmodf(x + TGT_PI, TGT_BOUNDARY) - TGT_PI;
}

float tgt_safe_expf(float x) {
    /* Clamp to avoid float32 overflow: expf(89) ≈ 5e38 ≈ overflow */
    if (x > 80.0f) x = 80.0f;
    if (x < -80.0f) return 0.0f;
    return expf(x);
}

/*
 * Centralized quantized matmul dispatch for SSM/GQA projections.
 * If quantized weight (W_q) is available and type is not F32, uses quantized_matmul.
 * Otherwise falls back to the provided F32 weight with a column loop.
 */
void proj_matmul(const float *x, int64_t n_rows, int64_t n_cols,
                 const float *W_f32, const uint8_t *W_q, int weight_type,
                 float *out) {
    if (W_q && weight_type != GGML_TYPE_F32 && n_cols > 0) {
        quantized_matmul(x, W_q, weight_type, n_rows, n_cols, 0, out);
    } else {
        #pragma omp parallel for if(n_cols > 4)
        for (int64_t j = 0; j < n_cols; j++) {
            double sum = 0.0;
            for (int64_t i = 0; i < n_rows; i++)
                sum += (double)x[i] * (double)W_f32[j * n_rows + i];
            out[j] = (float)sum;
        }
    }
}

/* C[M,N] = A[M,K] @ B[N,K]^T  (B is stored NxK) */
void matmul_nt(int M, int N, int K,
              const float *A, const float *B,
              float *C) {
    long long ops = (long long)M * N * K;
    #pragma omp parallel for if(ops > 500000)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

/*
 * Backward through matmul: output[s,j] = sum_i input[s,i] * W[i,j]
 *   d_input[s,i] += sum_j d_output[s,j] * W[i,j]
 *   dW[i,j] += sum_s input[s,i] * d_output[s,j]
 */
void backward_matmul_nt(int N, int Din, int Dout,
                        const float *input, const float *d_output,
                        const float *W, float *d_input, float *dW) {
    /* d_input = d_output @ W^T */
    for (int s = 0; s < N; s++) {
        for (int i = 0; i < Din; i++) {
            double sum = 0.0;
            for (int j = 0; j < Dout; j++)
                sum += (double)d_output[s * Dout + j] * (double)W[i * Dout + j];
            d_input[s * Din + i] += (float)sum;
        }
    }
    /* dW = input^T @ d_output */
    if (dW) {
        for (int i = 0; i < Din; i++) {
            for (int j = 0; j < Dout; j++) {
                double sum = 0.0;
                for (int s = 0; s < N; s++)
                    sum += (double)input[s * Din + i] * (double)d_output[s * Dout + j];
                dW[i * Dout + j] += (float)sum;
            }
        }
    }
}

/*
 * Backward through Conv1D.
 * Forward: output[t,c] = sum_{ki=0}^{k-1} input[t+ki,c] * kernel[ki,c]
 */
void backward_conv1d(int B, int T, int C, int k,
                     const float *input,   /* [B, T+k-1, C] */
                     const float *d_output, /* [B, T, C] */
                     const float *kernel,   /* [k, C] */
                     float *d_input,        /* [B, T+k-1, C] */
                     float *d_kernel) {     /* [k, C] */
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                float do_val = d_output[(b * T + t) * C + c];
                for (int ki = 0; ki < k; ki++) {
                    int t_in = t + ki;
                    d_input[(b * (T + k - 1) + t_in) * C + c] += do_val * kernel[ki * C + c];
                }
            }
        }
    }
    /* d_kernel */
    for (int ki = 0; ki < k; ki++) {
        for (int c = 0; c < C; c++) {
            double sum = 0.0;
            for (int b = 0; b < B; b++) {
                for (int t = 0; t < T; t++) {
                    int t_in = t + ki;
                    sum += (double)d_output[(b * T + t) * C + c]
                         * (double)input[(b * (T + k - 1) + t_in) * C + c];
                }
            }
            if (d_kernel) d_kernel[ki * C + c] += (float)sum;
        }
    }
}
