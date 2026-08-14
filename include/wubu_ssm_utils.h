#ifndef WUBU_SSM_UTILS_H
#define WUBU_SSM_UTILS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* TGT safe exp (clamped to avoid float32 overflow) */
float tgt_safe_expf(float x);

/* TGT wrap: keep values in [-π, π] */
float tgt_wrap(float x);

/* Quantized matmul dispatch (used by SSM and GQA projections) */
void proj_matmul(const float *x, int64_t n_rows, int64_t n_cols,
                 const float *W_f32, const uint8_t *W_q, int weight_type,
                 float *out);

/* C[M,N] = A[M,K] @ B[N,K]^T */
void matmul_nt(int M, int N, int K,
              const float *A, const float *B,
              float *C);

/* Backward through matmul: output[s,j] = sum_i input[s,i] * W[i,j] */
void backward_matmul_nt(int N, int Din, int Dout,
                        const float *input, const float *d_output,
                        const float *W, float *d_input, float *dW);

/* Backward through Conv1D */
void backward_conv1d(int B, int T, int C, int k,
                     const float *input,
                     const float *d_output,
                     const float *kernel,
                     float *d_input,
                     float *d_kernel);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SSM_UTILS_H */