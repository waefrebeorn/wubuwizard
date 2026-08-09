#include <stdint.h>
#include <stddef.h>
#include "gguf_reader.h"
#include <immintrin.h>
#ifndef QK_K
#define QK_K 256
#endif
// weight_type: GGML_TYPE for W
// n_rows, n_cols: dimensions
// col_stride_bytes: byte stride between columns (0 = packed)
// y: [n_cols] F32 output
// ========================================================================
void quantized_matmul_from_q8(const void *q8_x,
                              const void *W, int weight_type,
                              int64_t n_rows, int64_t n_cols,
                              int64_t col_stride_bytes,
                              float *y) {
    // Handle IQ1_M and other rare types without vec_dot: dequant then SGEMM
    if (weight_type == GGML_TYPE_IQ1_M || weight_type == GGML_TYPE_IQ1_S ||
        weight_type == GGML_TYPE_IQ2_S || weight_type == GGML_TYPE_IQ2_XS ||
        weight_type == GGML_TYPE_IQ3_S ||
        weight_type == GGML_TYPE_Q2_K || weight_type == GGML_TYPE_Q3_K ||
        weight_type == GGML_TYPE_Q8_0) {
        int64_t total_elems = n_rows * n_cols;
        float *f32_w = (float *)malloc(total_elems * sizeof(float));
        if (!f32_w) { fprintf(stderr, "quantized_matmul_from_q8: alloc %lld failed\n", (long long)total_elems); return; }
        gguf_dequantize((const uint8_t *)W, weight_type, total_elems, f32_w);
        #pragma omp parallel for if(n_cols > 8)
        for (int64_t j = 0; j < n_cols; j++) {
            const block_q8_K *q8 = (const block_q8_K *)q8_x;
            float sum = 0.0f;
            for (int64_t qb = 0; qb < (n_rows + QK_K - 1) / QK_K; qb++) {
                float dq = q8[qb].d;
                for (int l = 0; l < 256 && qb * 256 + l < n_rows; l++) {
                    sum += dq * (float)q8[qb].qs[l] * f32_w[qb * 256 + l + j * n_rows];
                }
            }
            y[j] = sum;
        }
        free(f32_w);
        return;
    }

    // Handle F32 (type 0) - direct F32 dot product
    if (weight_type == GGML_TYPE_F32) {
        #pragma omp parallel for if(n_cols > 8)
        for (int64_t j = 0; j < n_cols; j++) {
            const float *w_col = (const float *)W + j * n_rows;
            const block_q8_K *q8 = (const block_q8_K *)q8_x;
            float sum = 0.0f;
            for (int64_t qb = 0; qb < (n_rows + QK_K - 1) / QK_K; qb++) {
                float dq = q8[qb].d;
                for (int l = 0; l < 256 && qb * 256 + l < n_rows; l++) {
                    sum += dq * (float)q8[qb].qs[l] * w_col[qb * 256 + l];
                }
            }
            y[j] = sum;
        }
        return;
    }

    typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    vec_dot_fn dot_fn = NULL;

    void q4_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void q5_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void q6_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq2_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq3_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq4_xs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);

    switch (weight_type) {
        case GGML_TYPE_IQ2_XXS: dot_fn = (vec_dot_fn)iq2_xxs_vec_dot; break;
        case GGML_TYPE_IQ3_XXS: dot_fn = (vec_dot_fn)iq3_xxs_vec_dot; break;
        case GGML_TYPE_IQ4_XS:  dot_fn = (vec_dot_fn)iq4_xs_vec_dot;  break;
        case GGML_TYPE_Q5_K:    dot_fn = (vec_dot_fn)q5_K_vec_dot;    break;
        case GGML_TYPE_Q4_K:    dot_fn = (vec_dot_fn)q4_K_vec_dot;    break;
        case GGML_TYPE_Q6_K:    dot_fn = (vec_dot_fn)q6_K_vec_dot;    break;
        default:
            fprintf(stderr, "quantized_matmul_from_q8: unsupported quant type %d\n", weight_type);
            return;
    }

    int64_t blk_sz = block_size_for_type(weight_type);
    int64_t n_blocks_per_col = (n_rows + QK_K - 1) / QK_K;
    int64_t col_stride = (col_stride_bytes > 0) ? col_stride_bytes : (n_blocks_per_col * blk_sz);

    #pragma omp parallel for if(n_cols > 8)
    for (int64_t j = 0; j < n_cols; j++) {
        const void *w_col = (const uint8_t *)W + j * col_stride;
        if (j + 1 < n_cols) {
            _mm_prefetch((const char *)W + (j + 1) * col_stride, _MM_HINT_T0);
        }
        dot_fn((int)n_rows, &y[j], 0, w_col, 0, q8_x, 0, 1);
    }
}
