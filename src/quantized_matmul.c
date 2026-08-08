/**
 * quantized_matmul.c — Generic Q8_K-based quantized matrix multiplication.
 *
 * Self-contained: all vec_dot implementations in src/quantized_dot_generic.c.
 * For each output column, quantizes the F32 input to Q8_K then calls
 * the appropriate ggml_vec_dot_{type}_q8_K function.
 *
 * Supports: F32, IQ2_XXS, IQ3_XXS, IQ4_XS, Q5_K, Q6_K
 * Falls back to SGEMM for F32/F16 types.
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <immintrin.h>  // _mm_prefetch
#include <omp.h>

#include "gguf_reader.h"
#include "wubu_ssm.h"

// ========================================================================
// Block sizes (from ggml-common.h)
// ========================================================================
#define QK_K 256
// block_q8_K: float d (4) + int8_t qs[256] (256) + int16_t bsums[16] (32) = 292
#define Q8K_BLOCK_SIZE 292
// block_iq2_xxs: uint16_t d (2) + uint16_t qs[32] (64) = 66
#define IQ2XXS_BLOCK_SIZE 66
// block_iq3_xxs: uint16_t d (2) + uint8_t qs[96] (96) = 98
#define IQ3XXS_BLOCK_SIZE 98
// block_iq4_xs: uint16_t d (2) + uint16_t scales_h (2) + uint8_t scales_l[4] + uint8_t qs[128] = 136
#define IQ4XS_BLOCK_SIZE 136
// block_q5_K: ggml_half d (2) + ggml_half dmin (2) + uint8_t scales[12] + uint8_t qh[32] + uint8_t qs[128] = 176
#define Q5K_BLOCK_SIZE 176
// block_q4_K: ggml_half d (2) + ggml_half dmin (2) + uint8_t scales[12] + uint8_t qs[128] = 144
#define Q4K_BLOCK_SIZE 144
// block_q6_K: ggml_half d (2) + uint8_t ql[128] + uint8_t qh[64] + int8_t scales[16] = 210
#define Q6K_BLOCK_SIZE 210
// block_q2_K: scales[16] + qs[64] + ggml_half d (2) + ggml_half dmin (2) = 84
#define Q2K_BLOCK_SIZE 84
// block_q3_K: hmask[32] + qs[64] + scales[12] + ggml_half d (2) = 110
#define Q3K_BLOCK_SIZE 110

// ========================================================================
// Raw size per type (elements → bytes)
// ========================================================================
static int64_t raw_size_for_type(int ggml_type, int64_t n_elems) {
    int64_t n_blocks = (n_elems + QK_K - 1) / QK_K;
    switch (ggml_type) {
        case GGML_TYPE_F32:      return n_elems * 4;
        case GGML_TYPE_F16:      return n_elems * 2;
        case GGML_TYPE_IQ2_XXS:  return n_blocks * IQ2XXS_BLOCK_SIZE;
        case GGML_TYPE_IQ3_XXS:  return n_blocks * IQ3XXS_BLOCK_SIZE;
        case GGML_TYPE_IQ4_XS:   return n_blocks * IQ4XS_BLOCK_SIZE;
        case GGML_TYPE_Q5_K:     return n_blocks * Q5K_BLOCK_SIZE;
        case GGML_TYPE_Q4_K:     return n_blocks * Q4K_BLOCK_SIZE;
        case GGML_TYPE_Q6_K:     return n_blocks * Q6K_BLOCK_SIZE;
        case GGML_TYPE_Q2_K:     return n_blocks * Q2K_BLOCK_SIZE;
        case GGML_TYPE_Q3_K:     return n_blocks * Q3K_BLOCK_SIZE;
        case GGML_TYPE_BF16:     return n_elems * 2;
        default:
            fprintf(stderr, "quantized_matmul: unsupported type %d\n", ggml_type);
            return 0;
    }
}

// ========================================================================
// Block size per type (elements per block → byte offset per column)
// ========================================================================
static int64_t block_size_for_type(int ggml_type) {
    switch (ggml_type) {
        case GGML_TYPE_F32:      return 4;     // per element
        case GGML_TYPE_F16:      return 2;     // per element
        case GGML_TYPE_IQ2_XXS:  return IQ2XXS_BLOCK_SIZE;
        case GGML_TYPE_IQ3_XXS:  return IQ3XXS_BLOCK_SIZE;
        case GGML_TYPE_IQ4_XS:   return IQ4XS_BLOCK_SIZE;
        case GGML_TYPE_Q5_K:     return Q5K_BLOCK_SIZE;
        case GGML_TYPE_Q4_K:     return Q4K_BLOCK_SIZE;
        case GGML_TYPE_Q6_K:     return Q6K_BLOCK_SIZE;
        case GGML_TYPE_Q2_K:     return Q2K_BLOCK_SIZE;
        case GGML_TYPE_Q3_K:     return Q3K_BLOCK_SIZE;
        case GGML_TYPE_BF16:     return 2;   // per element
        default:                 return 0;
    }
}

// ========================================================================
// Core quantized matmul: y = x @ W
//
// x:  [n_rows] F32 input (will be quantized to Q8_K internally)
// W:  quantized weight data (column-major: each column has n_rows elements)
// type: GGML type of W (IQ2_XXS, Q5_K, etc.)
// n_rows: number of rows (elements per column)
// n_cols: number of columns (output dimension)
// col_stride_bytes: byte stride between columns in W (0 = packed)
// y:  [n_cols] F32 output
//
// Thread-safe: uses OpenMP for column parallelism
// ========================================================================
void quantized_matmul(const float *x,
                      const void *W, int weight_type,
                      int64_t n_rows, int64_t n_cols,
                      int64_t col_stride_bytes,
                      float *y) {
    if (n_rows <= 0 || n_cols <= 0) return;
    
    // Handle F32 directly (no quantization needed)
    if (weight_type == GGML_TYPE_F32) {
        const float *w = (const float *)W;
        int64_t stride = (col_stride_bytes > 0) ? (col_stride_bytes / 4) : n_rows;
        #pragma omp parallel for if(n_cols > 16)
        for (int64_t j = 0; j < n_cols; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < n_rows; k++) {
                sum += x[k] * w[k + j * stride];
            }
            y[j] = sum;
        }
        return;
    }
    
    // Handle F16: dequantize to F32, then SGEMM
    if (weight_type == GGML_TYPE_F16) {
        const uint16_t *w = (const uint16_t *)W;
        int64_t stride_elems = (col_stride_bytes > 0) ? (col_stride_bytes / 2) : n_rows;
        #pragma omp parallel for if(n_cols > 16)
        for (int64_t j = 0; j < n_cols; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < n_rows; k++) {
                // F16 to F32
                uint16_t h = w[k + j * stride_elems];
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp  = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x03FF;
                uint32_t f32;
                if (exp == 0) {
                    f32 = (sign << 31) | ((uint32_t)(127 - 15 + 1) << 23) | (mant << 13);
                } else if (exp == 31) {
                    f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
                } else {
                    f32 = (sign << 31) | ((uint32_t)(127 - 15 + exp) << 23) | (mant << 13);
                }
                float val;
                memcpy(&val, &f32, 4);
                sum += x[k] * val;
            }
            y[j] = sum;
        }
        return;
    }
    
    // Handle BF16: dequantize to F32, then SGEMM
    // Also handles type 30 (older BF16 enum value from newer GGUF files)
    if (weight_type == GGML_TYPE_BF16 || weight_type == 30) {
        const uint16_t *w = (const uint16_t *)W;
        int64_t stride_elems = (col_stride_bytes > 0) ? (col_stride_bytes / 2) : n_rows;
        #pragma omp parallel for if(n_cols > 16)
        for (int64_t j = 0; j < n_cols; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < n_rows; k++) {
                uint32_t bits = (uint32_t)w[k + j * stride_elems] << 16;  // BF16 = high 16 bits of F32
                float val;
                memcpy(&val, &bits, 4);
                sum += x[k] * val;
            }
            y[j] = sum;
        }
        return;
    }
    
    // Handle Q8_0: dequant on-the-fly, SGEMM
    // Block size 32: d(half) [2] + qs[32] = 34 bytes per 32 elements
    if (weight_type == GGML_TYPE_Q8_0) {
        const int64_t BLK = 32, BLK_BYTES = 34;
        int64_t n_blocks_per_col = (n_rows + BLK - 1) / BLK;
        int64_t stride = (col_stride_bytes > 0) ? col_stride_bytes : n_blocks_per_col * BLK_BYTES;
        #pragma omp parallel for if(n_cols > 8)
        for (int64_t j = 0; j < n_cols; j++) {
            const uint8_t *wj = (const uint8_t *)W + j * stride;
            float sum = 0.0f;
            for (int64_t b = 0; b < n_blocks_per_col; b++) {
                const uint8_t *blk = wj + b * BLK_BYTES;
                uint16_t d_bits; memcpy(&d_bits, blk, 2);
                // F16 to F32
                uint32_t sign = (d_bits >> 15) & 1;
                uint32_t exp  = (d_bits >> 10) & 0x1F;
                uint32_t mant = d_bits & 0x03FF;
                uint32_t f32;
                if (exp == 0) f32 = (sign << 31) | ((uint32_t)(127 - 15 + 1) << 23) | (mant << 13);
                else if (exp == 31) f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
                else f32 = (sign << 31) | ((uint32_t)(127 - 15 + exp) << 23) | (mant << 13);
                float d; memcpy(&d, &f32, 4);
                const int8_t *qs = (const int8_t *)(blk + 2);
                int64_t remaining = n_rows - b * BLK;
                if (remaining > BLK) remaining = BLK;
                for (int64_t l = 0; l < remaining; l++) {
                    sum += x[b * BLK + l] * (d * (float)qs[l]);
                }
            }
            y[j] = sum;
        }
        return;
    }
    
    // Handle IQ2_S: dequant via gguf_dequantize helper, then SGEMM
    if (weight_type == GGML_TYPE_IQ2_S || weight_type == GGML_TYPE_IQ2_XS || 
        weight_type == GGML_TYPE_IQ1_S || weight_type == GGML_TYPE_IQ1_M ||
        weight_type == GGML_TYPE_IQ3_S ||
        weight_type == GGML_TYPE_Q2_K || weight_type == GGML_TYPE_Q3_K) {
        // These are rare types — dequant entire weight to F32 then SGEMM
        int64_t total_elems = n_rows * n_cols;
        float *f32_w = (float *)malloc(total_elems * sizeof(float));
        if (!f32_w) { fprintf(stderr, "quantized_matmul: alloc %lld failed\n", (long long)total_elems); return; }
        gguf_dequantize((const uint8_t *)W, weight_type, total_elems, f32_w);
        // Now do SGEMM
        #pragma omp parallel for if(n_cols > 8)
        for (int64_t j = 0; j < n_cols; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < n_rows; k++) {
                sum += x[k] * f32_w[k + j * n_rows];
            }
            y[j] = sum;
        }
        free(f32_w);
        return;
    }
    
    // Quantized types: use Q8_K activation quantization + ggml_vec_dot
    int64_t n_q8_blocks = (n_rows + QK_K - 1) / QK_K;
    int64_t q8_size = n_q8_blocks * Q8K_BLOCK_SIZE;
    
    // Stack-allocate Q8_K buffer for small sizes, heap for large
    void *q8_buf = NULL;
    uint8_t stack_buf[4096]; // up to ~14 Q8_K blocks
    if (q8_size <= (int64_t)sizeof(stack_buf)) {
        q8_buf = stack_buf;
    } else {
        q8_buf = malloc(q8_size);
        if (!q8_buf) {
            fprintf(stderr, "quantized_matmul: allocation failed (%ld bytes)\n", (long)q8_size);
            return;
        }
    }
    
    // Quantize input to Q8_K
    quantize_row_q8_K(x, q8_buf, n_rows);
    int64_t blk_sz = block_size_for_type(weight_type);
    int64_t n_blocks_per_col = (n_rows + QK_K - 1) / QK_K;
    int64_t col_stride = (col_stride_bytes > 0) ? col_stride_bytes : (n_blocks_per_col * blk_sz);
    
    // Select the right vec_dot function
    typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    vec_dot_fn dot_fn = NULL;
    
    // Self-contained generic vec_dot (no libggml-cpu.so dependency)
    // Full signature matching ggml_vec_dot_*_q8_K: (n, s, bs, vx, bx, vy, by, nrc)
    void q4_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void q5_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void q6_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq2_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq3_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq4_xs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void ggml_vec_dot_q4_K_q8_K_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    
    switch (weight_type) {
        case GGML_TYPE_IQ2_XXS: dot_fn = (vec_dot_fn)iq2_xxs_vec_dot; break;
        case GGML_TYPE_IQ3_XXS: dot_fn = (vec_dot_fn)iq3_xxs_vec_dot; break;
        case GGML_TYPE_IQ4_XS:  dot_fn = (vec_dot_fn)iq4_xs_vec_dot;  break;
        case GGML_TYPE_Q5_K:    dot_fn = (vec_dot_fn)q5_K_vec_dot;    break;
        case GGML_TYPE_Q4_K:    dot_fn = (vec_dot_fn)ggml_vec_dot_q4_K_q8_K_generic; break;
        case GGML_TYPE_Q6_K:    dot_fn = (vec_dot_fn)q6_K_vec_dot;    break;
        default:
            fprintf(stderr, "quantized_matmul: unsupported quant type %d\n", weight_type);
            if (q8_buf != stack_buf) free(q8_buf);
            return;
    }
    
    // Compute each column using the vec_dot function
    // Prefetch next column's weight data to L1 while computing current column
    #pragma omp parallel for if(n_cols > 8)
    for (int64_t j = 0; j < n_cols; j++) {
        const void *w_col = (const uint8_t *)W + j * col_stride;
        // Prefetch next column into L1 cache
        if (j + 1 < n_cols) {
            _mm_prefetch((const char *)W + (j + 1) * col_stride, _MM_HINT_T0);
        }
        dot_fn((int)n_rows, &y[j], 0, w_col, 0, q8_buf, 0, 1);
    }
    
    // Debug: for Q4_K output projection, check first few results
    if (n_cols > 100000 && weight_type == GGML_TYPE_Q4_K && getenv("QUANTIZED_MATMUL_DEBUG")) {
        int nonz = 0;
        for (int j = 0; j < 1000; j++) if (fabsf(y[j]) > 1e-10f) nonz++;
        printf("  [quantized_matmul Q4_K] n_rows=%ld n_cols=%ld first5: %.6f %.6f %.6f %.6f %.6f nonz_1000=%d\\n",
               (long)n_rows, (long)n_cols, (double)y[0], (double)y[1], (double)y[2], (double)y[3], (double)y[4], nonz);
    }
    
    if (q8_buf != stack_buf) free(q8_buf);
}

// ========================================================================
// Quantized matmul for MoE expert: single expert's gate/up/down
// Matches ggml_mul_mat_id for one expert exactly
//
// x: [D_MODEL] F32 input
// gate_q: IQ2_XXS weight [D_MODEL, D_FF]
// up_q:   IQ2_XXS weight [D_MODEL, D_FF]
// down_q: IQ3_XXS weight [D_FF, D_MODEL]
// temp: [D_FF * 3] scratch
// output: [D_MODEL]
// ========================================================================
void moe_expert_forward_lib(const float *x,
                            const void *gate_q, int gate_type,
                            const void *up_q,   int up_type,
                            const void *down_q, int down_type,
                            int64_t n_ff,
                            float *temp, float *output) {
    // temp layout: [gate_out(n_ff) | up_out(n_ff) | act(n_ff)]
    float *gate_out = temp;
    float *up_out   = temp + n_ff;
    float *act      = temp + 2 * n_ff;
    
    // gate = x @ gate_q  [D_MODEL] @ [D_MODEL, n_ff] -> [n_ff]
    quantized_matmul(x, gate_q, gate_type,
                     D_MODEL, n_ff, 0, gate_out);
    
    // up = x @ up_q  [D_MODEL] @ [D_MODEL, n_ff] -> [n_ff]
    quantized_matmul(x, up_q, up_type,
                     D_MODEL, n_ff, 0, up_out);
    
    // act = silu(gate) * up
    for (int64_t j = 0; j < n_ff; j++) {
        float g = gate_out[j];
        float silu_g;
        if (g < -80.0f) silu_g = 0.0f;
        else silu_g = g / (1.0f + expf(-g));
        act[j] = silu_g * up_out[j];
    }
    
    // output = act @ down_q  [n_ff] @ [n_ff, D_MODEL] -> [D_MODEL]
    quantized_matmul(act, down_q, down_type,
                     n_ff, D_MODEL, 0, output);
}

// ========================================================================
// Quantized matmul with pre-quantized Q8_K activation
//
// Like quantized_matmul() but caller provides a pre-quantized Q8_K buffer.
// This allows multiple matmuls sharing the same input to quantize once.
//
// q8_x: pre-quantized Q8_K buffer (from quantize_row_q8_K, must be [n_rows])
// W: quantized weight
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
        weight_type == GGML_TYPE_Q2_K || weight_type == GGML_TYPE_Q3_K) {
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

    typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    vec_dot_fn dot_fn = NULL;

    void q4_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void q5_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void q6_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq2_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq3_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq4_xs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void ggml_vec_dot_q4_K_q8_K_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);

    switch (weight_type) {
        case GGML_TYPE_IQ2_XXS: dot_fn = (vec_dot_fn)iq2_xxs_vec_dot; break;
        case GGML_TYPE_IQ3_XXS: dot_fn = (vec_dot_fn)iq3_xxs_vec_dot; break;
        case GGML_TYPE_IQ4_XS:  dot_fn = (vec_dot_fn)iq4_xs_vec_dot;  break;
        case GGML_TYPE_Q5_K:    dot_fn = (vec_dot_fn)q5_K_vec_dot;    break;
        case GGML_TYPE_Q4_K:    dot_fn = (vec_dot_fn)ggml_vec_dot_q4_K_q8_K_generic; break;
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


// ========================================================================
// Batched quantized matmul: N input vectors through the same weight.
// x:  [N, D_MODEL] F32 input
// W:  quantized weight [D_MODEL, n_cols]
// Key: weight data read ONCE from RAM, shared across all N tokens.
// Supports: F32, IQ2_XXS, IQ3_XXS, IQ4_XS, Q5_K, Q6_K, Q4_K
// ========================================================================
void quantized_matmul_batched(const float *x,
                              const void *W, int weight_type,
                              int64_t n_rows, int64_t n_cols,
                              int64_t col_stride_bytes,
                              int N,
                              float *y) {
    if (n_rows <= 0 || n_cols <= 0 || N <= 0) return;

    if (weight_type == GGML_TYPE_F32) {
        const float *w = (const float *)W;
        int64_t stride = (col_stride_bytes > 0) ? (col_stride_bytes / 4) : n_rows;
        #pragma omp parallel for collapse(2) if(N * n_cols > 64)
        for (int i = 0; i < N; i++)
            for (int64_t j = 0; j < n_cols; j++) {
                float sum = 0.0f;
                for (int64_t k = 0; k < n_rows; k++)
                    sum += x[i * n_rows + k] * w[k + j * stride];
                y[i * n_cols + j] = sum;
            }
        return;
    }

    if (weight_type == GGML_TYPE_F16 || weight_type == GGML_TYPE_BF16 || weight_type == 30 ||
        weight_type == GGML_TYPE_IQ1_M || weight_type == GGML_TYPE_IQ1_S ||
        weight_type == GGML_TYPE_IQ2_S || weight_type == GGML_TYPE_IQ2_XS ||
        weight_type == GGML_TYPE_IQ3_S ||
        weight_type == GGML_TYPE_Q2_K || weight_type == GGML_TYPE_Q3_K ||
        weight_type == GGML_TYPE_Q8_0) {
        for (int i = 0; i < N; i++)
            quantized_matmul(x + i * n_rows, W, weight_type, n_rows, n_cols, col_stride_bytes, y + i * n_cols);
        return;
    }

    int64_t n_q8_blocks = (n_rows + QK_K - 1) / QK_K;
    int64_t q8_tok_bytes = n_q8_blocks * Q8K_BLOCK_SIZE;
    int64_t q8_total = N * q8_tok_bytes;

    void *q8_all = NULL;
    uint8_t stack_buf[16384];
    if (q8_total <= (int64_t)sizeof(stack_buf))
        q8_all = stack_buf;
    else {
        q8_all = malloc(q8_total);
        if (!q8_all) return;
    }

    for (int i = 0; i < N; i++)
        quantize_row_q8_K(x + i * n_rows, (block_q8_K *)((uint8_t *)q8_all + i * q8_tok_bytes), n_rows);

    typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    vec_dot_fn dot_fn = NULL;
    void q4_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void q5_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void q6_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq2_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq3_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq4_xs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void ggml_vec_dot_q4_K_q8_K_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);

    switch (weight_type) {
        case GGML_TYPE_IQ2_XXS: dot_fn = (vec_dot_fn)iq2_xxs_vec_dot; break;
        case GGML_TYPE_IQ3_XXS: dot_fn = (vec_dot_fn)iq3_xxs_vec_dot; break;
        case GGML_TYPE_IQ4_XS:  dot_fn = (vec_dot_fn)iq4_xs_vec_dot;  break;
        case GGML_TYPE_Q5_K:    dot_fn = (vec_dot_fn)q5_K_vec_dot;    break;
        case GGML_TYPE_Q4_K:    dot_fn = (vec_dot_fn)ggml_vec_dot_q4_K_q8_K_generic; break;
        case GGML_TYPE_Q6_K:    dot_fn = (vec_dot_fn)q6_K_vec_dot;    break;
        default:
            fprintf(stderr, "quantized_matmul_batched: unsupported type %d\n", weight_type);
            if (q8_all != stack_buf) free(q8_all);
            return;
    }

    int64_t blk_sz = block_size_for_type(weight_type);
    int64_t n_blocks_per_col = (n_rows + QK_K - 1) / QK_K;
    int64_t col_stride = (col_stride_bytes > 0) ? col_stride_bytes : (n_blocks_per_col * blk_sz);

    #pragma omp parallel for if(n_cols > 32)
    for (int64_t j = 0; j < n_cols; j++) {
        const void *w_col = (const uint8_t *)W + j * col_stride;
        _mm_prefetch((const char *)W + ((j + 1) < n_cols ? (j + 1) : j) * col_stride, _MM_HINT_T0);
        for (int i = 0; i < N; i++) {
            const void *q8_i = (const uint8_t *)q8_all + i * q8_tok_bytes;
            dot_fn((int)n_rows, &y[i * n_cols + j], 0, w_col, 0, q8_i, 0, 1);
        }
    }

    if (q8_all != stack_buf) free(q8_all);
}

// Subset matmul: compute only specified columns (for output proj logit cache)
// Supports Q5_K, Q4_K, Q6_K (types with vec_dot). Other types fall back to dequant+SGEMM.
void quantized_matmul_subset(const float *x,
                              const void *W, int weight_type,
                              int64_t n_rows,
                              const int *col_indices, int n_cols,
                              int64_t col_stride_bytes,
                              float *y) {
    if (n_cols <= 0) return;
    
    // Handle non-vec_dot types via dequant
    if (weight_type == GGML_TYPE_IQ1_M || weight_type == GGML_TYPE_IQ1_S ||
        weight_type == GGML_TYPE_IQ2_S || weight_type == GGML_TYPE_IQ2_XS ||
        weight_type == GGML_TYPE_IQ3_S ||
        weight_type == GGML_TYPE_Q2_K || weight_type == GGML_TYPE_Q3_K) {
        // For these types, just compute each column individually
        int64_t blk_sz = block_size_for_type(weight_type);
        int64_t n_blocks = (n_rows + QK_K - 1) / QK_K;
        int64_t stride = (col_stride_bytes > 0) ? col_stride_bytes : (n_blocks * blk_sz);
        #pragma omp parallel for if(n_cols > 4)
        for (int i = 0; i < n_cols; i++) {
            int64_t j = col_indices[i];
            const uint8_t *wj = (const uint8_t *)W + j * stride;
            float *f32_buf = (float *)malloc(n_rows * sizeof(float));
            gguf_dequantize(wj, weight_type, n_rows, f32_buf);
            double sum = 0.0;
            for (int64_t k = 0; k < n_rows; k++)
                sum += (double)x[k] * f32_buf[k];
            y[i] = (float)sum;
            free(f32_buf);
        }
        return;
    }
    
    // Vec_dot types: quantize x to Q8_K, then dot per selected column
    block_q8_K q8_buf[32];  // max 8192/256 = 32 blocks — enough for D_MODEL up to 8192
    
    quantize_row_q8_K(x, q8_buf, n_rows);
    
    // Declare vec_dot functions (same as quantized_matmul_batched)
    typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    vec_dot_fn dot_fn = NULL;
    
    void q4_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void q5_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void q6_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    
    switch (weight_type) {
        case GGML_TYPE_Q4_K: dot_fn = (vec_dot_fn)q4_K_vec_dot; break;
        case GGML_TYPE_Q5_K: dot_fn = (vec_dot_fn)q5_K_vec_dot; break;
        case GGML_TYPE_Q6_K: dot_fn = (vec_dot_fn)q6_K_vec_dot; break;
        default:
            fprintf(stderr, "quantized_matmul_subset: unsupported type %d\n", weight_type);
            return;
    }
    
    int64_t blk_sz = block_size_for_type(weight_type);
    int64_t n_blocks = (n_rows + QK_K - 1) / QK_K;
    int64_t stride = (col_stride_bytes > 0) ? col_stride_bytes : (n_blocks * blk_sz);
    
    #pragma omp parallel for if(n_cols > 8)
    for (int i = 0; i < n_cols; i++) {
        int64_t j = col_indices[i];
        const void *wj = (const uint8_t *)W + j * stride;
        dot_fn((int)n_rows, &y[i], 0, wj, 0, q8_buf, 0, 1);
    }
}
