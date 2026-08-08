#ifndef GGUF_READER_H
#define GGUF_READER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// GGML type enum (relevant subset)
enum ggml_type {
    GGML_TYPE_F32    = 0,
    GGML_TYPE_F16    = 1,
    GGML_TYPE_Q4_0   = 2,
    GGML_TYPE_Q4_1   = 3,
    GGML_TYPE_Q5_0   = 6,
    GGML_TYPE_Q5_1   = 7,
    GGML_TYPE_Q8_0   = 8,
    GGML_TYPE_Q8_1   = 9,
    GGML_TYPE_Q2_K   = 10,
    GGML_TYPE_Q3_K   = 11,
    GGML_TYPE_Q4_K   = 12,
    GGML_TYPE_Q5_K   = 13,
    GGML_TYPE_Q6_K   = 14,
    GGML_TYPE_Q8_K   = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,  // bfloat16 (IEEE)
    /* TurboQuant family (IDs per TheTom/llama-cpp-turboquant tom/merge-upstream-dsv4) */
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    /* TurboQuant branch extensions (IDs 36-47, PACE 2026-08-04) */
    GGML_TYPE_MXFP4   = 39,  /* OCP Microscaling FP4: E2M1 + E8M0 per 32-elm block */
    GGML_TYPE_NVFP4   = 40,  /* NVIDIA FP4: E2M1 + E4M3 mscale per 16-elm sub-block (64 total) */
    GGML_TYPE_Q1_0    = 41,  /* 1.5625 bpw: 4-bit code + 1-bit sign per 32-element block */
    GGML_TYPE_TURBO2_0 = 42, /* TurboQuant 2-bit */
    GGML_TYPE_TURBO3_0 = 43, /* TurboQuant 3-bit */
    GGML_TYPE_TURBO4_0 = 44, /* TurboQuant 4-bit */
    GGML_TYPE_TQ3_1S  = 45,  /* WHT-rotated 3-bit Lloyd-Max, block 32, 16 B */
    GGML_TYPE_TQ4_1S  = 46,  // WHT-rotated 4-bit Lloyd-Max, block 32, 20 B
    GGML_TYPE_Q2_0    = 47,  /* 2-bit, block 64, 18 B (branch Q2_0; legacy alias 42) */
};

// GGUF tensor info
typedef struct {
    char name[256];
    int n_dims;
    int64_t dims[4];
    int ggml_type;        // enum ggml_type
    uint64_t data_offset; // offset from data blob start
} gguf_tensor_info;

// GGUF context
typedef struct {
    // File info
    uint32_t version;
    int64_t n_tensors;
    int64_t n_kv;
    uint32_t alignment;

    // Tensor info array
    gguf_tensor_info *tensors;

    // Data blob location in file
    uint64_t data_blob_offset;

    // File handle
    FILE *file;

    // Optional: buffered data blob (mmap or malloc'd copy)
    void *data_blob;
    size_t data_blob_size;

    // Track if data_blob is mmap'd (for proper cleanup)
    int data_blob_is_mmap;

    // File size (for clamping) + per-tensor raw byte spans derived from the
    // file's own data offsets (the byte truth for unknown/TurboQuant types)
    long file_size;
    int64_t *tensor_raw_bytes;   // n_tensors entries

    // KV value store (captured during open, small types only):
    // 32 slots, key → {type, value bytes}. Used by gguf_kv_get_i32/f32/arr.
    int n_kv_store;
    struct gguf_kv_ent { char key[64]; int32_t type; uint8_t data[256]; int64_t len; } kv_store[32];
} gguf_ctx;

// Open GGUF file and parse headers
gguf_ctx* gguf_open(const char *path);

// Buffer entire data blob in RAM for fast tensor reads
// Call after gguf_open, before any gguf_read_tensor_f32 calls
int gguf_buffer_data(gguf_ctx *ctx);

// Calculate raw (quantized) byte size for a tensor type/element count
int64_t gguf_raw_size(int ggml_type, int64_t n_elems);

// The ONE canonical half -> float converter. Every dequant path
// (gguf_dequantize, wubu_weight, quantized_matmul) must call this —
// inline F16 copies have historically diverged (zero -> 6.1e-5 bug).
float gguf_f16_to_f32(uint16_t h);

// Dequantize raw quantized bytes to f32
void gguf_dequantize(const uint8_t *data, int ggml_type, int64_t n_elems, float *output);

// Find a tensor by name
gguf_tensor_info* gguf_find_tensor(gguf_ctx *ctx, const char *name);

// KV value getters (values captured during gguf_open). Return 1 on hit.
// type_filter: GGUF KV type (4=u32, 5=i32, 6=f32, 9=array) or 0 for any.
int gguf_kv_get_i32(gguf_ctx *ctx, const char *key, int *out);
int gguf_kv_get_f32(gguf_ctx *ctx, const char *key, float *out);
/* array of i32: copies up to max_n elements. Returns count, or -1. */
int gguf_kv_get_i32_arr(gguf_ctx *ctx, const char *key, int *out, int max_n);

// Read tensor data (dequantized to float32)
// Returns number of floats written, or 0 on error
int gguf_read_tensor_f32(gguf_ctx *ctx, gguf_tensor_info *tensor, float *output, int64_t max_elems);

// Close GGUF file and free context
void gguf_close(gguf_ctx *ctx);

// IQ1_S / Q6_K dequantization (called internally by gguf_read_tensor_f32)
void dequantize_q6_K_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_q4_K_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq1_s_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq2_xxs_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq2_s_row(const uint8_t *data, float *output, int64_t n_elems);

// IQ2_XXS block-level operations (for on-the-fly dequant dot product)
void dequantize_iq2_xxs_block(const uint8_t *block, float *output);
float iq2_xxs_dot_block(const uint8_t *block, const float *x);
float iq2_xxs_dot_row(const uint8_t *qdata, int n_elems, const float *x);

// Embedding ops
void wubu_exp_map(const float *input, int dim, float R, float *output);
void wubu_log_map(const float *input, int dim, float R, float *output);
float wubu_norm(const float *v, int dim);
float wubu_dot(const float *a, const float *b, int dim);

// IQ block block sizes (256 elements per block)
#define IQ2_XXS_BLOCK_SIZE 66
#define IQ3_XXS_BLOCK_SIZE 98
#define IQ4_XS_BLOCK_SIZE  136

// Block struct for Q8_K (256 elements)
typedef struct {
    float   d;               // delta (scale)
    int8_t  qs[256];         // quants (256 int8 values)
    int16_t bsums[16];       // sum of quants in groups of 16
} block_q8_K;

// Quantize F32 values to Q8_K blocks (k must be multiple of 256)
void quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k);

// Generic Q8_K-based quantized matmul
// Optionally activate per-thread SmoothQuant via quantized_matmul_set_smoothquant()
void quantized_matmul(const float *x,
                      const void *W, int weight_type,
                      int64_t n_rows, int64_t n_cols,
                      int64_t col_stride_bytes,
                      float *y);

// Quantized matmul with pre-quantized Q8_K input (reuses activation quant)
void quantized_matmul_from_q8(const void *q8_x,
                              const void *W, int weight_type,
                              int64_t n_rows, int64_t n_cols,
                              int64_t col_stride_bytes,
                              float *y);

// IQ1_S grid table (2048 × uint64) for GPU constant memory upload
const uint64_t *gguf_get_iq1s_grid(void);

#ifdef __cplusplus
}
#endif

#endif /* GGUF_READER_H */
