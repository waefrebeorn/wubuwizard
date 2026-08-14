#ifndef WUBU_GGUF_TENSOR_H
#define WUBU_GGUF_TENSOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Tensor reading and dequantization dispatch. These functions convert raw
 * GGUF tensor data into float32 (or pass-through for f32/f16). */

/* Read a tensor dequantized to float32. Returns count or 0 on error. */
int gguf_read_tensor_f32(gguf_ctx *ctx, gguf_tensor_info *tensor,
                         float *output, int64_t max_elems);

/* Read raw quantized bytes (no dequantization). */
int gguf_read_raw_tensor(gguf_ctx *ctx, gguf_tensor_info *tensor, void *output);

/* Dequantize raw bytes to float32 in-place. */
void gguf_dequantize(const uint8_t *data, int ggml_type,
                     int64_t n_elems, float *output);

/* Raw byte size for a quantized tensor. */
int64_t gguf_raw_size(int ggml_type, int64_t n_elems);

/* Individual dequant row functions (for GPU/shared use) */
void dequantize_q6_K_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq1_s_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq2_xxs_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq2_s_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq3_xxs_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq3_s_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq1_m_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq4_xs_row(const uint8_t *data, float *output, int64_t n_elems);

/* IQ1_S grid table pointer (for GPU constant memory upload) */
const uint64_t *gguf_get_iq1s_grid(void);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_GGUF_TENSOR_H */