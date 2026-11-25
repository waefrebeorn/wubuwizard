#ifndef WUBU_DEQUANT_FP4_H
#define WUBU_DEQUANT_FP4_H

#include <stddef.h>

/*
 * wubu_dequant_fp4 -- C11 row-level dequantization for the 4-bit-float and 2-bit
 * microscaling types, matching ggml's on-disk layouts bit for bit.
 *
 * MXFP4 (ggml type 39): 32-element blocks
 *   1 byte E8M0 shared scale   (scale = 2^(E-128), ggml GGML_E8M0_TO_FP32_HALF)
 *   16 bytes packed E2M1 codes (low nibble -> element j, high nibble -> j+16)
 *   = 17 bytes per 32 elements
 *
 * NVFP4 (ggml type 40): 64-element blocks
 *   4 bytes UE4M3 scales (one per 16-element sub-block)
 *   32 bytes packed E2M1 codes (8 bytes per sub-block: low -> j, high -> j+8)
 *   = 36 bytes per 64 elements
 *
 * Q2_0  (ggml type 42): 64-element blocks
 *   fp16 d + 16 bytes of 2-bit codes (4 per byte, low bits first)
 *   value = ((q & 3) - 1) * d
 *   = 18 bytes per 64 elements
 *
 * E2M1 codes are the full 16-entry table, not 2-bit magnitudes: a code of 4..7 is
 * 4, 6, 8, 12 and 8..15 are the negatives. Tables and scale math are copied from
 * ggml's kvalues_fp4 / ggml_ue4m3_to_fp32 so that a GGUF written by llama.cpp
 * decodes to the same floats here.
 */

/* Raw byte size for a tensor of n_elems elements in the given type.
   Returns -1 for types this module does not handle (39, 40, 42 only). */
long wubu_fp4_raw_size(int ggml_type, long n_elems);

/* Dequantize n_elems floats from a row of quantized data. */
void dequantize_row_mxfp4(const unsigned char *data, float *output, long n_elems);
void dequantize_row_nvfp4(const unsigned char *data, float *output, long n_elems);
void dequantize_row_q2_0(const unsigned char *data, float *output, long n_elems);
/* Dot of a Q2_0 row against x WITHOUT materialising the row as floats.
 * value_j = (q_j - 1) * d, q_j in 0..3, so the block contributes d * (sum q_j*x_j - sum x_j). */
void dot_row_q2_0(const unsigned char *data, const float *x, long n_elems, float *out);

#endif /* WUBU_DEQUANT_FP4_H */
