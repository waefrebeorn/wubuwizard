/*
 * wubu_vecdot_iq.h -- fused IQ2_S / IQ3_S x Q8_K dot products (EG-111).
 *
 * The 12B IQ2 GGUF this engine runs is 204 IQ2_S + 95 IQ3_S tensors. Until now
 * those types had no Q8 vec_dot, so quantized_matmul dequantised every weight
 * matrix to F32 per call (see docs/ENGINE-THROUGHPUT-BASELINE.md: 0.1 tok/s
 * decode). These functions compute the dot straight from the packed 2.5625 bpw
 * / 3.4375 bpw blocks against Q8_K activations: no F32 weight buffer.
 *
 * Ported from ggml's reference scalar kernels (llama/ggml-cpu/quants.c,
 * ggml_vec_dot_iq2_s_q8_K_generic / ggml_vec_dot_iq3_s_q8_K_generic).
 * Strict C11. Integer accumulation, one float multiply per block.
 */
#ifndef WUBU_VECDOT_IQ_H
#define WUBU_VECDOT_IQ_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_IQ2S_BLOCK_SIZE 82   /* fp16 d(2) + qs[64] + qh[8] + scales[8]  */
#define WUBU_IQ3S_BLOCK_SIZE 110  /* fp16 d(2) + qs[64] + qh[8] + signs[32] + scales[4] */

/* n must be a multiple of 256. Returns 0 on success, -1 on bad arguments.
 * The plain names dispatch to AVX2 when the translation unit is built with
 * -mavx2, otherwise to the scalar port; both are always available for
 * cross-checking. */
int wubu_vec_dot_iq2_s_q8_k(int n, float *s, const void *w, const void *q8);
int wubu_vec_dot_iq3_s_q8_k(int n, float *s, const void *w, const void *q8);

int wubu_vec_dot_iq2_s_q8_k_scalar(int n, float *s, const void *w, const void *q8);
int wubu_vec_dot_iq3_s_q8_k_scalar(int n, float *s, const void *w, const void *q8);

#if defined(__AVX2__)
int wubu_vec_dot_iq2_s_q8_k_avx2(int n, float *s, const void *w, const void *q8);
int wubu_vec_dot_iq3_s_q8_k_avx2(int n, float *s, const void *w, const void *q8);
#endif

/* ggml-style vec_dot signature so these can sit in the existing dispatch table. */
void wubu_vec_dot_iq2_s_q8_k_vd(int n, float *s, size_t bs, const void *vx,
                                size_t bx, const void *vy, size_t by, int nrc);
void wubu_vec_dot_iq3_s_q8_k_vd(int n, float *s, size_t bs, const void *vx,
                                size_t bx, const void *vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_VECDOT_IQ_H */
