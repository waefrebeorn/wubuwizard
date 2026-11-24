/*
 * wubu_vecdot_q3_k.h -- fused Q3_K x Q8_K dot product (EG-114).
 *
 * The 12B Gemma-4 IQ2 GGUF keeps its tied token-embedding / LM head as a single
 * Q3_K tensor (412 MB). Without a vec_dot the engine dequantised all of it to F32
 * every token, which dominated decode. This kernel reads the packed 3-bit blocks
 * directly against Q8_K activations.
 *
 * Ported from ggml_vec_dot_q3_K_q8_K_generic (llama/ggml-cpu/quants.c) and
 * ggml_vec_dot_q3_K_q8_K (llama/ggml-cpu/arch/x86/quants.c), MIT.
 */
#ifndef WUBU_VECDOT_Q3_K_H
#define WUBU_VECDOT_Q3_K_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* block_q3_K: fp16 d (2) + hmask[32] + qs[64] + scales[12] = 110 bytes / 256 elems */
#define WUBU_Q3_K_BLOCK_SIZE 110

/* n must be a multiple of 256. 0 on success, -1 on bad arguments. */
int wubu_vec_dot_q3_k_q8_k(int n, float *s, const void *w, const void *q8);
int wubu_vec_dot_q3_k_q8_k_scalar(int n, float *s, const void *w, const void *q8);

#if defined(__AVX2__)
int wubu_vec_dot_q3_k_q8_k_avx2(int n, float *s, const void *w, const void *q8);
#endif

void wubu_vec_dot_q3_k_q8_k_vd(int n, float *s, size_t bs, const void *vx,
                               size_t bx, const void *vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_VECDOT_Q3_K_H */
