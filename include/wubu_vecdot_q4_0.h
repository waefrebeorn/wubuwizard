/*
 * wubu_vecdot_q4_0.h -- fused Q4_0 x Q8_K dot product (EG-113).
 *
 * The tied token-embedding / LM-head matrix of the 12B Gemma-4 IQ2 GGUF is Q4_0
 * (262,144 x 3,840), and the engine had no fused dot for it: every token ran a
 * scalar nibble-unpack per column. Measured effect of the head alone: caching it
 * as F32 (`G4_CACHE_LM_HEAD=1`) takes decode from 1.1 to 2.8 tok/s.
 *
 * This is ggml's Q4_0 x Q8 layout adapted to this repo's Q8_K activations
 * (per-256 block scale): the per-32 sub-block scale of Q4_0 is applied inside the
 * block loop, and the 8 bias of the nibbles is removed with the Q8_K per-16 sums
 * (`block_q8_K.bsums`), so the integer path stays exact.
 */
#ifndef WUBU_VECDOT_Q4_0_H
#define WUBU_VECDOT_Q4_0_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* block_q4_0: fp16 d (2) + qs[16] = 18 bytes per 32 elements */
#define WUBU_Q4_0_BLOCK_SIZE 18
#define WUBU_Q4_0_BLOCK_ELEMS 32

/* n must be a multiple of 256 (Q8_K block size). 0 on success, -1 on bad args. */
int wubu_vec_dot_q4_0_q8_k(int n, float *s, const void *w, const void *q8);
int wubu_vec_dot_q4_0_q8_k_scalar(int n, float *s, const void *w, const void *q8);

#if defined(__AVX2__)
int wubu_vec_dot_q4_0_q8_k_avx2(int n, float *s, const void *w, const void *q8);
#endif

void wubu_vec_dot_q4_0_q8_k_vd(int n, float *s, size_t bs, const void *vx,
                               size_t bx, const void *vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_VECDOT_Q4_0_H */
