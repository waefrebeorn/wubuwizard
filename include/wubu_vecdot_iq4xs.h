/*
 * wubu_vecdot_iq4xs.h -- fused IQ4_XS x Q8_K dot product (EG-116).
 *
 * block_iq4_xs = fp16 d(2) + uint16 scales_h + uint8 scales_l[4] + qs[128]
 * = 136 bytes per 256 elements. Values are 4-bit codes into kvalues_iq4nl.
 */
#ifndef WUBU_VECDOT_IQ4XS_H
#define WUBU_VECDOT_IQ4XS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_IQ4XS_BLOCK_SIZE 136

int wubu_vec_dot_iq4_xs_q8_k(int n, float *s, const void *w, const void *q8);
int wubu_vec_dot_iq4_xs_q8_k_scalar(int n, float *s, const void *w, const void *q8);

#if defined(__AVX2__)
int wubu_vec_dot_iq4_xs_q8_k_avx2(int n, float *s, const void *w, const void *q8);
#endif

void wubu_vec_dot_iq4_xs_q8_k_vd(int n, float *s, size_t bs, const void *vx,
                                 size_t bx, const void *vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_VECDOT_IQ4XS_H */
