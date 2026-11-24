/*
 * wubu_vecdot_iq2xs.h -- fused IQ2_XS x Q8_K dot product (EG-117).
 *
 * block_iq2_xs = fp16 d(2) + uint16 qs[32] + uint8 scales[8] = 74 bytes / 256
 * elements: 9-bit grid index + 7-bit sign pattern per 32-element group.
 */
#ifndef WUBU_VECDOT_IQ2XS_H
#define WUBU_VECDOT_IQ2XS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_IQ2XS_BLOCK_SIZE 74

int wubu_vec_dot_iq2_xs_q8_k(int n, float *s, const void *w, const void *q8);
int wubu_vec_dot_iq2_xs_q8_k_scalar(int n, float *s, const void *w, const void *q8);

void wubu_vec_dot_iq2_xs_q8_k_vd(int n, float *s, size_t bs, const void *vx,
                                 size_t bx, const void *vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_VECDOT_IQ2XS_H */
