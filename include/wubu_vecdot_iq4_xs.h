/*
 * wubu_vecdot_iq4_xs.h -- fused IQ4_XS x Q8_K dot product (EG-121).
 *
 * Block: fp16 d | uint16 scales_h | uint8 scales_l[4] | uint8 qs[128]
 *        = 136 bytes per 256 elements. The 6-bit per-32-element scale is signed
 *        (ls - 32) and the two high bits come from scales_h.
 */
#ifndef WUBU_VECDOT_IQ4_XS_H
#define WUBU_VECDOT_IQ4_XS_H

#ifdef __cplusplus
extern "C" {
#endif

float wubu_vec_dot_iq4_xs_q8_K(const void *vx, const void *vy, int n);

/* 16 x int8 code table, so the CUDA bridge can copy it to the device. */
const void *wubu_vecdot_iq4_xs_table(void);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_VECDOT_IQ4_XS_H */
