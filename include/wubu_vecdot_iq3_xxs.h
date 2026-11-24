/*
 * wubu_vecdot_iq3_xxs.h -- fused IQ3_XXS x Q8_K dot product (EG-120).
 *
 * IQ3_XXS had no vec_dot, so the 20 IQ3_XXS tensors in the 12B Gemma file (282 MB,
 * 6.7% of the weights) were dequantised to F32 on the CPU on every decode step.
 * Block layout: fp16 d + 64 bytes of 3-bit grid indices + 32 bytes of signs/scales
 * = 98 bytes per 256 elements (QK_K).
 */
#ifndef WUBU_VECDOT_IQ3_XXS_H
#define WUBU_VECDOT_IQ3_XXS_H

#ifdef __cplusplus
extern "C" {
#endif

/* x: IQ3_XXS blocks, y: Q8_K blocks, n a multiple of 256. Returns the dot. */
float wubu_vec_dot_iq3_xxs_q8_K(const void *vx, const void *vy, int n);

/* Table accessors so the CUDA bridge can copy the grid/signs to the device
 * (256 x u32 grid, 128 x u8 signs). */
const void *wubu_vecdot_iq3_xxs_grid(void);
const void *wubu_vecdot_iq3_xxs_signs(void);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_VECDOT_IQ3_XXS_H */
