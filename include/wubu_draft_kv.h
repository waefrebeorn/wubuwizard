#ifndef WUBU_DRAFT_KV_H
#define WUBU_DRAFT_KV_H

/*
 * wubu_draft_kv.h -- EG-43: quantized draft KV, 4-bit keys with a scale per
 * channel and 4-bit values with a scale per token (refines EG-28/EG-36).
 * Pure C11.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Pack K and V matrices [rows,d] into 4-bit nibbles (two per byte).
 * k_scales has d entries (per channel); v_scales has rows entries (per token).
 * Returns 0 or -1.
 */
int wubu_draft_kv_pack(const float *k, const float *v, int rows, int d,
                       uint8_t *kq, uint8_t *vq,
                       float *k_scales, float *v_scales);

/* Restore either matrix from its nibble-packed form. */
int wubu_draft_kv_unpack_k(const uint8_t *kq, int rows, int d,
                           const float *k_scales, float *out);
int wubu_draft_kv_unpack_v(const uint8_t *vq, int rows, int d,
                           const float *v_scales, float *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_DRAFT_KV_H */
