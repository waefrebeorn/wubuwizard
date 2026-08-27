#ifndef WUBU_FLASH_ATTN_H
#define WUBU_FLASH_ATTN_H

#include <stdint.h>
#include <stddef.h>

/* FlashAttention-style tiled attention (CPU online softmax). */
void wubu_flash_attn_forward(const float *q, const float *k, const float *v,
                              float *out, int nq, int nkv, int heads,
                              int dim, int causal);

/* Same but outputs FP16 (IEEE 754 half). */
void wubu_flash_attn_forward_f16(const float *q, const float *k, const float *v,
                                  uint16_t *out, int nq, int nkv, int heads,
                                  int dim, int causal);

#endif /* WUBU_FLASH_ATTN_H */
