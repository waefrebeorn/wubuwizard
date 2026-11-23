#ifndef WUBU_FP8_ATTN_H
#define WUBU_FP8_ATTN_H

/*
 * wubu_fp8_attn.h -- EG-30: FP8 stored Q/K/V with F32 accumulation in scaled
 * dot-product attention. Reuses the B07 E4M3 codec; no third-party code.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * q: [nq,d], k: [nk,d], v: [nk,d] E4M3 bytes; out: [nq,d] F32.
 * Each tensor has its own scale; scores and softmax accumulate in F32.
 * Returns 0 or -1 on bad arguments.
 */
int wubu_fp8_attn(const uint8_t *q, int nq, int d, float q_scale,
                  const uint8_t *k, int nk, float k_scale,
                  const uint8_t *v, float v_scale,
                  float *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_FP8_ATTN_H */
