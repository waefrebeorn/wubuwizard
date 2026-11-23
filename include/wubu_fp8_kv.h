#ifndef WUBU_FP8_KV_H
#define WUBU_FP8_KV_H

/*
 * wubu_fp8_kv.h -- EG-36a: OCP-style FP8 (E4M3) KV cache storage with a
 * per-row scale, reimplemented with the B07 codec. Pure C11.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Pack one KV row of n floats to n E4M3 bytes. Returns 0 or -1. */
int wubu_fp8_kv_pack(const float *src, int n, uint8_t *out, float *scale);

/* Restore a packed row. Returns 0 or -1. */
int wubu_fp8_kv_unpack(const uint8_t *packed, int n, float scale, float *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_FP8_KV_H */
