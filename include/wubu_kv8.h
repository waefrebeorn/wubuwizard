#ifndef WUBU_KV8_H
#define WUBU_KV8_H

/*
 * wubu_kv8.h -- EG-66d: kv8 INT8 K/V codec.
 * One per-row f32 scale maps signed INT8 codes back to K/V values. This
 * gates the codec and its byte accounting; no attention-quality result is
 * claimed.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_KV8_MAX_DIM 1024

int wubu_kv8_pack(const float *row, int n, signed char *codes, float *scale);
int wubu_kv8_unpack(const signed char *codes, float scale, int n, float *out);

size_t wubu_kv8_storage_bytes(int n);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KV8_H */
