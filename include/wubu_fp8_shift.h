#ifndef WUBU_FP8_SHIFT_H
#define WUBU_FP8_SHIFT_H

/*
 * wubu_fp8_shift.h -- EG-109: table-free arithmetic E4M3 decode.
 * Inspired by the KTransformers AVX2 `fp8x8_to_fp32x8` patch: magnitude<<20 with
 * a bias constant for normals, magnitude/512 for subnormals, NaN magnitude 127
 * mapped to zero. This is a scalar C11 contract gate; no AVX2 intrinsics or
 * copied kernel bodies are used, and no speedup is claimed.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_FP8_SHIFT_MAX 4096

float wubu_fp8_shift_one(uint8_t code);
int wubu_fp8_shift_decode(const uint8_t *codes, float *out, int n);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_FP8_SHIFT_H */
