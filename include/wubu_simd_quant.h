#ifndef WUBU_SIMD_QUANT_H
#define WUBU_SIMD_QUANT_H

/*
 * wubu_simd_quant.h -- LP-15: per-shape native SIMD quant decode.
 *
 * Quant token (uint64_t): high 32 bits = tag, low 32 bits = payload.
 *   tag == 0            -> finite: payload low byte is a signed int8 code,
 *                          value = code * scale.
 *   tag == 0xFFFFFFFFu  -> NaN/Inf: payload is the exact IEEE-754 f32 bit
 *                          pattern, restored bit-for-bit on decode.
 *
 * Dispatch is by element count (tensor shape): 8-wide AVX2 block, 2-wide SSE
 * block, or scalar tail. Paths are observable and forceable for the gate.
 * Opaque module seam: internals stay in the .c file.
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_simd_quant wubu_simd_quant_t;

typedef enum {
    WUBU_SIMD_Q_SCALAR = 0,
    WUBU_SIMD_Q_SSE_2 = 1,
    WUBU_SIMD_Q_AVX2_8 = 2
} wubu_simd_q_path_t;

wubu_simd_quant_t *wubu_simd_quant_create(float scale);
void wubu_simd_quant_free(wubu_simd_quant_t *s);

/* Is AVX2 available on this CPU? */
int wubu_simd_quant_has_avx2(void);

/* Native dispatch for a shape of n elements. */
wubu_simd_q_path_t wubu_simd_quant_dispatch(int n);

/* Encode n floats: int8 code for finite values, raw bit passthrough for
 * NaN/Inf. Returns 0 on success, -1 on bad arguments. */
int wubu_simd_quant_encode(const wubu_simd_quant_t *s, const float *in,
                           uint64_t *out, int n);

/* Decode with native per-shape dispatch; records the path used. */
int wubu_simd_quant_decode(wubu_simd_quant_t *s, const uint64_t *in,
                           float *out, int n);

/* Force a particular kernel (used by the gate to validate every SIMD path). */
int wubu_simd_quant_decode_path(wubu_simd_quant_t *s, wubu_simd_q_path_t path,
                                const uint64_t *in, float *out, int n);

wubu_simd_q_path_t wubu_simd_quant_last_path(const wubu_simd_quant_t *s);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SIMD_QUANT_H */
