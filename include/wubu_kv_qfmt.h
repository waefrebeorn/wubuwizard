#ifndef WUBU_KV_QFMT_H
#define WUBU_KV_QFMT_H

/*
 * wubu_kv_qfmt.h -- LP-11: KV reconstruction in the compute format.
 *
 * Each element is ONE packed byte holding two nibbles:
 *   high nibble = index into a small deterministic FP16 anchor set (8 values)
 *   low nibble  = NVFP4-style E2M1 4-bit value (sign|exp2|mant1)
 * One E8M0 (MX) block scale - 2^(e-127) - is shared by the block.
 *
 * Reconstruction goes straight to f32; attention dot products accumulate in
 * f64 and return f32. There is NO fp16 intermediate for the attention values:
 * the anchors are fp16 *constants* decoded once to f32, never a rounding step
 * applied to reconstructed K/V or to the dot product.
 *
 * Opaque module seam (AGENTS.md rule 2): only the typedef and functions are
 * public; the anchor table and internal state stay in the .c file.
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_kv_qfmt wubu_kv_qfmt_t;

#define WUBU_KV_QFMT_NANCHORS 8

/* Context creation. The anchor set is fixed and identical on every machine. */
wubu_kv_qfmt_t *wubu_kv_qfmt_create(void);
void wubu_kv_qfmt_free(wubu_kv_qfmt_t *q);

/* Number of packed bytes required for n values (one byte per value). */
size_t wubu_kv_qfmt_packed_bytes(int n);

/* i-th anchor as a C float (decoded from its fp16 bit pattern). */
float wubu_kv_qfmt_anchor(const wubu_kv_qfmt_t *q, int i);

/* Encode n f32 values into packed (anchor_nibble<<4 | fp4_nibble) bytes plus
 * one E8M0 exponent. Returns 0 on success, -1 on bad arguments. */
int wubu_kv_qfmt_encode(const wubu_kv_qfmt_t *q, const float *src, int n,
                        uint8_t *packed, uint8_t *mx_exp);

/* Reconstruct n f32 values directly from packed nibbles + E8M0 scale. The
 * output is never rounded through fp16. Returns 0 on success. */
int wubu_kv_qfmt_decode(const wubu_kv_qfmt_t *q, const uint8_t *packed,
                        uint8_t mx_exp, float *out, int n);

/* Attention dot: decode into f32, accumulate in f64, return f32. No fp16
 * intermediate is used for the reconstructed values. */
int wubu_kv_qfmt_dot(const wubu_kv_qfmt_t *q, const uint8_t *packed,
                     uint8_t mx_exp, const float *query, int n, float *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KV_QFMT_H */
