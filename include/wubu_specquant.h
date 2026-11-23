#ifndef WUBU_SPECQUANT_H
#define WUBU_SPECQUANT_H

/*
 * wubu_specquant.h -- EG-63: INT4 / FP8 / FP16 variants derived from one base
 * weight tensor, selected by query complexity (SpecQuant lineage).
 * One gate: routed accuracy loss and bytes fetched versus F32.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { WUBU_SQ_INT4 = 0, WUBU_SQ_FP8 = 1, WUBU_SQ_FP16 = 2 };

/* Bytes per element for each variant. */
size_t wubu_specquant_bytes(int variant, int n);

/* Pack n base weights. dst must have wubu_specquant_bytes(variant,n) bytes. */
int wubu_specquant_pack(const float *w, int n, int variant,
                        unsigned char *dst, float *scale);

/* Reconstruct into out[n]. */
int wubu_specquant_unpack(const unsigned char *src, int n, int variant,
                          float scale, float *out);

/* Query complexity: high variance -> FP16, low -> INT4, mid -> FP8. */
int wubu_specquant_route(const float *q, int dim, float mid, float high);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SPECQUANT_H */
