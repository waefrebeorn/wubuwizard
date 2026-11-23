#ifndef WUBU_DYNACT_H
#define WUBU_DYNACT_H

/*
 * wubu_dynact.h -- EG-26: per-token dynamic activation quantization
 * (W4A16 / FP8 lineage), reimplemented in portable C11.
 *
 * Each row (token) gets its own symmetric scale derived from its own maximum
 * absolute value, so a quiet token is not crushed by a loud token's scale.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Quantize one row to signed `bits`-bit integers. Returns 0 or -1. */
int wubu_dynact_quant_row(const float *x, int n, int bits,
                          unsigned char *out, float *scale);

/* Dequantize a row produced above. Returns 0 or -1. */
int wubu_dynact_dequant_row(const unsigned char *q, int n, int bits,
                            float scale, float *out);

/* Max absolute error between two float rows. */
float wubu_dynact_maxerr(const float *a, const float *b, int n);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_DYNACT_H */
