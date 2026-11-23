#ifndef WUBU_VARBIT_VQ_H
#define WUBU_VARBIT_VQ_H

/*
 * wubu_varbit_vq.h -- EG-45b: variable-bit packed VQ codes.
 * Each scalar in [-1,1] gets its own bit width (1..8); codes are packed
 * MSB-first into a byte stream and decoded with the same width table.
 * This gates the codec and its byte accounting, not a learned codebook.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_VQ_MAX_ITEMS 256

size_t wubu_varbit_vq_encoded_bytes(const int *bits, int n);

int wubu_varbit_vq_encode(const float *values, const int *bits, int n,
                          unsigned char *out);
int wubu_varbit_vq_decode(const unsigned char *in, const int *bits, int n,
                          float *out);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_VARBIT_VQ_H */
