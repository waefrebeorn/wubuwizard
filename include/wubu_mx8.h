#ifndef WUBU_MX8_H
#define WUBU_MX8_H

/*
 * wubu_mx8.h -- EG-102: MX-style block microscaling (Strata SF4+MX8 SQ-20
 * lineage). One E8M0 power-of-two exponent shared by a block of 4-bit signed
 * codes. Pure C11; no third-party code.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_MX8_BLOCK 32
#define WUBU_MX8_BIAS 127

typedef struct {
    uint8_t exponent;                       /* E8M0: scale = 2^(e - 127) */
    uint8_t codes[WUBU_MX8_BLOCK];          /* signed 4-bit, zero point 8 */
} wubu_mx8_block_t;

/* Encode n floats into ceil(n/32) blocks (tail zero-padded). Returns 0 or -1. */
int wubu_mx8_encode(const float *x, int n,
                    wubu_mx8_block_t *blocks, int *n_blocks);

/* Decode n floats from blocks. Returns 0 or -1. */
int wubu_mx8_decode(const wubu_mx8_block_t *blocks, int n_blocks,
                    int n, float *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MX8_H */
