/*
 * wubu_mx8.c -- EG-102 MX8 block microscaling. Pure C11.
 */
#include "wubu_mx8.h"

#include <math.h>

int wubu_mx8_encode(const float *x, int n,
                    wubu_mx8_block_t *blocks, int *n_blocks) {
    int nb;
    if (!x || !blocks || !n_blocks || n <= 0) return -1;
    nb = (n + WUBU_MX8_BLOCK - 1) / WUBU_MX8_BLOCK;
    *n_blocks = nb;

    for (int b = 0; b < nb; b++) {
        wubu_mx8_block_t *blk = &blocks[b];
        int count = n - b * WUBU_MX8_BLOCK;
        float mx = 0.0f, scale;
        int e, j;
        if (count > WUBU_MX8_BLOCK) count = WUBU_MX8_BLOCK;
        for (j = 0; j < count; j++) {
            float a = fabsf(x[b*WUBU_MX8_BLOCK + j]);
            if (a > mx) mx = a;
        }
        /* Largest magnitude code is 8; choose a power-of-two scale that fits. */
        e = mx > 0.0f ? (int)floorf(log2f(mx / 8.0f)) : 0;
        for (;;) {
            scale = ldexpf(1.0f, e);
            int fits = 1;
            for (j = 0; j < count; j++) {
                float q = fabsf(x[b*WUBU_MX8_BLOCK + j] / scale);
                if (q > 8.0f) { fits = 0; break; }
            }
            if (fits) break;
            e++;
            if (e > 60) return -1;
        }
        blk->exponent = (uint8_t)(e + WUBU_MX8_BIAS);
        for (j = 0; j < WUBU_MX8_BLOCK; j++) {
            int q;
            if (j >= count) { blk->codes[j] = 8; continue; }
            q = (int)roundf(x[b*WUBU_MX8_BLOCK + j] / scale) + 8;
            if (q < 0) q = 0;
            if (q > 15) q = 15;
            blk->codes[j] = (uint8_t)q;
        }
    }
    return 0;
}

int wubu_mx8_decode(const wubu_mx8_block_t *blocks, int n_blocks,
                    int n, float *out) {
    if (!blocks || !out || n_blocks <= 0 || n <= 0) return -1;
    for (int b = 0; b < n_blocks; b++) {
        float scale = ldexpf(1.0f, (int)blocks[b].exponent - WUBU_MX8_BIAS);
        for (int j = 0; j < WUBU_MX8_BLOCK; j++) {
            int idx = b * WUBU_MX8_BLOCK + j;
            if (idx >= n) return 0;
            out[idx] = (float)((int)blocks[b].codes[j] - 8) * scale;
        }
    }
    return 0;
}
