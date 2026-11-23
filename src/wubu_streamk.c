/* wubu_streamk.c -- EG-79 Stream-K partition/fixup gate. Strict C11. */
#include "wubu_streamk.h"

#include <stdlib.h>

int wubu_streamk_partition(int m, int blocks, int *lo, int *hi)
{
    if (!lo || !hi || m < 1 || m > WUBU_STREAMK_MAX_M ||
        blocks < 1 || blocks > m || blocks > WUBU_STREAMK_MAX_BLOCKS) return -1;
    for (int b = 0; b < blocks; ++b) {
        lo[b] = b * m / blocks;
        hi[b] = (b + 1) * m / blocks;
    }
    return 0;
}

int wubu_streamk_matmul(const float *A, const float *B,
                        int m, int n, int k, int blocks, float *out)
{
    float *partial;
    int lo[WUBU_STREAMK_MAX_BLOCKS], hi[WUBU_STREAMK_MAX_BLOCKS];
    if (!A || !B || !out || m < 1 || n < 1 || k < 1 ||
        m > WUBU_STREAMK_MAX_M || n > WUBU_STREAMK_MAX_M ||
        k > WUBU_STREAMK_MAX_M ||
        wubu_streamk_partition(m, blocks, lo, hi) != 0) return -1;
    partial = (float *)calloc((size_t)blocks * m * n, sizeof(float));
    if (!partial) return -1;
    for (int b = 0; b < blocks; ++b) {
        for (int i = lo[b]; i < hi[b]; ++i) {
            for (int j = 0; j < n; ++j) {
                double v = 0.0;
                for (int p = 0; p < k; ++p)
                    v += (double)A[(size_t)i * k + p] * (double)B[(size_t)p * n + j];
                partial[((size_t)b * m + i) * n + j] = (float)v;
            }
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float v = 0.0f;
            for (int b = 0; b < blocks; ++b)
                v += partial[((size_t)b * m + i) * n + j];
            out[(size_t)i * n + j] = v;
        }
    }
    free(partial);
    return 0;
}

int wubu_streamk_fixups(const float *out, int m, int n, int blocks)
{
    if (!out || m < 1 || n < 1 || m > WUBU_STREAMK_MAX_M ||
        n > WUBU_STREAMK_MAX_M || blocks < 1 ||
        blocks > WUBU_STREAMK_MAX_BLOCKS) return 0;
    return blocks;
}
