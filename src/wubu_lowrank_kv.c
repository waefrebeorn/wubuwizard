/*
 * wubu_lowrank_kv.c -- EG-45a low-rank keys. Pure C11.
 */
#include "wubu_lowrank_kv.h"

#include <math.h>
#include <stdlib.h>

int wubu_lowrank_keys(const float *K, int rows, int dim, int rank,
                      int iters, float *U, float *V, float *out) {
    float *tmp;
    if (!K || !U || !V || !out || rows <= 0 || dim <= 0 ||
        rank <= 0 || rank > dim || rank > WUBU_LOWRANK_MAX_RANK ||
        iters <= 0) return -1;
    tmp = (float *)malloc((size_t)dim * sizeof(float));
    if (!tmp) return -1;

    /* Deterministic starting basis for the right singular subspace. */
    for (int r = 0; r < rank; r++)
        for (int j = 0; j < dim; j++)
            V[r*dim + j] = ((r + j) % 3 == 0) ? 1.0f : ((r + j) % 3 == 1 ? -0.5f : 0.25f);

    for (int it = 0; it < iters; it++) {
        for (int r = 0; r < rank; r++) {
            float norm;
            /* Apply K^T K to the r-th basis vector. */
            for (int j = 0; j < dim; j++) tmp[j] = 0.0f;
            for (int j = 0; j < dim; j++) {
                float s = 0.0f;
                for (int t = 0; t < dim; t++) {
                    float kkt = 0.0f;
                    for (int i = 0; i < rows; i++)
                        kkt += K[i*dim + t] * K[i*dim + j];
                    s += kkt * V[r*dim + t];
                }
                tmp[j] = s;
            }
            /* Gram-Schmidt against earlier vectors. */
            for (int p = 0; p < r; p++) {
                float dot = 0.0f;
                for (int j = 0; j < dim; j++) dot += tmp[j] * V[p*dim + j];
                for (int j = 0; j < dim; j++) tmp[j] -= dot * V[p*dim + j];
            }
            norm = 0.0f;
            for (int j = 0; j < dim; j++) norm += tmp[j] * tmp[j];
            norm = sqrtf(norm);
            if (norm < 1e-12f) norm = 1.0f;
            for (int j = 0; j < dim; j++) V[r*dim + j] = tmp[j] / norm;
        }
    }

    for (int i = 0; i < rows; i++)
        for (int r = 0; r < rank; r++) {
            float s = 0.0f;
            for (int j = 0; j < dim; j++) s += K[i*dim + j] * V[r*dim + j];
            U[i*rank + r] = s;
        }
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < dim; j++) {
            float s = 0.0f;
            for (int r = 0; r < rank; r++) s += U[i*rank + r] * V[r*dim + j];
            out[i*dim + j] = s;
        }
    free(tmp);
    return 0;
}

float wubu_lowrank_relerr(const float *a, const float *b, int n) {
    float num = 0.0f, den = 0.0f;
    if (!a || !b || n <= 0) return -1.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        num += d * d; den += a[i] * a[i];
    }
    return den > 0.0f ? sqrtf(num / den) : 0.0f;
}
