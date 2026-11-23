/*
 * wubu_sparse_indexer.c -- EG-44 indexer policy. Pure C11.
 */
#include "wubu_sparse_indexer.h"

#include <math.h>
#include <stdlib.h>

int wubu_sparse_indexer_init(wubu_sparse_indexer_t *ix, int dim, int rank,
                             unsigned seed) {
    unsigned state = seed ? seed : 1u;
    if (!ix || dim <= 0 || rank <= 0 || rank > dim ||
        dim > WUBU_INDEXER_MAX_RANK) return -1;
    ix->dim = dim;
    ix->rank = rank;
    for (int r = 0; r < rank; r++)
        for (int j = 0; j < dim; j++)
            ix->proj[r*dim + j] = (j == r) ? 1.0f : 0.0f;
    (void)state;
    return 0;
}

static void push_top(const float *scores, int n, int k, int *out) {
    for (int i = 0; i < n; i++) {
        if (k <= 0) break;
        int j = k - 1;
        while (j >= 0 && scores[out[j]] < scores[i]) j--;
        if (j >= k - 1) continue;
        for (int t = k - 1; t > j + 1; t--) out[t] = out[t - 1];
        if (j + 1 < k) out[j + 1] = i;
    }
}

int wubu_sparse_indexer_select(const wubu_sparse_indexer_t *ix,
                               const float *q, const float *keys, int n,
                               int k, int *out) {
    float qp[WUBU_INDEXER_MAX_RANK], *scores;
    if (!ix || !q || !keys || !out || n <= 0 || k <= 0 || k > n) return -1;
    scores = (float *)malloc((size_t)n * sizeof(float));
    if (!scores) return -1;
    for (int r = 0; r < ix->rank; r++) {
        float s = 0.0f;
        for (int j = 0; j < ix->dim; j++) s += ix->proj[r*ix->dim + j] * q[j];
        qp[r] = s;
    }
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        for (int r = 0; r < ix->rank; r++) {
            float kp = 0.0f;
            for (int j = 0; j < ix->dim; j++)
                kp += ix->proj[r*ix->dim + j] * keys[i*ix->dim + j];
            s += qp[r] * kp;
        }
        scores[i] = s;
    }
    for (int i = 0; i < k; i++) out[i] = i;
    push_top(scores, n, k, out);
    free(scores);
    return 0;
}

int wubu_sparse_indexer_exact(const float *q, const float *keys, int n,
                              int dim, int k, int *out) {
    float *scores;
    if (!q || !keys || !out || n <= 0 || dim <= 0 || k <= 0 || k > n) return -1;
    scores = (float *)malloc((size_t)n * sizeof(float));
    if (!scores) return -1;
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        for (int j = 0; j < dim; j++) s += q[j] * keys[i*dim + j];
        scores[i] = s;
    }
    for (int i = 0; i < k; i++) out[i] = i;
    push_top(scores, n, k, out);
    free(scores);
    return 0;
}
