/* test_eg44_indexer.c -- EG-44 gate: sparse indexer accuracy before speed. */
#include <stdio.h>

#include "wubu_sparse_indexer.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define N 32
#define D 8
#define K 8

int main(void) {
    wubu_sparse_indexer_t ix;
    float q[D], keys[N*D];
    int approx[K], exact[K], overlap = 0, i, j;
    unsigned state = 7u;

    for (j = 0; j < D; j++) {
        state = state * 1664525u + 1013904223u;
        q[j] = ((float)((state >> 8) & 0xFFFFFFu) / 8388608.0f) - 1.0f;
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < D; j++) {
            state = state * 1664525u + 1013904223u;
            float r = ((float)((state >> 8) & 0xFFFFFFu) / 8388608.0f) - 1.0f;
            /* Dominant low-rank structure plus a small residual. */
            keys[i*D + j] = j < 4 ? r * 2.0f : r * 0.1f;
        }
    }

    CHECK(wubu_sparse_indexer_init(&ix, D, 4, 12345u) == 0);
    CHECK(wubu_sparse_indexer_select(&ix, q, keys, N, K, approx) == 0);
    CHECK(wubu_sparse_indexer_exact(q, keys, N, D, K, exact) == 0);

    for (i = 0; i < K; i++)
        for (j = 0; j < K; j++)
            if (approx[i] == exact[j]) { overlap++; break; }
    printf("EG-44 top-%d overlap=%d/%d sparsity=%.2f\n",
           K, overlap, K, (float)K / (float)N);
    CHECK(overlap >= 6);
    CHECK(K < N);

    CHECK(wubu_sparse_indexer_init(&ix, 0, 4, 1u) == -1);
    CHECK(wubu_sparse_indexer_init(&ix, 4, 8, 1u) == -1);
    CHECK(wubu_sparse_indexer_select(NULL, q, keys, N, K, approx) == -1);
    CHECK(wubu_sparse_indexer_exact(q, keys, N, D, 0, exact) == -1);
    CHECK(wubu_sparse_indexer_exact(q, keys, N, D, N + 1, exact) == -1);

    if (failures) {
        fprintf(stderr, "test_eg44_indexer: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg44_indexer: PASS\n");
    return 0;
}
