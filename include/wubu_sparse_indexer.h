#ifndef WUBU_SPARSE_INDEXER_H
#define WUBU_SPARSE_INDEXER_H

/*
 * wubu_sparse_indexer.h -- EG-44: trained-sparse-attention indexer policy
 * (NVIDIA sparse-attention lineage). A low-rank projection scores keys before
 * exact attention; the gate measures top-k recall against the exact selector
 * so no speed claim is made without an accuracy check. Pure C11.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_INDEXER_MAX_RANK 64

typedef struct {
    int rank;
    int dim;
    float proj[WUBU_INDEXER_MAX_RANK * WUBU_INDEXER_MAX_RANK];
} wubu_sparse_indexer_t;

/* Deterministic untrained projection over the first `rank` channels; the seed
 * is reserved for a future learned projection. rank <= dim <= 64. */
int wubu_sparse_indexer_init(wubu_sparse_indexer_t *ix, int dim, int rank,
                             unsigned seed);

/* Select k key indices by projected dot product. Returns 0 or -1. */
int wubu_sparse_indexer_select(const wubu_sparse_indexer_t *ix,
                               const float *q, const float *keys, int n,
                               int k, int *out);

/* Exact top-k selector used by the accuracy gate. */
int wubu_sparse_indexer_exact(const float *q, const float *keys, int n,
                              int dim, int k, int *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SPARSE_INDEXER_H */
