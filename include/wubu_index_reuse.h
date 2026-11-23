#ifndef WUBU_INDEX_REUSE_H
#define WUBU_INDEX_REUSE_H

/*
 * wubu_index_reuse.h -- EG-76: cross-layer DSA top-k index reuse.
 * A selector layer computes the top-k index set once; layers whose own top-k
 * matches reuse the same index-key KV allocation instead of a private copy.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_INDEX_MAX 64

typedef struct wubu_index_reuse wubu_index_reuse_t;

/* layers/top_k/dim/elem_bytes >= 1, top_k <= 64. NULL on bad input. */
wubu_index_reuse_t *wubu_index_reuse_create(int layers, int top_k,
                                            int dim, size_t elem_bytes);
void wubu_index_reuse_free(wubu_index_reuse_t *p);

/* Top-k indices by score descending (ties by ascending index). */
int wubu_index_reuse_compute(const wubu_index_reuse_t *p,
                             const double *scores, int n, int *out_idx);

/* Publish one shared index set from a layer; returns its KV byte cost. */
size_t wubu_index_reuse_publish(wubu_index_reuse_t *p, int layer,
                                const int *idx, int k);

/* Reuse the published set for another layer; 0 on reuse, -1 on mismatch. */
int wubu_index_reuse_reuse(wubu_index_reuse_t *p, int layer,
                           const int *idx, int k);

size_t wubu_index_reuse_bytes_allocated(const wubu_index_reuse_t *p);
size_t wubu_index_reuse_bytes_saved(const wubu_index_reuse_t *p);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_INDEX_REUSE_H */
