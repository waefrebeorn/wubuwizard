#ifndef WUBU_SPARSE_RETRIEVE_H
#define WUBU_SPARSE_RETRIEVE_H

/*
 * wubu_sparse_retrieve.h -- EG-75: training-free sparse-KV retrieval lifecycle.
 * Each token is its own page (page_size=1): representations are updated on
 * append and queried by top-k dot product, with the most recent pages always
 * retained. This gates the repr/update/query contract, not an accuracy claim.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_SPARSE_MAX_PAGES 128
#define WUBU_SPARSE_MAX_DIM   64

typedef struct wubu_sparse_retrieve wubu_sparse_retrieve_t;

wubu_sparse_retrieve_t *wubu_sparse_retrieve_create(int dim, int max_pages,
                                                    int recent_window);
void wubu_sparse_retrieve_free(wubu_sparse_retrieve_t *p);

int wubu_sparse_retrieve_update(wubu_sparse_retrieve_t *p,
                                const float *key, int dim);
int wubu_sparse_retrieve_query(const wubu_sparse_retrieve_t *p,
                               const float *q, int k,
                               int *out_pages, double *out_scores);
int wubu_sparse_retrieve_page_count(const wubu_sparse_retrieve_t *p);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_SPARSE_RETRIEVE_H */
