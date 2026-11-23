#ifndef WUBU_PROXY_ROUTE_H
#define WUBU_PROXY_ROUTE_H

/*
 * wubu_proxy_route.h -- EG-54: block-sparse proxy routes. Unselected KV blocks
 * contribute a precomputed mean-key summary instead of being dropped entirely,
 * reimplemented in C11. One gate: quality versus the dropped-block baseline at
 * the same number of fetched blocks.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Build one mean-key summary per block. out layout: n_blocks * dim. */
int wubu_proxy_summary(const float *keys, int n_blocks, int block_size, int dim,
                       float *out);

/* Attention over topk blocks plus optional proxy contributions from the rest.
 * proxy_weight 0 reproduces the dropped-block baseline; 1 uses full summaries.
 * Returns 0 or -1. */
int wubu_proxy_route(const float *keys, const float *summary,
                     int n_blocks, int block_size, int dim,
                     const float *q, int topk, float proxy_weight,
                     float *out, int *selected);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_PROXY_ROUTE_H */
