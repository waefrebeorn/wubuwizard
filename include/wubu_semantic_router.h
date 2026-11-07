/* wubu_semantic_router.h — route a query to the KV region it attends to
 *
 * The KV cache is a file system; a query about a file should be routed to
 * the KV region(s) holding that file's knowledge. This router does that
 * from the model's OWN attention: given the query's attention map over the
 * context, it ranks the registered KV regions by how much attention the
 * query placed inside each region, and returns the top-K (research/055;
 * build-order module #11).
 *
 * The route is the "which region served this query" decision — the same
 * attention signal the coherence bridge and grow/shrink operators consume,
 * so routing, diagnosis, and pruning all speak one language.
 *
 * C11, opaque structs, no third-party deps.
 * Design: docs/wubu1-improvement-plan.md module #11 (AN22).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_SEMANTIC_ROUTER_H
#define WUBU_SEMANTIC_ROUTER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A candidate KV region in the context: its token window + optional label. */
typedef struct {
    size_t      start;   /* first context token of the region */
    size_t      len;     /* length in context tokens */
    const char *label;   /* optional /kv/in/... path (may be NULL) */
} wubu_route_region_t;

/* Configuration. */
typedef struct {
    size_t top_k;   /* max regions to return (route honors this) */
} wubu_semantic_router_cfg_t;

/* Opaque handle. */
typedef struct wubu_semantic_router wubu_semantic_router_t;

wubu_semantic_router_cfg_t wubu_semantic_router_default_cfg(void);

/* Create a semantic router. cfg may be NULL for defaults. */
wubu_semantic_router_t *wubu_semantic_router_create(
    const wubu_semantic_router_cfg_t *cfg);

/* ROUTE: rank the regions by the attention the query placed inside each.
 * attention_weights is [n_query_tokens][n_context_tokens] row-major.
 * Each region's mass = sum of attention, over all query rows, within its
 * context window. Regions with strictly positive mass are returned in
 * descending mass order into out_rank (region indices); out_labels (may be
 * NULL) gets the matching region->label. Returns the number of regions
 * returned (<= cfg.top_k). Returns -1 on invalid args / no attention. */
int wubu_semantic_router_route(wubu_semantic_router_t *r,
                               const float *attention_weights,
                               size_t n_query_tokens,
                               size_t n_context_tokens,
                               const wubu_route_region_t *regions,
                               size_t n_regions,
                               size_t *out_rank, const char **out_labels,
                               size_t max_out);

void wubu_semantic_router_free(wubu_semantic_router_t *r);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_SEMANTIC_ROUTER_H */
