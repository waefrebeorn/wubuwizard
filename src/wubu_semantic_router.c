/* wubu_semantic_router.c — route a query to the KV region it attends to
 *
 * Ranks registered KV regions by the total attention the query placed
 * inside each region's context window (research/055; module #11).
 *
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_semantic_router.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t region_index;
    double mass;
} route_entry_t;

struct wubu_semantic_router {
    size_t top_k;
};

wubu_semantic_router_cfg_t wubu_semantic_router_default_cfg(void) {
    wubu_semantic_router_cfg_t c;
    c.top_k = 4;
    return c;
}

wubu_semantic_router_t *wubu_semantic_router_create(
    const wubu_semantic_router_cfg_t *cfg) {
    wubu_semantic_router_t *r =
        (wubu_semantic_router_t *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->top_k = cfg ? (cfg->top_k ? cfg->top_k : 1) : 4;
    return r;
}

static int cmp_desc(const void *a, const void *b) {
    const route_entry_t *x = (const route_entry_t *)a;
    const route_entry_t *y = (const route_entry_t *)b;
    if (x->mass < y->mass) return 1;
    if (x->mass > y->mass) return -1;
    return 0;
}

int wubu_semantic_router_route(wubu_semantic_router_t *r,
                               const float *attention_weights,
                               size_t n_query_tokens,
                               size_t n_context_tokens,
                               const wubu_route_region_t *regions,
                               size_t n_regions,
                               size_t *out_rank, const char **out_labels,
                               size_t max_out) {
    if (!r || !attention_weights || !regions || !out_rank ||
        n_query_tokens == 0 || n_context_tokens == 0 || n_regions == 0)
        return -1;

    /* Accumulate per-region attention mass. */
    route_entry_t *entries =
        (route_entry_t *)calloc(n_regions ? n_regions : 1, sizeof(*entries));
    if (!entries) return -1;
    for (size_t i = 0; i < n_regions; i++) {
        entries[i].region_index = i;
        entries[i].mass = 0.0;
        for (size_t q = 0; q < n_query_tokens; q++) {
            const float *row = attention_weights + q * n_context_tokens;
            size_t s = regions[i].start;
            size_t e = (s + regions[i].len > n_context_tokens)
                           ? n_context_tokens : s + regions[i].len;
            for (size_t t = s; t < e; t++) entries[i].mass += (double)row[t];
        }
    }

    /* Only regions with positive attention are routable. */
    route_entry_t *cand = entries;
    size_t n_cand = 0;
    for (size_t i = 0; i < n_regions; i++)
        if (entries[i].mass > 0.0) cand[n_cand++] = entries[i];

    if (n_cand == 0) { free(entries); return 0; }
    qsort(cand, n_cand, sizeof(*cand), cmp_desc);

    size_t take = n_cand < r->top_k ? n_cand : r->top_k;
    if (take > max_out) take = max_out;
    for (size_t i = 0; i < take; i++) {
        out_rank[i] = cand[i].region_index;
        if (out_labels)
            out_labels[i] = regions[cand[i].region_index].label;
    }

    free(entries);
    return (int)take;
}

void wubu_semantic_router_free(wubu_semantic_router_t *r) {
    free(r);
}
