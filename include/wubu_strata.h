#ifndef WUBU_STRATA_H
#define WUBU_STRATA_H

/*
 * wubu_strata.h -- EG-68: Strata-style zero-allocation host, prefill launch
 * amortization and 4-way split-KV decode reduction, reimplemented in C11.
 * One gate: allocation count, launch count and reduction correctness.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned char *base;
    size_t cap, used;
    int allocations;
} wubu_strata_host_t;

/* One allocation up front; all later host storage is a bump pointer. */
int wubu_strata_host_init(wubu_strata_host_t *h, size_t cap);
void *wubu_strata_host_alloc(wubu_strata_host_t *h, size_t bytes);
void wubu_strata_host_free(wubu_strata_host_t *h);

/* Number of kernel launches to prefill n tokens with a given batch size. */
int wubu_strata_prefill_launches(int n_tokens, int batch, int *launches);

/* 4-way split-KV reduction: four partial reductions merged into out.
 * Returns the number of partial launches, or -1 on bad input. */
int wubu_strata_splitkv(const float *kv, int n, int dims, float *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_STRATA_H */
