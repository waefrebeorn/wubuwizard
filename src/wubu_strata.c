/*
 * wubu_strata.c -- EG-68 Strata-style host and reduction paths. Pure C11.
 */
#include "wubu_strata.h"

#include <stdlib.h>

#define STRATA_SPLITS 4
#define STRATA_ALIGN 16u

int wubu_strata_host_init(wubu_strata_host_t *h, size_t cap) {
    if (!h || cap == 0) return -1;
    h->base = (unsigned char *)malloc(cap);
    if (!h->base) return -1;
    h->cap = cap;
    h->used = 0;
    h->allocations = 1;
    return 0;
}

void *wubu_strata_host_alloc(wubu_strata_host_t *h, size_t bytes) {
    size_t off, pad;
    void *p;
    if (!h || !h->base || bytes == 0 || bytes > h->cap - h->used) return NULL;
    /* SIMD host storage needs 16-byte alignment even for mixed-size requests. */
    off = (h->used + STRATA_ALIGN - 1) & ~(size_t)(STRATA_ALIGN - 1);
    pad = off - h->used;
    if (pad > h->cap - h->used || bytes > h->cap - off) return NULL;
    p = h->base + off;
    h->used = off + bytes;
    return p;
}

void wubu_strata_host_free(wubu_strata_host_t *h) {
    if (!h || !h->base) return;
    free(h->base);
    h->base = NULL;
    h->cap = h->used = 0;
}

int wubu_strata_prefill_launches(int n_tokens, int batch, int *launches) {
    if (!launches || n_tokens <= 0 || batch <= 0) return -1;
    *launches = (n_tokens + batch - 1) / batch;
    return 0;
}

int wubu_strata_splitkv(const float *kv, int n, int dims, float *out) {
    float partial[STRATA_SPLITS][64];
    if (!kv || !out || n <= 0 || dims <= 0 || dims > 64) return -1;
    for (int s = 0; s < STRATA_SPLITS; s++)
        for (int d = 0; d < dims; d++) partial[s][d] = 0.0f;
    for (int s = 0; s < STRATA_SPLITS; s++) {
        int lo = s * n / STRATA_SPLITS;
        int hi = (s + 1) * n / STRATA_SPLITS;
        for (int i = lo; i < hi; i++)
            for (int d = 0; d < dims; d++)
                partial[s][d] += kv[(size_t)i * dims + d];
    }
    for (int d = 0; d < dims; d++) {
        float v = 0.0f;
        for (int s = 0; s < STRATA_SPLITS; s++) v += partial[s][d];
        out[d] = v;
    }
    return STRATA_SPLITS;
}
