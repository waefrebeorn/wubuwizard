/* wubu_cond_mem.c -- EG-82 Engram-style conditional memory. Strict C11. */
#include "wubu_cond_mem.h"

#include <stdlib.h>
#include <string.h>

struct wubu_cond_mem {
    int heads, n, dim, total;
    int primes[WUBU_COND_MAX_HEADS], offset[WUBU_COND_MAX_HEADS];
    float *table;
    unsigned char *used;
};

wubu_cond_mem_t *wubu_cond_mem_create(int heads, const int *primes,
                                      int n, int dim)
{
    wubu_cond_mem_t *p;
    if (!primes || heads < 1 || heads > WUBU_COND_MAX_HEADS ||
        n < 1 || n > WUBU_COND_MAX_N || dim < 1 || dim > WUBU_COND_MAX_DIM)
        return NULL;
    p = (wubu_cond_mem_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->heads = heads; p->n = n; p->dim = dim;
    for (int h = 0; h < heads; ++h) {
        if (primes[h] < 2) { free(p); return NULL; }
        for (int k = 0; k < h; ++k)
            if (primes[h] == primes[k]) { free(p); return NULL; }
        p->primes[h] = primes[h];
        p->offset[h] = h ? p->offset[h - 1] + p->primes[h - 1] : 0;
    }
    p->total = p->offset[heads - 1] + p->primes[heads - 1];
    p->table = (float *)calloc((size_t)p->total * (size_t)dim, sizeof(float));
    p->used = (unsigned char *)calloc((size_t)p->total, 1);
    if (!p->table || !p->used) {
        free(p->table); free(p->used); free(p);
        return NULL;
    }
    return p;
}

void wubu_cond_mem_free(wubu_cond_mem_t *p)
{
    if (!p) return;
    free(p->table); free(p->used); free(p);
}

/* Stand-in for NFKC/accent/case folding: collapse surface variants by id. */
int wubu_cond_mem_normalize(int raw_id)
{
    return raw_id < 0 ? -1 : (raw_id & 0xff);
}

void wubu_cond_mem_hash(const wubu_cond_mem_t *p, const int *tokens,
                        int n, int *buckets)
{
    for (int h = 0; h < p->heads; ++h) {
        uint64_t x = 0x9e3779b97f4a7c15ULL ^ (uint64_t)(h + 1) * 0xbf58476d1ce4e5b9ULL;
        for (int i = 0; i < n; ++i) {
            int v = wubu_cond_mem_normalize(tokens[i]);
            x ^= (uint64_t)(v < 0 ? 0 : v) + 0x9e3779b9ULL + (x << 6) + (x >> 2);
            x *= 0xbf58476d1ce4e5b9ULL;
        }
        buckets[h] = p->offset[h] + (int)(x % (uint64_t)p->primes[h]);
    }
}

int wubu_cond_mem_store(wubu_cond_mem_t *p, const int *tokens, int n,
                        const float *vec)
{
    int buckets[WUBU_COND_MAX_HEADS];
    if (!p || !tokens || !vec || n != p->n) return -1;
    wubu_cond_mem_hash(p, tokens, n, buckets);
    for (int h = 0; h < p->heads; ++h) {
        memcpy(p->table + (size_t)buckets[h] * p->dim, vec,
               (size_t)p->dim * sizeof(float));
        p->used[buckets[h]] = 1;
    }
    return 0;
}

int wubu_cond_mem_lookup(const wubu_cond_mem_t *p, const int *tokens, int n,
                         float *out_vec)
{
    int buckets[WUBU_COND_MAX_HEADS];
    if (!p || !tokens || !out_vec || n != p->n) return -1;
    wubu_cond_mem_hash(p, tokens, n, buckets);
    if (!p->used[buckets[0]]) return 0;
    memcpy(out_vec, p->table + (size_t)buckets[0] * p->dim,
           (size_t)p->dim * sizeof(float));
    return 1;
}

int wubu_cond_mem_fuse(const wubu_cond_mem_t *p, const float *hidden,
                       const float *memory, float gate, float *out)
{
    if (!p || !hidden || !memory || !out) return -1;
    for (int i = 0; i < p->dim; ++i)
        out[i] = hidden[i] + gate * memory[i];
    return 0;
}
