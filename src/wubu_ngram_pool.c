/* wubu_ngram_pool.c -- EG-71 shared rolling-hash n-gram pool. Strict C11. */
#include "wubu_ngram_pool.h"

#include <stdlib.h>

struct wubu_ngram_pool {
    int pool, n, entries, resets, streak, streak_limit;
    float occupancy;
    uint64_t *keys;
    int *values;
    unsigned char *used;
};

static uint64_t ngram_hash(const int *tokens, int n)
{
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; ++i) {
        uint64_t v = (uint64_t)(unsigned)tokens[i];
        h ^= v; h *= 1099511628211ULL;
        h ^= v >> 32; h *= 1099511628211ULL;
    }
    return h;
}

wubu_ngram_pool_t *wubu_ngram_pool_create(int pool_entries, int n,
                                          float occupancy, int streak)
{
    wubu_ngram_pool_t *p;
    if (pool_entries < 2 || n < 1 || n > WUBU_NGRAM_MAX_N ||
        !(occupancy > 0.0f && occupancy <= 1.0f) || streak < 1) return NULL;
    p = (wubu_ngram_pool_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->keys = (uint64_t *)calloc((size_t)pool_entries, sizeof(*p->keys));
    p->values = (int *)calloc((size_t)pool_entries, sizeof(*p->values));
    p->used = (unsigned char *)calloc((size_t)pool_entries, 1);
    if (!p->keys || !p->values || !p->used) {
        free(p->keys); free(p->values); free(p->used); free(p);
        return NULL;
    }
    p->pool = pool_entries;
    p->n = n;
    p->occupancy = occupancy;
    p->streak_limit = streak;
    return p;
}

void wubu_ngram_pool_free(wubu_ngram_pool_t *p)
{
    if (!p) return;
    free(p->keys); free(p->values); free(p->used); free(p);
}

static void clear_pool(wubu_ngram_pool_t *p)
{
    for (int i = 0; i < p->pool; ++i) p->used[i] = 0;
    p->entries = 0;
    ++p->resets;
}

int wubu_ngram_pool_add(wubu_ngram_pool_t *p, const int *tokens,
                        int n, int next_token)
{
    uint64_t key;
    size_t at;
    if (!p || !tokens || n != p->n || next_token < 0) return -1;
    if ((float)(p->entries + 1) > (float)p->pool * p->occupancy) clear_pool(p);
    key = ngram_hash(tokens, n);
    at = (size_t)(key % (uint64_t)p->pool);
    for (int probe = 0; probe < p->pool; ++probe) {
        size_t i = (at + (size_t)probe) % (size_t)p->pool;
        if (!p->used[i] || p->keys[i] == key) {
            if (!p->used[i]) ++p->entries;
            p->used[i] = 1;
            p->keys[i] = key;
            p->values[i] = next_token;
            return 0;
        }
    }
    p->values[at] = next_token;      /* full table: overwrite the home slot */
    p->keys[at] = key;
    return 0;
}

int wubu_ngram_pool_get(const wubu_ngram_pool_t *p, const int *tokens, int n)
{
    uint64_t key;
    size_t at;
    if (!p || !tokens || n != p->n) return -1;
    key = ngram_hash(tokens, n);
    at = (size_t)(key % (uint64_t)p->pool);
    for (int probe = 0; probe < p->pool; ++probe) {
        size_t i = (at + (size_t)probe) % (size_t)p->pool;
        if (!p->used[i]) return -1;
        if (p->keys[i] == key) return p->values[i];
    }
    return -1;
}

int wubu_ngram_pool_entries(const wubu_ngram_pool_t *p)
{
    return p ? p->entries : -1;
}

int wubu_ngram_pool_observe(wubu_ngram_pool_t *p, int accepted)
{
    if (!p || (accepted != 0 && accepted != 1)) return -1;
    if (accepted) {
        p->streak = 0;
    } else if (++p->streak >= p->streak_limit) {
        p->streak = 0;
        clear_pool(p);
    }
    return 0;
}

int wubu_ngram_pool_resets(const wubu_ngram_pool_t *p)
{
    return p ? p->resets : -1;
}
