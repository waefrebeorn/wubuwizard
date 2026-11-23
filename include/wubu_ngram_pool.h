#ifndef WUBU_NGRAM_POOL_H
#define WUBU_NGRAM_POOL_H

/*
 * wubu_ngram_pool.h -- EG-71: llama.cpp-style rolling-hash n-gram pool.
 * A fixed-size table of n-gram -> next-token entries shared across all slots,
 * with an occupancy reset and a low-acceptance-streak reset. No draft model
 * and no draft KV: constant memory for any number of concurrent slots.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_NGRAM_MAX_N 16

typedef struct wubu_ngram_pool wubu_ngram_pool_t;

/* pool_entries >= 2, n in [1,16], occupancy in (0,1], streak >= 1. */
wubu_ngram_pool_t *wubu_ngram_pool_create(int pool_entries, int n,
                                          float occupancy, int streak);
void wubu_ngram_pool_free(wubu_ngram_pool_t *p);

/* Map the n-gram tokens[0..n-1] to the next token. 0/-1. */
int wubu_ngram_pool_add(wubu_ngram_pool_t *p, const int *tokens,
                        int n, int next_token);

/* Look up the next token; -1 when the n-gram is absent. */
int wubu_ngram_pool_get(const wubu_ngram_pool_t *p, const int *tokens, int n);

/* Record one draft outcome; a streak of rejections resets the shared pool. */
int wubu_ngram_pool_observe(wubu_ngram_pool_t *p, int accepted);

int wubu_ngram_pool_entries(const wubu_ngram_pool_t *p);
int wubu_ngram_pool_resets(const wubu_ngram_pool_t *p);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_NGRAM_POOL_H */
