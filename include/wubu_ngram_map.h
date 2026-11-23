#ifndef WUBU_NGRAM_MAP_H
#define WUBU_NGRAM_MAP_H

/*
 * wubu_ngram_map.h -- EG-72: llama.cpp n-gram-map k4v.
 * Each n-gram maps to up to four next-token values with hit counts; the least
 * hit (oldest on ties) is evicted. The map has a byte-exact persisted form so
 * it can be merged across sessions.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_NGRAM_K4V_MAX 4

typedef struct wubu_ngram_map wubu_ngram_map_t;

/* capacity >= 1, n in [1,16]. NULL on bad input. */
wubu_ngram_map_t *wubu_ngram_map_create(int capacity, int n);
void wubu_ngram_map_free(wubu_ngram_map_t *p);

/* Add one observed continuation; increments its hit count when present. */
int wubu_ngram_map_observe(wubu_ngram_map_t *p, const int *tokens, int n,
                           int next_token);

/* Copy values sorted by count desc (ties by insertion order). Returns 0..4,
 * or -1 on bad input. */
int wubu_ngram_map_values(const wubu_ngram_map_t *p, const int *tokens, int n,
                          int *out_values, int *out_counts);

size_t wubu_ngram_map_encoded_size(const wubu_ngram_map_t *p);
int wubu_ngram_map_encode(const wubu_ngram_map_t *p,
                          unsigned char *buf, size_t cap);
int wubu_ngram_map_decode(const unsigned char *buf, size_t bytes,
                          wubu_ngram_map_t **out);

/* Merge another map's hit counts into this one. 0/-1. */
int wubu_ngram_map_merge(wubu_ngram_map_t *p, const wubu_ngram_map_t *other);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_NGRAM_MAP_H */
