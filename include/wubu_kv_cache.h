#ifndef WUBU_KV_CACHE_H
#define WUBU_KV_CACHE_H

#include <stdint.h>
#include <stddef.h>

/* Opaque KV cache handle */
typedef struct wubu_kv_cache wubu_kv_cache_t;

/* Create a KV cache for up to max_sequences concurrent sequences */
wubu_kv_cache_t *wubu_kv_create(int max_sequences);

/* Destroy cache and free all pages */
void wubu_kv_destroy(wubu_kv_cache_t *cache);

/* Append n_tokens of K/V data for sequence seq_id */
int wubu_kv_append(wubu_kv_cache_t *cache, uint32_t seq_id,
                   const void *k_data, const void *v_data, int n_tokens);

/* Read KV data for token_idx of sequence seq_id */
int wubu_kv_read(const wubu_kv_cache_t *cache, uint32_t seq_id,
                 int token_idx, void *k_out, void *v_out);

/* Free all pages for a sequence */
void wubu_kv_free_sequence(wubu_kv_cache_t *cache, uint32_t seq_id);

/* Get cache statistics */
void wubu_kv_stats(const wubu_kv_cache_t *cache, int *pages_used, int *pages_free,
                    size_t *bytes_used, size_t *bytes_allocated);

#endif /* WUBU_KV_CACHE_H */
