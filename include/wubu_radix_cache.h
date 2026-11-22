#ifndef WUBU_RADIX_CACHE_H
#define WUBU_RADIX_CACHE_H

/*
 * wubu_radix_cache.h -- EG-08: RadixAttention-style prefix tree (SGLang
 * lineage), reimplemented in C11. Unlike the existing hash-prefix cache,
 * this shares common token prefixes across sibling continuations and
 * evicts least-recently-used subtrees under a node budget.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_radix_node wubu_radix_node_t;
typedef struct wubu_radix_cache wubu_radix_cache_t;

wubu_radix_cache_t *wubu_radix_cache_create(int max_nodes);
void wubu_radix_cache_free(wubu_radix_cache_t *c);

/* Longest cached prefix length for tokens[0..n). Touches visited nodes. */
int wubu_radix_cache_match(wubu_radix_cache_t *c, const int *tokens, int n,
                           const wubu_radix_node_t **node_out);

/* Insert a token sequence associated with block_id. 0 on success. */
int wubu_radix_cache_insert(wubu_radix_cache_t *c, const int *tokens, int n,
                            int block_id);

/* Evict up to `count` least-recently-used leaves (root is never evicted).
 * Returns the number of nodes actually removed. */
int wubu_radix_cache_evict_lru(wubu_radix_cache_t *c, int count);

int wubu_radix_cache_node_count(const wubu_radix_cache_t *c);
int wubu_radix_cache_child_count(const wubu_radix_node_t *node);
int wubu_radix_cache_block(const wubu_radix_node_t *node);
uint64_t wubu_radix_cache_hits(const wubu_radix_cache_t *c);
uint64_t wubu_radix_cache_misses(const wubu_radix_cache_t *c);
uint64_t wubu_radix_cache_evictions(const wubu_radix_cache_t *c);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_RADIX_CACHE_H */
