#ifndef WUBU_COND_MEM_H
#define WUBU_COND_MEM_H

/*
 * wubu_cond_mem.h -- EG-82: Engram-style conditional memory.
 * Normalized suffix n-grams are hashed with multiplicative-XOR into disjoint
 * prime-sized bucket ranges per head; tables are O(1) lookups with no FLOPs
 * over weights, and a gate fuses the retrieved vector into the hidden state.
 * The NFKC step is represented by a compact id normalization for this gate.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_COND_MAX_HEADS 8
#define WUBU_COND_MAX_N     16
#define WUBU_COND_MAX_DIM   64

typedef struct wubu_cond_mem wubu_cond_mem_t;

/* heads in [1,8], distinct primes, n in [1,16], dim in [1,64]. NULL on bad. */
wubu_cond_mem_t *wubu_cond_mem_create(int heads, const int *primes,
                                      int n, int dim);
void wubu_cond_mem_free(wubu_cond_mem_t *p);

/* Mock NFKC id compression: surface variants collapse to one canonical id. */
int wubu_cond_mem_normalize(int raw_id);

/* Absolute buckets, one per head, in disjoint prime-sized ranges. */
void wubu_cond_mem_hash(const wubu_cond_mem_t *p, const int *tokens,
                        int n, int *buckets);

int wubu_cond_mem_store(wubu_cond_mem_t *p, const int *tokens, int n,
                        const float *vec);
int wubu_cond_mem_lookup(const wubu_cond_mem_t *p, const int *tokens, int n,
                         float *out_vec);

/* out = hidden + gate * memory. 0/-1. */
int wubu_cond_mem_fuse(const wubu_cond_mem_t *p, const float *hidden,
                       const float *memory, float gate, float *out);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_COND_MEM_H */
