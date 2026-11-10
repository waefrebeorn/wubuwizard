/*
 * wubu_ctxvm.h -- AGI-OS context virtual-memory hierarchy (AF08-AF10).
 *
 * The context ring is HIVE-backed (wubu_hive): linked fixed blocks with a
 * skipfield + freelist, instead of a dense FIFO array.  The hive gives
 * stable slot pointers, O(1) erase, cache-friendly iteration, and -- the
 * reason it replaces FIFO here -- eviction by IMPORTANCE rather than pure
 * age: the ring can keep a high-value token resident while an older,
 * low-value one is dropped.
 */
#ifndef WUBU_CTXVM_H
#define WUBU_CTXVM_H

#include <stddef.h>
#include <stdint.h>
#include "wubu_hive.h"

#define WUBU_CTX_L1 1  /* gen window */
#define WUBU_CTX_L2 2  /* session */
#define WUBU_CTX_L3 3  /* long-term */
#define WUBU_CTX_L4 4  /* cross-session */

/* one resident context token (ring-owned, heap) */
typedef struct {
    long    tok;
    uint64_t seq;        /* insertion order (monotonic) */
    float   importance;  /* 0..1; from wubu_ctx_tier */
    long    ttl;
} wubu_ctx_slot_t;

typedef struct {
    wubu_hive_t hive;    /* holds wubu_ctx_slot_t* -- THE HIVE */
    size_t capacity;     /* max live slots */
    uint64_t seq;        /* monotonic insertion counter */
    size_t size;         /* live count (== hive.total_live) */
} wubu_ctxring_t;

int  wubu_ctx_tier(float importance, long ttl);                 /* AF08 */

/* ring lifecycle (hive-backed) */
int  wubu_ctxring_init(wubu_ctxring_t *r, size_t capacity);
void wubu_ctxring_free(wubu_ctxring_t *r);

/* AF09: insert a token; on overflow evict the lowest-importance slot
 * (oldest wins ties) -- the hive-backed replacement for the FIFO ring.
 * Returns 1 if an eviction occurred this call, else 0. */
int  wubu_ctx_evict_fifo(wubu_ctxring_t *r, long tok);
int  wubu_ctx_resident(const wubu_ctxring_t *r, long tok, int ws); /* AF09 */

/* set a resident slot's importance + ttl (drives hive eviction: the
 * lowest-importance slot is dropped on overflow, oldest wins ties) */
int  wubu_ctx_importance(wubu_ctxring_t *r, long tok, float importance, long ttl);

float wubu_cosine(const float *a, const float *b, int n);      /* AF10 */
int  wubu_sem_cache_hit(const float *q, const float *c, int n, float thr); /* AF10 */

#endif
