#ifndef WUBU_HICACHE_H
#define WUBU_HICACHE_H

/*
 * wubu_hicache.h -- EG-18: three-tier KV cache with asynchronous promotion
 * and write-back (Mooncake HiCache / SGLang lineage), reimplemented in C11.
 *
 * L1 is the fast tier, L2 host, L3 SSD. Entries are fixed-size so capacity is
 * expressed in slots. Eviction from a tier is LRU; a demotion from L1 is
 * queued as an asynchronous write-back and completes on tick(). prefetch()
 * schedules a promotion only when its ready time fits the caller's deadline.
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_HICACHE_ENTRY_BYTES 16

typedef struct wubu_hicache wubu_hicache_t;

typedef struct {
    int hits[3];       /* hits in L1/L2/L3 (index 0..2) */
    int misses;
    int pending;       /* transfers not yet applied */
    int l1, l2, l3;    /* entries currently resident per tier */
    int moves;         /* promotion/demotion count */
} wubu_hicache_stats_t;

wubu_hicache_t *wubu_hicache_create(int n1, int n2, int n3);
void wubu_hicache_free(wubu_hicache_t *h);

/* Insert/replace a 16-byte value; lands in L1. Returns 0 or -1. */
int wubu_hicache_put(wubu_hicache_t *h, uint64_t key, const void *data);

/* Fetch a value. Returns tier number 1..3, or -1 on miss / bad argument.
 * L2/L3 hits are promoted immediately. */
int wubu_hicache_get(wubu_hicache_t *h, uint64_t key, void *out);

/* Schedule promotion of a key found in L2/L3 into L1. Returns 1 scheduled,
 * 0 if the deadline cannot be met or the key is already in L1, -1 if absent.
 * Transfers complete when tick() reaches ready_ms. */
int wubu_hicache_prefetch(wubu_hicache_t *h, uint64_t key,
                          unsigned long long deadline_ms);

/* Advance the clock to now_ms (monotonic) and apply ready transfers.
 * Returns the number of transfers completed. */
int wubu_hicache_tick(wubu_hicache_t *h, unsigned long long now_ms);

void wubu_hicache_stats(const wubu_hicache_t *h, wubu_hicache_stats_t *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_HICACHE_H */
