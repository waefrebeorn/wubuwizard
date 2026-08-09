/*
 * wubu_hive.h -- THE HIVE: the AGI's memory structure (WuBu).
 *
 * The user's diagram: Vector = contiguous array (fast but reallocation
 * moves everything); List = scattered nodes (stable pointers but cache
 * misses); Hive = linked FIXED BLOCKS + skipfield + freelist.
 *
 *   struct block {
 *       void **slots;      // fixed block of pointers (cache-friendly)
 *       uint8_t *skip;     // skipfield: 1 = erased, 0 = live
 *       size_t live, cap;  // live count + block capacity
 *       struct block *next;
 *   };
 *
 * Why the hive beats both:
 *   - cache: blocks are contiguous pointer arrays (vector-like locality)
 *   - stable ptrs: the slots arrays never move; the values they point
 *     to are caller-owned and stable (list-like)
 *   - fast erase: mark the skip bit + push the slot to the freelist --
 *     O(1), no compaction, no shifting
 *   - fast insert: reuse a freelist slot or allocate a new block --
 *     O(1) amortized, no full reallocation
 *   - fast iterate: jump the skipfield (skip erased slots in one read)
 *
 * Pure C11, no templates, no third-party. The hive is the memory of
 * the WuBu AGI: tokens, routing history, KV lives, context slots --
 * whatever needs stable pointers with cache-friendly iteration and
 * O(1) erase/insert.
 */
#ifndef WUBU_HIVE_H
#define WUBU_HIVE_H

#include <stddef.h>
#include <stdint.h>

/* The fixed block size: 64 slots per block (a cache-line-friendly
 * choice; 64 void* = 512 bytes). */
#define WUBU_HIVE_BLOCK_CAP 64

typedef struct wubu_hive_block {
    void **slots;                 /* [cap] caller-owned pointers */
    uint8_t *skip;                /* [cap] 1 = erased, 0 = live */
    size_t live;                  /* live slots in this block */
    size_t cap;                   /* fixed at WUBU_HIVE_BLOCK_CAP */
    struct wubu_hive_block *next; /* the chain */
} wubu_hive_block_t;

/* one freelist entry: a reusable (block, slot) pair */
struct wubu_hive_free_entry {
    wubu_hive_block_t *block;
    size_t slot;
};

typedef struct {
    wubu_hive_block_t *head;      /* first block */
    wubu_hive_block_t *tail;      /* last block (append) */
    size_t n_blocks;
    size_t total_live;
    /* the freelist: a LIFO stack of (block, slot) entries. Erase pushes,
     * insert pops -- every erased slot is reusable, O(1). */
    struct wubu_hive_free_entry *free_entries;
    size_t n_free, free_cap;
    /* stats */
    size_t allocs;                /* slots allocated */
    size_t reuses;                /* slots reused from the freelist */
} wubu_hive_t;

/* H1: init an empty hive. */
int wubu_hive_init(wubu_hive_t *h);

/* H2: insert a pointer. Returns 0 on success (slot added). The slot
 * reuses a freelist entry when available, else a new block. */
int wubu_hive_insert(wubu_hive_t *h, void *ptr);

/* H3: erase a pointer (mark skip + push freelist). O(1). */
int wubu_hive_erase(wubu_hive_t *h, void *ptr);

/* H4: iterate all LIVE slots. The callback receives each live pointer;
 * return nonzero to stop early. Returns the count visited. */
size_t wubu_hive_foreach(wubu_hive_t *h,
                         int (*fn)(void *ptr, void *user), void *user);

/* H5: the live count. */
size_t wubu_hive_live(const wubu_hive_t *h);

/* H6: clear everything (frees all blocks). */
void wubu_hive_clear(wubu_hive_t *h);

/* H7: total capacity (slots across all blocks). */
size_t wubu_hive_capacity(const wubu_hive_t *h);

#endif
