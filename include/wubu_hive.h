/*
 * wubu_hive.h -- THE HIVE: the AGI's memory structure (WuBu).
 *
 * The user's diagram: Vector = contiguous array (fast but reallocation
 * moves everything); List = scattered nodes (stable pointers but cache
 * misses); Hive = linked FIXED BLOCKS + bit skipfield + freelist.
 *
 *   struct block {
 *       void **slots;      // fixed block of pointers (cache-friendly)
 *       uint8_t *skip;     // bit skipfield: bit i set = slot i dead
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
 * Implementation: bit-level skipfield (1 bit per slot, 8x less memory
 * than byte-per-slot), per-block freelist chained through the slots
 * array (zero extra storage). Same design as the kernel hive
 * (wubuos/src/kernel/wubu_hive.c) — ONE canonical hive.
 *
 * Pure C11, no templates, no third-party.
 */
#ifndef WUBU_HIVE_H
#define WUBU_HIVE_H

#include <stddef.h>
#include <stdint.h>

/* The fixed block size: 64 slots per block (cache-line-friendly;
 * 64 void* = 512 bytes on 64-bit). */
#define WUBU_HIVE_BLOCK_CAP 64

typedef struct wubu_hive_block {
    void **slots;                 /* [cap] caller-owned pointers (free slots chain indices) */
    uint8_t *skip;                /* [cap/8] bit skipfield: bit i set = slot i dead */
    size_t live;                  /* live slots in this block */
    size_t cap;                   /* = WUBU_HIVE_BLOCK_CAP */
    size_t free_head;             /* first free slot index, WUBU_HIVE_NONE if full */
    struct wubu_hive_block *next; /* the chain */
} wubu_hive_block_t;

typedef struct {
    wubu_hive_block_t *head;      /* first block */
    wubu_hive_block_t *tail;      /* last block (append) */
    size_t n_blocks;
    size_t total_live;
    /* stats */
    size_t allocs;                /* slots allocated */
    size_t reuses;                /* slots reused from the freelist */
} wubu_hive_t;

#define WUBU_HIVE_NONE ((size_t)-1)

/* ---- bit skipfield helpers (for direct block access by callers) ---- */
static inline int hive_skip_get(const wubu_hive_block_t *b, size_t i) {
    return (b->skip[i >> 3] >> (i & 7)) & 1;
}

/* H1: init an empty hive. */
int wubu_hive_init(wubu_hive_t *h);

/* H2: insert a pointer. Returns 0 on success. */
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

/* H8: clear + free the user payloads first (ASan-clean teardown). */
void wubu_hive_clear_with(wubu_hive_t *h, void (*free_fn)(void *ptr));

#endif
