/*
 * wubu_hive.c -- THE HIVE: the AGI's memory structure (WuBu).
 *
 * Linked fixed blocks + bit skipfield + per-block freelist. Pure C11.
 *
 * The skipfield is one BIT per slot (cap/8 bytes per block). A slot is
 * "live" when skip bit i == 0. Erase: set bit, live--, chain the slot
 * onto the per-block freelist. Insert: pop the freelist first (clearing
 * the bit), else append to the tail block, else allocate a new block.
 *
 * The per-block freelist is chained INSIDE the slots array: a free slot
 * stores the index of the next free slot as a uintptr_t. Zero extra
 * storage. This is the same design as the kernel hive
 * (wubuos/src/kernel/wubu_hive.c) — ONE canonical implementation.
 *
 * Iteration walks blocks, and within a block checks the skip bit to
 * jump erased slots. The freelist makes erase/insert O(1) with stable
 * pointers and no compaction.
 */
#include "wubu_hive.h"
#include <stdlib.h>
#include <string.h>

#define BLOCK_CAP WUBU_HIVE_BLOCK_CAP
#define SKIP_BYTES ((BLOCK_CAP + 7) / 8)

/* ---- bit skipfield helpers ---- */
static inline int skip_get(const wubu_hive_block_t *b, size_t i) {
    return (b->skip[i >> 3] >> (i & 7)) & 1;
}
static inline void skip_set(wubu_hive_block_t *b, size_t i) {
    b->skip[i >> 3] |= (uint8_t)(1u << (i & 7));
}
static inline void skip_clr(wubu_hive_block_t *b, size_t i) {
    b->skip[i >> 3] &= (uint8_t)~(1u << (i & 7));
}

static wubu_hive_block_t *block_alloc(void)
{
    wubu_hive_block_t *blk = (wubu_hive_block_t *)calloc(1, sizeof(*blk));
    if (!blk) return NULL;
    blk->slots = (void **)calloc(BLOCK_CAP, sizeof(void *));
    blk->skip = (uint8_t *)calloc(SKIP_BYTES, 1);
    if (!blk->slots || !blk->skip) {
        free(blk->slots); free(blk->skip); free(blk);
        return NULL;
    }
    /* A fresh block: EVERY slot is dead/free. skip bit set = dead,
     * so fill the skipfield with 0xFF. The freelist chain fills the
     * slots array (slot i -> i+1, last -> NONE). */
    memset(blk->skip, 0xFF, SKIP_BYTES);
    for (size_t i = 0; i < BLOCK_CAP; i++)
        blk->slots[i] = (void *)((i + 1 < BLOCK_CAP) ? (i + 1) : WUBU_HIVE_NONE);
    blk->cap = BLOCK_CAP;
    blk->live = 0;
    blk->free_head = 0;
    blk->next = NULL;
    return blk;
}

static void block_free(wubu_hive_block_t *blk)
{
    if (!blk) return;
    free(blk->slots);
    free(blk->skip);
    free(blk);
}

int wubu_hive_init(wubu_hive_t *h)
{
    if (!h) return -1;
    memset(h, 0, sizeof(*h));
    return 0;
}

int wubu_hive_insert(wubu_hive_t *h, void *ptr)
{
    if (!h || !ptr) return -1;

    /* 1. Find a block with a free slot (free_head != NONE) */
    wubu_hive_block_t *blk = h->head;
    for (; blk && blk->free_head == WUBU_HIVE_NONE; blk = blk->next) {}
    if (!blk) {
        /* 2. No free slot anywhere — allocate a new block */
        blk = block_alloc();
        if (!blk) return -1;
        if (h->tail) h->tail->next = blk;
        else h->head = blk;
        h->tail = blk;
        h->n_blocks++;
        h->allocs++;
    } else {
        h->reuses++;
    }

    /* 3. Pop from the per-block freelist */
    size_t idx = blk->free_head;
    blk->free_head = (size_t)(uintptr_t)blk->slots[idx];
    blk->slots[idx] = ptr;
    skip_clr(blk, idx);
    blk->live++;
    h->total_live++;
    return 0;
}

int wubu_hive_erase(wubu_hive_t *h, void *ptr)
{
    if (!h || !ptr) return -1;
    for (wubu_hive_block_t *blk = h->head; blk; blk = blk->next) {
        if (blk->live == 0) continue;
        for (size_t s = 0; s < blk->cap; s++) {
            if (!skip_get(blk, s) && blk->slots[s] == ptr) {
                /* Push onto the per-block freelist */
                blk->slots[s] = (void *)(uintptr_t)blk->free_head;
                blk->free_head = s;
                skip_set(blk, s);
                blk->live--;
                h->total_live--;
                return 0;
            }
        }
    }
    return -1;   /* not found */
}

size_t wubu_hive_foreach(wubu_hive_t *h,
                         int (*fn)(void *ptr, void *user), void *user)
{
    if (!h || !fn) return 0;
    size_t visited = 0;
    for (wubu_hive_block_t *blk = h->head; blk; blk = blk->next) {
        if (blk->live == 0) continue;
        for (size_t s = 0; s < blk->cap; s++) {
            if (!skip_get(blk, s)) {
                visited++;
                if (fn(blk->slots[s], user)) return visited;
            }
        }
    }
    return visited;
}

size_t wubu_hive_live(const wubu_hive_t *h)
{
    return h ? h->total_live : 0;
}

size_t wubu_hive_capacity(const wubu_hive_t *h)
{
    if (!h) return 0;
    return h->n_blocks * BLOCK_CAP;
}

void wubu_hive_clear(wubu_hive_t *h)
{
    if (!h) return;
    wubu_hive_block_t *blk = h->head;
    while (blk) {
        wubu_hive_block_t *nx = blk->next;
        block_free(blk);
        blk = nx;
    }
    memset(h, 0, sizeof(*h));
}

void wubu_hive_clear_with(wubu_hive_t *h, void (*free_fn)(void *ptr))
{
    if (!h) return;
    if (free_fn) {
        wubu_hive_foreach(h, (int (*)(void *, void *))free_fn, NULL);
    }
    wubu_hive_clear(h);
}
