/*
 * wubu_hive.c -- THE HIVE: the AGI's memory structure (WuBu).
 *
 * Linked fixed blocks + skipfield + freelist. Pure C11.
 *
 * The skipfield lives in the block (uint8_t per slot). A slot is
 * "live" when skip[s] == 0. Erase: skip[s] = 1, live--, and push the
 * block/slot onto the hive freelist. Insert: pop the freelist first,
 * else append to the tail block, else allocate a new block.
 *
 * Iteration walks blocks, and within a block uses the skipfield to
 * jump erased slots (memchr-free scan; the skip byte read is one load
 * per slot -- cache-friendly). The freelist makes erase/insert O(1)
 * with stable pointers and no compaction.
 */
#include "wubu_hive.h"
#include <stdlib.h>
#include <string.h>

#define BLOCK_CAP WUBU_HIVE_BLOCK_CAP

static wubu_hive_block_t *block_alloc(void)
{
    wubu_hive_block_t *blk = (wubu_hive_block_t *)calloc(1, sizeof(*blk));
    if (!blk) return NULL;
    blk->slots = (void **)calloc(BLOCK_CAP, sizeof(void *));
    blk->skip = (uint8_t *)calloc(BLOCK_CAP, sizeof(uint8_t));
    if (!blk->slots || !blk->skip) {
        free(blk->slots); free(blk->skip); free(blk);
        return NULL;
    }
    /* a fresh block has NO live slots: the skipfield must be 1 (erased)
     * everywhere, otherwise foreach would visit never-written slots. The
     * DA test caught this: fresh calloc zeroed skip, so a 64-slot block
     * with 5 live reported 64 "live". Insert sets skip[s]=0. */
    memset(blk->skip, 1, BLOCK_CAP);
    blk->cap = BLOCK_CAP;
    blk->live = 0;
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

static void free_entries_push(wubu_hive_t *h, wubu_hive_block_t *blk, size_t s)
{
    if (h->n_free == h->free_cap) {
        size_t nc = h->free_cap ? h->free_cap * 2 : 16;
        struct wubu_hive_free_entry *nf =
            (struct wubu_hive_free_entry *)realloc(h->free_entries, nc * sizeof(*nf));
        if (!nf) return;
        h->free_entries = nf;
        h->free_cap = nc;
    }
    h->free_entries[h->n_free].block = blk;
    h->free_entries[h->n_free].slot = s;
    h->n_free++;
}

static int free_entries_pop(wubu_hive_t *h, wubu_hive_block_t **blk, size_t *s)
{
    if (h->n_free == 0) return -1;
    h->n_free--;
    *blk = h->free_entries[h->n_free].block;
    *s = h->free_entries[h->n_free].slot;
    return 0;
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
    /* 1. reuse a freelist entry (the LIFO pop) */
    wubu_hive_block_t *blk;
    size_t s;
    if (free_entries_pop(h, &blk, &s) == 0) {
        blk->slots[s] = ptr;
        blk->skip[s] = 0;
        blk->live++;
        h->total_live++;
        h->reuses++;
        return 0;
    }
    /* 2. append to the tail block if it has room. In a freelist-less
     * tail, live slots are contiguous at indices [0, live) -- the next
     * free slot is exactly at index `live` (skip is 0 there). */
    if (h->tail && h->tail->live < h->tail->cap) {
        wubu_hive_block_t *t = h->tail;
        size_t s2 = t->live;
        t->slots[s2] = ptr;
        t->skip[s2] = 0;
        t->live++;
        h->total_live++;
        h->allocs++;
        return 0;
    }
    /* 3. new block */
    wubu_hive_block_t *nb = block_alloc();
    if (!nb) return -1;
    nb->slots[0] = ptr;
    nb->skip[0] = 0;
    nb->live = 1;
    if (h->tail) h->tail->next = nb;
    else h->head = nb;
    h->tail = nb;
    h->n_blocks++;
    h->total_live++;
    h->allocs++;
    return 0;
}

int wubu_hive_erase(wubu_hive_t *h, void *ptr)
{
    if (!h || !ptr) return -1;
    for (wubu_hive_block_t *blk = h->head; blk; blk = blk->next) {
        if (blk->live == 0) continue;
        for (size_t s = 0; s < blk->cap; s++) {
            if (blk->skip[s] == 0 && blk->slots[s] == ptr) {
                blk->skip[s] = 1;
                blk->slots[s] = NULL;   /* drop the stale pointer */
                blk->live--;
                h->total_live--;
                /* push onto the freelist (LIFO) */
                free_entries_push(h, blk, s);
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
            if (blk->skip[s] == 0) {
                visited++;
                if (fn(blk->slots[s], user)) return visited;   /* stop */
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
    free(h->free_entries);
    memset(h, 0, sizeof(*h));
}

/* the hive holds USER payloads (opaque pointers); clear cannot free
 * them. clear_with(free_fn) calls free_fn on every live payload first
 * (the owner's destructor), then clears. The ASan-clean teardown for
 * hive-backed modules. */
void wubu_hive_clear_with(wubu_hive_t *h, void (*free_fn)(void *ptr))
{
    if (!h) return;
    if (free_fn) {
        wubu_hive_foreach(h, (int (*)(void *, void *))free_fn, NULL);
    }
    wubu_hive_clear(h);
}
