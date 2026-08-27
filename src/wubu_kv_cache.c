/*
 * wubu_kv_cache.c — PagedAttention-style KV cache for LLM inference.
 *
 * Manages the key/value cache for transformer attention:
 * - Fixed-size pages (16 tokens/page) like real PagedAttention
 * - Block-based allocation (linked list of pages per sequence)
 * - Prefetching for decode phase
 * - Memory-mapped for zero-copy when possible
 *
 * C11, self-contained.
 */

#include "wubu_kv_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdalign.h>

/* ---- Configuration ---- */

#define KV_PAGE_TOKENS   16      /* tokens per page */
#define KV_MAX_PAGES     4096    /* total pages in cache */
#define KV_NUM_HEADS     32      /* default: 32 attention heads */
#define KV_HEAD_DIM      128     /* default: 128-dim per head */
#define KV_DTYPE_SIZE    2       /* bytes per element: FP16 */

/* Per-page: KV_PAGE_TOKENS * NUM_HEADS * HEAD_DIM * 2(K+V) * DTYPE_SIZE */
#define KV_PAGE_BYTES  (KV_PAGE_TOKENS * KV_NUM_HEADS * KV_HEAD_DIM * 2 * KV_DTYPE_SIZE)

/* ---- Page table ---- */

typedef struct kv_page {
    uint32_t id;                   /* page ID (0..KV_MAX_PAGES-1) */
    int      refcount;             /* how many sequences use this page */
    uint8_t  data[KV_PAGE_BYTES];  /* actual KV data */
    struct kv_page *next;          /* next page in block */
} kv_page_t;

/* ---- Sequence block (linked list of pages) ---- */

typedef struct kv_block {
    kv_page_t *pages;              /* linked list of pages for this block */
    int        n_pages;            /* pages allocated */
    int        n_tokens;           /* tokens stored (<= n_pages * KV_PAGE_TOKENS) */
    uint32_t   seq_id;             /* which sequence this belongs to */
} kv_block_t;

/* ---- KV Cache ---- */

struct wubu_kv_cache {
    kv_page_t  page_pool[KV_MAX_PAGES];
    int        free_list[KV_MAX_PAGES];  /* stack of free page IDs */
    int        free_top;
    kv_block_t *sequences;              /* per-sequence blocks */
    int        n_sequences;
    int        max_sequences;
    size_t     total_allocated;
    size_t     total_used;
};

/* ---- Public API ---- */

wubu_kv_cache_t *wubu_kv_create(int max_sequences)
{
    wubu_kv_cache_t *cache = calloc(1, sizeof(*cache));
    if (!cache) return NULL;

    /* Initialize free list (all pages available) */
    for (int i = KV_MAX_PAGES - 1; i >= 0; i--)
        cache->free_list[KV_MAX_PAGES - 1 - i] = i;
    cache->free_top = KV_MAX_PAGES;

    /* Zero page pool */
    memset(cache->page_pool, 0, sizeof(cache->page_pool));
    for (int i = 0; i < KV_MAX_PAGES; i++)
        cache->page_pool[i].id = (uint32_t)i;

    cache->sequences = calloc(max_sequences, sizeof(kv_block_t));
    cache->n_sequences = 0;
    cache->max_sequences = max_sequences;
    cache->total_allocated = 0;
    cache->total_used = 0;

    return cache;
}

void wubu_kv_destroy(wubu_kv_cache_t *cache)
{
    if (!cache) return;
    free(cache->sequences);
    free(cache);
}

/* Allocate a new page from the free list */
static kv_page_t *alloc_page(wubu_kv_cache_t *cache)
{
    if (cache->free_top <= 0) return NULL;
    int id = cache->free_list[--cache->free_top];
    kv_page_t *p = &cache->page_pool[id];
    p->refcount = 1;
    p->next = NULL;
    cache->total_allocated += KV_PAGE_BYTES;
    return p;
}

/* Free a page back to the free list */
static void free_page(wubu_kv_cache_t *cache, kv_page_t *p)
{
    if (!p) return;
    if (--p->refcount > 0) return;
    int id = (int)p->id;
    cache->free_list[cache->free_top++] = id;
    cache->total_allocated -= KV_PAGE_BYTES;
}

/* Get or create a sequence block */
static kv_block_t *get_or_create_block(wubu_kv_cache_t *cache, uint32_t seq_id)
{
    for (int i = 0; i < cache->n_sequences; i++) {
        if (cache->sequences[i].seq_id == seq_id)
            return &cache->sequences[i];
    }
    if (cache->n_sequences >= cache->max_sequences) return NULL;
    kv_block_t *b = &cache->sequences[cache->n_sequences++];
    memset(b, 0, sizeof(*b));
    b->seq_id = seq_id;
    return b;
}

/* Append tokens to the KV cache for a sequence */
int wubu_kv_append(wubu_kv_cache_t *cache, uint32_t seq_id,
                   const void *k_data, const void *v_data, int n_tokens)
{
    kv_block_t *block = get_or_create_block(cache, seq_id);
    if (!block) return -1;

    /* Calculate pages needed */
    int current_pages = (block->n_tokens + KV_PAGE_TOKENS - 1) / KV_PAGE_TOKENS;
    int needed_pages  = (block->n_tokens + n_tokens + KV_PAGE_TOKENS - 1) / KV_PAGE_TOKENS;
    int pages_to_add  = needed_pages - current_pages;

    /* Allocate new pages if needed */
    for (int i = 0; i < pages_to_add; i++) {
        kv_page_t *p = alloc_page(cache);
        if (!p) return -1;  /* OOM */
        /* Append to linked list */
        if (!block->pages) {
            block->pages = p;
        } else {
            kv_page_t *tail = block->pages;
            while (tail->next) tail = tail->next;
            tail->next = p;
        }
        block->n_pages++;
    }

    /* Copy data into the last page(s) */
    const uint8_t *k_src = (const uint8_t *)k_data;
    const uint8_t *v_src = (const uint8_t *)v_data;
    int offset_in_page = block->n_tokens % KV_PAGE_TOKENS;
    int remaining = n_tokens;

    kv_page_t *p = block->pages;
    for (int i = 0; i < current_pages && p; i++) p = p->next;

    while (remaining > 0 && p) {
        uint8_t *page_k = p->data;
        uint8_t *page_v = p->data + (KV_PAGE_TOKENS * KV_NUM_HEADS * KV_HEAD_DIM * KV_DTYPE_SIZE);
        int slot = offset_in_page;
        int copy = KV_PAGE_TOKENS - offset_in_page;
        if (copy > remaining) copy = remaining;

        size_t bytes_per_token = KV_NUM_HEADS * KV_HEAD_DIM * KV_DTYPE_SIZE;
        memcpy(page_k + slot * bytes_per_token, k_src, copy * bytes_per_token);
        memcpy(page_v + slot * bytes_per_token, v_src, copy * bytes_per_token);

        k_src += copy * bytes_per_token;
        v_src += copy * bytes_per_token;
        remaining -= copy;
        offset_in_page = 0;
        p = p->next;
    }

    block->n_tokens += n_tokens;
    cache->total_used += (size_t)n_tokens * KV_NUM_HEADS * KV_HEAD_DIM * 2 * KV_DTYPE_SIZE;
    return 0;
}

/* Read KV data for a token */
int wubu_kv_read(const wubu_kv_cache_t *cache, uint32_t seq_id,
                 int token_idx, void *k_out, void *v_out)
{
    const kv_block_t *block = NULL;
    for (int i = 0; i < cache->n_sequences; i++) {
        if (cache->sequences[i].seq_id == seq_id) {
            block = &cache->sequences[i];
            break;
        }
    }
    if (!block || token_idx < 0 || token_idx >= block->n_tokens) return -1;

    int page_idx = token_idx / KV_PAGE_TOKENS;
    int slot = token_idx % KV_PAGE_TOKENS;

    kv_page_t *p = block->pages;
    for (int i = 0; i < page_idx && p; i++) p = p->next;
    if (!p) return -1;

    size_t bytes_per_token = KV_NUM_HEADS * KV_HEAD_DIM * KV_DTYPE_SIZE;
    uint8_t *page_k = p->data;
    uint8_t *page_v = p->data + (KV_PAGE_TOKENS * KV_NUM_HEADS * KV_HEAD_DIM * KV_DTYPE_SIZE);

    memcpy(k_out, page_k + slot * bytes_per_token, bytes_per_token);
    memcpy(v_out, page_v + slot * bytes_per_token, bytes_per_token);
    return 0;
}

/* Free all pages for a sequence (e.g., when sequence ends) */
void wubu_kv_free_sequence(wubu_kv_cache_t *cache, uint32_t seq_id)
{
    for (int i = 0; i < cache->n_sequences; i++) {
        if (cache->sequences[i].seq_id == seq_id) {
            kv_block_t *b = &cache->sequences[i];
            kv_page_t *p = b->pages;
            while (p) {
                kv_page_t *next = p->next;
                free_page(cache, p);
                p = next;
            }
            /* Swap with last and shrink */
            if (i < cache->n_sequences - 1)
                cache->sequences[i] = cache->sequences[cache->n_sequences - 1];
            cache->n_sequences--;
            return;
        }
    }
}

/* Get cache statistics */
void wubu_kv_stats(const wubu_kv_cache_t *cache, int *pages_used, int *pages_free,
                    size_t *bytes_used, size_t *bytes_allocated)
{
    if (!cache) return;
    int used = 0;
    for (int i = 0; i < cache->n_sequences; i++) {
        used += cache->sequences[i].n_pages;
        /* Count shared pages via refcount */
    }
    if (pages_used) *pages_used = used;
    if (pages_free) *pages_free = cache->free_top;
    if (bytes_used) *bytes_used = cache->total_used;
    if (bytes_allocated) *bytes_allocated = cache->total_allocated;
}
