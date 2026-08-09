/*
 * wubu_arena.h -- the kernel's single allocator of record.
 *
 * Theory/07 ADR-004: the model uses ONE arena, bump-allocated from mem_alloc,
 * NOT 14 separate calloc calls. The arena owns: weights (frozen slab),
 * KV cache (tiered slab), and intermediates (transient scratch, reset per
 * step). Every slab is alignment-tagged so AVX-512 (64 B) and QKV striding
 * (block-granularity) hold.
 *
 * The byte budget is a slab table: name → {base, bytes, align, precision}.
 * A Styx/KVFS export reads the table for path-addressable slab access.
 */
#ifndef WUBU_ARENA_H
#define WUBU_ARENA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Precision tag for a slab (the quant cascade). */
typedef enum {
    WUBU_PREC_F32   = 0,   /* 4 bytes/elem — fallback */
    WUBU_PREC_F16   = 1,   /* 2 bytes/elem — kernel default for weights */
    WUBU_PREC_INT4  = 2,   /* 0.5 bytes/elem — compute-bound crossover */
    WUBU_PREC_Q8_0  = 3,   /* 1.125 bytes/elem — near-lossless KV */
    WUBU_PREC_Q4_0  = 4,   /* 0.56 bytes/elem — long-context KV */
} wubu_prec_t;

/* One slab in the byte budget: name → location → precision. */
typedef struct {
    const char  *name;     /* slab name (Styx path key) */
    void        *base;     /* slab start (aligned) */
    size_t       bytes;    /* slab size (UNpadded — the used bytes) */
    size_t       padded;   /* bytes incl. alignment padding to .align */
    size_t       align;    /* alignment requirement */
    int          prec;     /* wubu_prec_t — the quant cascade step */
    int          elems;    /* element count (for introspection) */
} wubu_slab_t;

typedef struct wubu_arena {
    uint8_t      *base;    /* backing allocation (one mem_alloc) */
    size_t        capacity;/* total bytes reserved */
    size_t        pos;     /* bump cursor */
    wubu_slab_t  *slabs;   /* slab table (growable) */
    int           n_slabs;
    int           slab_cap;
} wubu_arena_t;

/* Create an arena of `bytes` (rounded up to 64). Returns NULL on OOM. */
wubu_arena_t *wubu_arena_create(size_t bytes);

/* Bump-allocate `sz` bytes aligned to `align`. Returns NULL on overflow.
 * Records a slab for introspection/export. */
void *wubu_arena_push(wubu_arena_t *a, const char *name,
                      size_t sz, size_t align, int prec, int elems);

/* Rewind the cursor (transient scratch slabs — KV reset, step reset).
 * Does NOT zero; caller owns stale data until it overwrites. */
void wubu_arena_reset(wubu_arena_t *a);

/* Free the arena (single call → single mem_free). */
int   wubu_arena_free(wubu_arena_t *a);

/* Introspection (for the byte-budget test / Styx export). */
size_t wubu_arena_used(const wubu_arena_t *a);
int    wubu_arena_nslabs(const wubu_arena_t *a);
const wubu_slab_t *wubu_arena_slab(const wubu_arena_t *a, int i);
const wubu_slab_t *wubu_arena_find(const wubu_arena_t *a, const char *name);

/* bytes_per_element for a precision tag (0 = unknown). */
int    wubu_prec_bytes(int prec);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_ARENA_H */
