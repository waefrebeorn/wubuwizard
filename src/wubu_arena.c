/*
 * wubu_arena.c -- the kernel's single allocator of record.
 *
 * Theory/07: one arena, bump-allocated; slabs are alignment-tagged so
 * the byte budget is predictable and Styx-exportable. The arena replaces
 * the 14 separate calloc calls in wubu_load with a single mem_alloc +
 * recorded slab table.
 *
 * Alignments used by the kernel:
 *   AVX-512 GEMM ........ 64  (vmovdqa / vpmovs-wx)
 *   F32 tensor ........... 4
 *   F16 / q8 dequant .... 2
 *   int4 pack ........... 1 (but slab padded to parent align)
 */
#include "wubu_arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static size_t align_up(size_t n, size_t a) {
    return (n + (a - 1)) & ~(a - 1);
}

static size_t clamp_align(size_t a) {
    if (a < 1) return 1;
    if (a & (a - 1)) return 64;   /* not a power of two → use max safe */
    return a;
}

int wubu_prec_bytes(int prec) {
    switch (prec) {
        case WUBU_PREC_F32:  return 4;
        case WUBU_PREC_F16:  return 2;
        case WUBU_PREC_INT4: return 0;   /* 0.5 (nybble) — caller handles packing */
        case WUBU_PREC_Q8_0: return 9;   /* 1.0 + 1/8 (scale per group of 32) */
        case WUBU_PREC_Q4_0: return 5;   /* 0.56 (4-bit + per-block scale) */
        default:             return 0;
    }
}

wubu_arena_t *wubu_arena_create(size_t bytes)
{
    wubu_arena_t *a = (wubu_arena_t *)calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->capacity = align_up(bytes, 64);
    a->slab_cap = 16;
    a->slabs = (wubu_slab_t *)calloc(a->slab_cap, sizeof(wubu_slab_t));
    if (!a->slabs) { free(a); return NULL; }
    /* The kernel's mem_alloc guarantees 64-byte alignment (page-mapped).
     * On libc (tests) posix_memalign gives the same guarantee so the
     * alignment invariant holds without a second allocator. */
    if (posix_memalign((void **)&a->base, 64, a->capacity) != 0 || !a->base) {
        free(a->slabs); free(a); return NULL;
    }
    a->pos = 0;
    return a;
}

void *wubu_arena_push(wubu_arena_t *a, const char *name,
                      size_t sz, size_t align, int prec, int elems)
{
    if (!a) return NULL;
    size_t a_align = clamp_align(align);
    size_t off = align_up(a->pos, a_align);
    if (off + sz > a->capacity) return NULL;   /* slab overflow = OOM */
    void *p = a->base + off;
    a->pos = off + sz;

    /* Grow slab table if needed. */
    if (a->n_slabs >= a->slab_cap) {
        a->slab_cap *= 2;
        wubu_slab_t *ns = (wubu_slab_t *)realloc(a->slabs,
            (size_t)a->slab_cap * sizeof(wubu_slab_t));
        if (!ns) return NULL;
        a->slabs = ns;
    }
    wubu_slab_t *s = &a->slabs[a->n_slabs++];
    s->name = name;
    s->base = p;
    s->bytes = sz;
    s->padded = off - (a->pos - sz) + sz;  /* sz + leading pad */
    s->align = a_align;
    s->prec = prec;
    s->elems = elems;
    return p;
}

void wubu_arena_reset(wubu_arena_t *a)
{
    if (!a) return;
    a->pos = 0;
}

int wubu_arena_free(wubu_arena_t *a)
{
    if (!a) return 0;
    free(a->base);
    free(a->slabs);
    free(a);
    return 0;
}

size_t wubu_arena_used(const wubu_arena_t *a) {
    return a ? a->pos : 0;
}

int wubu_arena_nslabs(const wubu_arena_t *a) {
    return a ? a->n_slabs : 0;
}

const wubu_slab_t *wubu_arena_slab(const wubu_arena_t *a, int i) {
    if (!a || i < 0 || i >= a->n_slabs) return NULL;
    return &a->slabs[i];
}

const wubu_slab_t *wubu_arena_find(const wubu_arena_t *a, const char *name) {
    if (!a || !name) return NULL;
    for (int i = 0; i < a->n_slabs; i++)
        if (a->slabs[i].name && strcmp(a->slabs[i].name, name) == 0)
            return &a->slabs[i];
    return NULL;
}
