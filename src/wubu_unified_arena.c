/*
 * wubu_unified_arena.c -- EG-37 unified arena. Pure C11.
 */
#include "wubu_unified_arena.h"

#include <stdlib.h>
#include <string.h>

wubu_arena_t *wubu_uarena_create(size_t capacity) {
    wubu_arena_t *a;
    if (capacity == 0) return NULL;
    a = (wubu_arena_t *)calloc(1, sizeof(*a));
    if (a) a->capacity = capacity;
    return a;
}

void wubu_uarena_free(wubu_arena_t *a) { free(a); }

static size_t arena_free_bytes(const wubu_arena_t *a) {
    return a->capacity - wubu_uarena_used(a);
}

static size_t first_fit_offset(const wubu_arena_t *a, size_t bytes) {
    size_t cursor = 0;
    for (int i = 0; i < a->n_blocks; i++) {
        const wubu_arena_block_t *b = &a->blocks[i];
        if (!b->used) continue;
        if (b->offset > cursor && b->offset - cursor >= bytes) return cursor;
        if (b->offset + b->bytes > cursor) cursor = b->offset + b->bytes;
    }
    return a->capacity - cursor >= bytes ? cursor : (size_t)-1;
}

int wubu_uarena_alloc(wubu_arena_t *a, int token_id,
                     wubu_arena_kind_t kind, size_t bytes) {
    size_t off;
    wubu_arena_block_t *b;
    if (!a || bytes == 0 || a->n_blocks >= WUBU_ARENA_MAX_BLOCKS) return -1;
    off = first_fit_offset(a, bytes);
    if (off == (size_t)-1) return -1;
    b = &a->blocks[a->n_blocks++];
    b->token_id = token_id;
    b->kind = kind;
    b->offset = off;
    b->bytes = bytes;
    b->used = 1;
    return 0;
}

int wubu_uarena_alloc_request(wubu_arena_t *a, int base_token, int n_tokens,
                             wubu_arena_kind_t kind, size_t bytes_per_token) {
    int i, start;
    if (!a || n_tokens <= 0 || bytes_per_token == 0) return -1;
    if ((size_t)n_tokens * bytes_per_token > arena_free_bytes(a)) return -1;
    (void)wubu_uarena_compact(a); /* guarantee one contiguous free region */
    start = a->n_blocks;
    for (i = 0; i < n_tokens; i++) {
        if (wubu_uarena_alloc(a, base_token + i, kind, bytes_per_token) != 0) {
            while (a->n_blocks > start) {
                a->n_blocks--;
                memset(&a->blocks[a->n_blocks], 0, sizeof(a->blocks[0]));
            }
            return -1;
        }
    }
    return 0;
}

int wubu_uarena_free_token(wubu_arena_t *a, int token_id) {
    int freed = 0;
    if (!a) return 0;
    for (int i = 0; i < a->n_blocks; i++)
        if (a->blocks[i].used && a->blocks[i].token_id == token_id) {
            a->blocks[i].used = 0;
            freed++;
        }
    return freed;
}

size_t wubu_uarena_compact(wubu_arena_t *a) {
    size_t cursor = 0, before;
    if (!a) return 0;
    before = wubu_uarena_used(a);
    for (int i = 0; i < a->n_blocks; i++) {
        wubu_arena_block_t *b = &a->blocks[i];
        if (!b->used) { b->offset = 0; b->bytes = 0; continue; }
        for (int j = i + 1; j < a->n_blocks; j++) {
            wubu_arena_block_t *c = &a->blocks[j];
            if (c->used && c->offset < b->offset) {
                wubu_arena_block_t t = *b; *b = *c; *c = t;
            }
        }
    }
    for (int i = 0; i < a->n_blocks; i++) {
        wubu_arena_block_t *b = &a->blocks[i];
        if (!b->used) continue;
        b->offset = cursor;
        cursor += b->bytes;
    }
    return before - wubu_uarena_used(a);
}

size_t wubu_uarena_used(const wubu_arena_t *a) {
    size_t total = 0;
    if (!a) return 0;
    for (int i = 0; i < a->n_blocks; i++)
        if (a->blocks[i].used) total += a->blocks[i].bytes;
    return total;
}

int wubu_uarena_validate(const wubu_arena_t *a) {
    if (!a) return 0;
    for (int i = 0; i < a->n_blocks; i++) {
        const wubu_arena_block_t *b = &a->blocks[i];
        if (!b->used) continue;
        if (b->offset > a->capacity || b->bytes > a->capacity - b->offset)
            return 0;
        for (int j = i + 1; j < a->n_blocks; j++) {
            const wubu_arena_block_t *c = &a->blocks[j];
            if (!c->used) continue;
            if (b->offset < c->offset + c->bytes &&
                c->offset < b->offset + b->bytes) return 0;
        }
    }
    return 1;
}
