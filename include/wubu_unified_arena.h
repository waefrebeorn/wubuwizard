#ifndef WUBU_UNIFIED_ARENA_H
#define WUBU_UNIFIED_ARENA_H

/*
 * wubu_unified_arena.h -- EG-37: one arena for KV and session state, one
 * allocation entry per token, compaction and whole-request admission
 * (SGLang unified memory lineage). Pure C11, no third-party code.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_ARENA_MAX_BLOCKS 256

typedef enum {
    WUBU_ARENA_KV = 0,
    WUBU_ARENA_STATE = 1
} wubu_arena_kind_t;

typedef struct {
    int token_id;
    wubu_arena_kind_t kind;
    size_t offset;
    size_t bytes;
    int used;
} wubu_arena_block_t;

typedef struct {
    size_t capacity;
    wubu_arena_block_t blocks[WUBU_ARENA_MAX_BLOCKS];
    int n_blocks;
} wubu_arena_t;

wubu_arena_t *wubu_arena_create(size_t capacity);
void wubu_arena_free(wubu_arena_t *a);

/* Allocate one token's slot in either KV or STATE space. */
int wubu_arena_alloc(wubu_arena_t *a, int token_id,
                     wubu_arena_kind_t kind, size_t bytes);

/* All-or-nothing admission of n_tokens consecutive token ids. */
int wubu_arena_alloc_request(wubu_arena_t *a, int base_token, int n_tokens,
                             wubu_arena_kind_t kind, size_t bytes_per_token);

/* Free every block owned by a token. Returns number of blocks freed. */
int wubu_arena_free_token(wubu_arena_t *a, int token_id);

/* Slide live blocks down to remove holes. Returns bytes reclaimed. */
size_t wubu_arena_compact(wubu_arena_t *a);

size_t wubu_arena_used(const wubu_arena_t *a);

/* 1 when no two live blocks overlap and all lie inside capacity. */
int wubu_arena_validate(const wubu_arena_t *a);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_UNIFIED_ARENA_H */
