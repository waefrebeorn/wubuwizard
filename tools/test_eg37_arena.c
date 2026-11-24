/* test_eg37_arena.c -- EG-37 gate: unified arena + whole-request admission. */
#include <stdio.h>

#include "wubu_unified_arena.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_arena_t *a = wubu_uarena_create(1000);
    const wubu_arena_block_t *b;
    CHECK(a != NULL);
    if (!a) return 1;

    CHECK(wubu_uarena_alloc_request(a, 0, 3, WUBU_ARENA_KV, 100) == 0);
    CHECK(wubu_uarena_alloc(a, 10, WUBU_ARENA_STATE, 150) == 0);
    CHECK(wubu_uarena_alloc(a, 20, WUBU_ARENA_KV, 100) == 0);
    CHECK(wubu_uarena_alloc(a, 30, WUBU_ARENA_STATE, 100) == 0);
    CHECK(wubu_uarena_used(a) == 650);
    CHECK(wubu_uarena_validate(a) == 1);

    CHECK(wubu_uarena_free_token(a, 0) == 1);
    CHECK(wubu_uarena_free_token(a, 1) == 1);
    CHECK(wubu_uarena_free_token(a, 2) == 1);
    CHECK(wubu_uarena_free_token(a, 20) == 1);
    CHECK(wubu_uarena_used(a) == 250);

    /* 400 bytes free but fragmented: compaction must make it admissible. */
    CHECK(wubu_uarena_alloc_request(a, 100, 2, WUBU_ARENA_KV, 200) == 0);
    printf("EG-37 after compaction used=%zu blocks=%d\n",
           wubu_uarena_used(a), a->n_blocks);
    CHECK(wubu_uarena_used(a) == 650 && wubu_uarena_validate(a) == 1);
    b = &a->blocks[0];
    while (b && !(b->used && b->token_id == 100)) {
        b = NULL;
        for (int i = 0; i < a->n_blocks; i++)
            if (a->blocks[i].used && a->blocks[i].token_id == 100) {
                b = &a->blocks[i]; break;
            }
    }
    CHECK(b != NULL && b->offset >= 250);

    /* Whole-request rejection: no partial allocation on insufficient space. */
    CHECK(wubu_uarena_alloc_request(a, 200, 2, WUBU_ARENA_KV, 200) == -1);
    CHECK(wubu_uarena_used(a) == 650);

    CHECK(wubu_uarena_create(0) == NULL);
    CHECK(wubu_uarena_alloc(NULL, 1, WUBU_ARENA_KV, 10) == -1);
    CHECK(wubu_uarena_alloc(a, 1, WUBU_ARENA_KV, 0) == -1);
    CHECK(wubu_uarena_alloc_request(a, 1, 0, WUBU_ARENA_KV, 10) == -1);
    CHECK(wubu_uarena_validate(NULL) == 0);

    wubu_uarena_free(a);
    if (failures) {
        fprintf(stderr, "test_eg37_arena: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg37_arena: PASS\n");
    return 0;
}
