/* test_eg81_dfly.c -- EG-81 gate: checkpoint-native shared-KV drafting. */
#include <stdio.h>
#include <string.h>

#include "wubu_dfly.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_dfly_t *p = wubu_dfly_create(4, 8, 64, 16);
    CHECK(p != NULL);
    CHECK(wubu_dfly_separate_kv_allocations(p) == 0);

    /* Draft layers read the target KV; no second KV allocation is made. */
    CHECK(wubu_dfly_share_kv(p, 0) == 512);  /* 8 draft tokens * 64 bytes */
    CHECK(wubu_dfly_share_kv(p, 1) == 512);
    CHECK(wubu_dfly_share_kv(p, 2) == 512);
    CHECK(wubu_dfly_separate_kv_allocations(p) == 0);
    CHECK(wubu_dfly_bytes_avoided(p) == 1536);
    printf("EG-81 shared KV on 3 layers, avoided=%zu bytes, separate_allocs=0\n",
           wubu_dfly_bytes_avoided(p));

    /* Sparse indices are set once and matched by all draft layers. */
    {
        int idx[4] = {2, 5, 7, 11}, wrong[4] = {2, 5, 7, 12};
        CHECK(wubu_dfly_set_indices(p, idx, 4) == 0);
        CHECK(wubu_dfly_indices_match(p, idx, 4) == 1);
        CHECK(wubu_dfly_indices_match(p, wrong, 4) == 0);
        CHECK(wubu_dfly_share_kv(p, 0) == 0);      /* no double count */
        CHECK(wubu_dfly_share_kv(p, 9) == 0);      /* bad layer */
        CHECK(wubu_dfly_bytes_avoided(p) == 1536);
        printf("EG-81 sparse indices shared; duplicate/invalid share adds 0 bytes\n");
        CHECK(wubu_dfly_set_indices(p, NULL, 4) == -1);
        CHECK(wubu_dfly_set_indices(p, idx, 0) == -1);
        CHECK(wubu_dfly_indices_match(p, idx, 0) == -1);
        CHECK(wubu_dfly_create(0, 8, 64, 16) == NULL);
        CHECK(wubu_dfly_create(4, 0, 64, 16) == NULL);
        CHECK(wubu_dfly_create(4, 8, 0, 16) == NULL);
        CHECK(wubu_dfly_separate_kv_allocations(NULL) == -1);
    }

    wubu_dfly_free(p);
    if (failures) { fprintf(stderr, "test_eg81_dfly: FAIL (%d)\n", failures); return 1; }
    printf("test_eg81_dfly: PASS\n");
    return 0;
}
