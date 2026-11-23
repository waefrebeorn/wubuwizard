/* test_eg36_expert_cache.c -- EG-36b gate: expert LRFU + ClassLRU policy. */
#include <stdio.h>

#include "wubu_expert_cache.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_expert_cache_t *p = wubu_expert_cache_create(4, 0);
    CHECK(p != NULL);

    /* Fill four slots: e0 is a protected shared expert, the rest are routed. */
    CHECK(wubu_expert_cache_access(p, 10, 1, 1) == 0);
    CHECK(wubu_expert_cache_access(p, 11, 1, 1) == 0);
    CHECK(wubu_expert_cache_access(p, 12, 1, 1) == 0);
    CHECK(wubu_expert_cache_access(p, 0, 0, 1) == 0);
    CHECK(wubu_expert_cache_count(p) == 4);

    /* Refresh e10; a miss at t=3 must evict an unprotected LRFU victim. */
    CHECK(wubu_expert_cache_access(p, 10, 1, 2) == 1);
    CHECK(wubu_expert_cache_access(p, 20, 1, 3) == 0);
    CHECK(wubu_expert_cache_find(p, 0) == 1);
    CHECK(wubu_expert_cache_find(p, 10) == 1);
    CHECK(wubu_expert_cache_find(p, 20) == 1);
    CHECK(wubu_expert_cache_find(p, 11) == 0);
    CHECK(wubu_expert_cache_evictions(p) == 1);
    CHECK(wubu_expert_cache_protected_evictions(p) == 0);
    CHECK(wubu_expert_cache_wait_units(p) == 0);
    printf("EG-36b evicted=11 protected=0 kept=[0 10 12 20] wait=0\n");

    /* When every slot is protected, ClassLRU must still evict and count it. */
    {
        wubu_expert_cache_t *q = wubu_expert_cache_create(2, 0);
        CHECK(q != NULL);
        CHECK(wubu_expert_cache_access(q, 1, 0, 1) == 0);
        CHECK(wubu_expert_cache_access(q, 2, 0, 1) == 0);
        CHECK(wubu_expert_cache_access(q, 3, 0, 2) == 0);
        CHECK(wubu_expert_cache_count(q) == 2);
        CHECK(wubu_expert_cache_evictions(q) == 1);
        CHECK(wubu_expert_cache_protected_evictions(q) == 1);
        CHECK(wubu_expert_cache_pending_fetches(q) == 3);
        CHECK(wubu_expert_cache_wait_units(q) == 0);
        printf("EG-36b all-protected eviction=1 pending=3 wait=0\n");
        CHECK(wubu_expert_cache_create(0, 0) == NULL);
        CHECK(wubu_expert_cache_access(NULL, 1, 0, 1) == -1);
        CHECK(wubu_expert_cache_find(q, -1) == -1);
        CHECK(wubu_expert_cache_count(NULL) == -1);
        wubu_expert_cache_free(q);
    }

    wubu_expert_cache_free(p);
    if (failures) { fprintf(stderr, "test_eg36_expert_cache: FAIL (%d)\n", failures); return 1; }
    printf("test_eg36_expert_cache: PASS\n");
    return 0;
}
