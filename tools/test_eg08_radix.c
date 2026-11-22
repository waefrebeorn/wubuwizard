/* test_eg08_radix.c -- EG-08 gate: radix prefix tree with sibling sharing
 * and LRU eviction. Prints: test_eg08_radix: PASS. */
#include <stdio.h>

#include "wubu_radix_cache.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    const int A[4] = { 10, 20, 30, 40 };
    const int B[4] = { 10, 20, 50, 60 };
    const int partial[4] = { 10, 20, 30, 77 };
    const int none[1] = { 99 };
    const wubu_radix_node_t *node = NULL;

    wubu_radix_cache_t *c = wubu_radix_cache_create(16);
    CHECK(c != NULL);
    if (!c) return 1;

    CHECK(wubu_radix_cache_insert(c, A, 4, 1) == 0);
    CHECK(wubu_radix_cache_insert(c, B, 4, 2) == 0);
    int nodes = wubu_radix_cache_node_count(c);
    printf("EG-08 nodes=%d (naive would be 9)\n", nodes);
    CHECK(nodes == 7); /* root + 2 shared + 2 + 2 tails */
    CHECK(nodes < 9);  /* sibling prefix sharing is real */

    /* Root has one child (token 10); the shared nodes branch underneath. */
    CHECK(wubu_radix_cache_match(c, A, 0, &node) == 0);
    CHECK(node != NULL);
    CHECK(wubu_radix_cache_child_count(node) == 1);
    int len = wubu_radix_cache_match(c, A, 4, &node);
    CHECK(len == 4);
    CHECK(node != NULL);
    CHECK(wubu_radix_cache_block(node) == 1);

    len = wubu_radix_cache_match(c, partial, 4, &node);
    printf("EG-08 partial match=%d\n", len);
    CHECK(len == 3);
    CHECK(wubu_radix_cache_block(node) == -1); /* not a terminal */

    len = wubu_radix_cache_match(c, B, 4, &node);
    CHECK(len == 4);
    CHECK(wubu_radix_cache_block(node) == 2);

    CHECK(wubu_radix_cache_match(c, none, 1, &node) == 0);
    CHECK(wubu_radix_cache_match(c, A, 2, &node) == 2);
    printf("EG-08 hits=%llu misses=%llu\n",
           (unsigned long long)wubu_radix_cache_hits(c),
           (unsigned long long)wubu_radix_cache_misses(c));
    CHECK(wubu_radix_cache_hits(c) >= 4);
    CHECK(wubu_radix_cache_misses(c) >= 1);

    int evicted = wubu_radix_cache_evict_lru(c, 1);
    printf("EG-08 evicted=%d nodes=%d total_evictions=%llu\n",
           evicted, wubu_radix_cache_node_count(c),
           (unsigned long long)wubu_radix_cache_evictions(c));
    CHECK(evicted == 1);
    CHECK(wubu_radix_cache_node_count(c) == 6);
    CHECK(wubu_radix_cache_evictions(c) == 1);
    CHECK(wubu_radix_cache_evict_lru(c, 0) == 0);

    /* Fail-closed validation. */
    CHECK(wubu_radix_cache_create(1) == NULL);
    CHECK(wubu_radix_cache_insert(c, NULL, 4, 1) == -1);
    CHECK(wubu_radix_cache_match(c, NULL, 4, &node) == -1);

    wubu_radix_cache_free(c);
    if (failures) {
        fprintf(stderr, "test_eg08_radix: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg08_radix: PASS\n");
    return 0;
}
