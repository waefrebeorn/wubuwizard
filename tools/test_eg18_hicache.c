/* test_eg18_hicache.c -- EG-18 gate: tiered cache, async write-back,
 * deadline-gated prefetch and immediate promotion on lower-tier hits. */
#include <stdio.h>
#include <string.h>

#include "wubu_hicache.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

static void fill(unsigned char *b, unsigned char v) {
    memset(b, v, WUBU_HICACHE_ENTRY_BYTES);
}
static int all(const unsigned char *b, unsigned char v) {
    for (int i = 0; i < WUBU_HICACHE_ENTRY_BYTES; i++)
        if (b[i] != v) return 0;
    return 1;
}

int main(void) {
    wubu_hicache_t *h = wubu_hicache_create(2, 2, 2);
    wubu_hicache_stats_t st;
    unsigned char d[WUBU_HICACHE_ENTRY_BYTES], out[WUBU_HICACHE_ENTRY_BYTES];

    CHECK(h != NULL);
    if (!h) return 1;

    for (int k = 1; k <= 4; k++) {
        fill(d, (unsigned char)(k * 10));
        CHECK(wubu_hicache_put(h, (uint64_t)k, d) == 0);
    }
    wubu_hicache_stats(h, &st);
    CHECK(st.l1 == 2 && st.pending == 2);   /* L1 evictions queued for L2 */

    CHECK(wubu_hicache_tick(h, 10) == 2);
    wubu_hicache_stats(h, &st);
    CHECK(st.l2 == 2 && st.pending == 0);

    /* L2 hit promotes to L1; L1 victim is queued as write-back. */
    CHECK(wubu_hicache_get(h, 1, out) == 2);
    CHECK(all(out, 10));
    wubu_hicache_stats(h, &st);
    CHECK(st.hits[1] == 1 && st.l1 == 2 && st.pending == 1);
    CHECK(wubu_hicache_tick(h, 20) == 1);

    /* Prefetch completes before get: L1 hit and no miss charged. */
    CHECK(wubu_hicache_prefetch(h, 2, 1000ULL) == 1);
    CHECK(wubu_hicache_tick(h, 70) == 1);
    CHECK(wubu_hicache_get(h, 2, out) == 1);
    CHECK(all(out, 20));

    /* Deadline too tight: no transfer scheduled. */
    CHECK(wubu_hicache_prefetch(h, 3, 71ULL) == 0);
    CHECK(wubu_hicache_get(h, 99, out) == -1);

    wubu_hicache_stats(h, &st);
    printf("EG-18 hits L1=%d L2=%d L3=%d miss=%d moves=%d pending=%d l1=%d l2=%d l3=%d\n",
           st.hits[0], st.hits[1], st.hits[2], st.misses, st.moves,
           st.pending, st.l1, st.l2, st.l3);
    CHECK(st.hits[0] >= 1 && st.hits[1] >= 1);
    CHECK(st.misses == 1);
    CHECK(st.l1 <= 2 && st.l2 <= 2 && st.l3 <= 2);
    CHECK(wubu_hicache_tick(h, 59) == -1); /* clock must be monotonic */

    wubu_hicache_free(h);
    if (failures) {
        fprintf(stderr, "test_eg18_hicache: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg18_hicache: PASS\n");
    return 0;
}
