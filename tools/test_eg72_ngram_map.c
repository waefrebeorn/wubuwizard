/* test_eg72_ngram_map.c -- EG-72 gate: k4v n-gram map with hit counts. */
#include <stdio.h>

#include "wubu_ngram_map.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_ngram_map_t *p = wubu_ngram_map_create(16, 2);
    int ctx[2] = {10, 11}, vals[4], counts[4];
    int n;
    CHECK(p != NULL);

    /* Five distinct continuations compete for four value slots. */
    CHECK(wubu_ngram_map_observe(p, ctx, 2, 12) == 0);
    CHECK(wubu_ngram_map_observe(p, ctx, 2, 12) == 0);
    CHECK(wubu_ngram_map_observe(p, ctx, 2, 12) == 0);
    CHECK(wubu_ngram_map_observe(p, ctx, 2, 13) == 0);
    CHECK(wubu_ngram_map_observe(p, ctx, 2, 14) == 0);
    CHECK(wubu_ngram_map_observe(p, ctx, 2, 14) == 0);
    CHECK(wubu_ngram_map_observe(p, ctx, 2, 15) == 0);
    CHECK(wubu_ngram_map_observe(p, ctx, 2, 16) == 0);

    n = wubu_ngram_map_values(p, ctx, 2, vals, counts);
    CHECK(n == 4);
    CHECK(vals[0] == 12 && counts[0] == 3);
    CHECK(vals[1] == 14 && counts[1] == 2);
    CHECK(vals[2] == 15 && counts[2] == 1);
    CHECK(vals[3] == 16 && counts[3] == 1);
    printf("EG-72 k4v values=[%d %d %d %d] counts=[%d %d %d %d] (evicted 13)\n",
           vals[0], vals[1], vals[2], vals[3],
           counts[0], counts[1], counts[2], counts[3]);

    /* Persisted form round-trips values and counts; merge adds hit counts. */
    {
        unsigned char buf[512];
        wubu_ngram_map_t *q = NULL, *r = wubu_ngram_map_create(8, 2);
        int v2[4], c2[4], rv[4], rc[4];
        size_t sz = wubu_ngram_map_encoded_size(p);
        CHECK(r && sz <= sizeof(buf));
        CHECK(wubu_ngram_map_encode(p, buf, sizeof(buf)) == (int)sz);
        CHECK(wubu_ngram_map_decode(buf, sz, &q) == 0);
        CHECK(q != NULL);
        CHECK(wubu_ngram_map_values(q, ctx, 2, v2, c2) == 4);
        CHECK(v2[0] == 12 && c2[0] == 3 && v2[1] == 14 && c2[1] == 2);
        CHECK(wubu_ngram_map_observe(r, ctx, 2, 12) == 0);
        CHECK(wubu_ngram_map_observe(r, ctx, 2, 12) == 0);
        CHECK(wubu_ngram_map_merge(p, r) == 0);
        CHECK(wubu_ngram_map_values(p, ctx, 2, rv, rc) == 4);
        CHECK(rv[0] == 12 && rc[0] == 5);
        printf("EG-72 persist=%zu bytes round-trip ok; merge 12 count 3->%d\n", sz, rc[0]);
        CHECK(wubu_ngram_map_create(0, 2) == NULL);
        CHECK(wubu_ngram_map_create(8, 0) == NULL);
        CHECK(wubu_ngram_map_observe(NULL, ctx, 2, 12) == -1);
        CHECK(wubu_ngram_map_values(NULL, ctx, 2, rv, rc) == -1);
        CHECK(wubu_ngram_map_merge(p, NULL) == -1);
        CHECK(wubu_ngram_map_decode(buf, 4, &q) == -1);
        wubu_ngram_map_free(q);
        wubu_ngram_map_free(r);
    }

    wubu_ngram_map_free(p);
    if (failures) { fprintf(stderr, "test_eg72_ngram_map: FAIL (%d)\n", failures); return 1; }
    printf("test_eg72_ngram_map: PASS\n");
    return 0;
}
