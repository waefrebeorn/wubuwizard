/* test_eg71_ngram_pool.c -- EG-71 gate: shared rolling-hash n-gram pool. */
#include <stdio.h>

#include "wubu_ngram_pool.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_ngram_pool_t *p = wubu_ngram_pool_create(8, 2, 0.75f, 2);
    int ctx1[2] = {10, 11}, ctx2[2] = {11, 10};
    CHECK(p != NULL);

    CHECK(wubu_ngram_pool_add(p, ctx1, 2, 12) == 0);
    CHECK(wubu_ngram_pool_get(p, ctx1, 2) == 12);
    CHECK(wubu_ngram_pool_get(p, ctx2, 2) == -1);
    CHECK(wubu_ngram_pool_add(p, ctx2, 2, 99) == 0);
    CHECK(wubu_ngram_pool_get(p, ctx2, 2) == 99);
    printf("EG-71 rolling hash: [10 11]->12, [11 10]->99, entries=%d\n",
           wubu_ngram_pool_entries(p));

    /* Occupancy above 75% clears the shared table. */
    {
        wubu_ngram_pool_t *q = wubu_ngram_pool_create(8, 2, 0.75f, 2);
        int ctx[2] = {10, 11};
        CHECK(q != NULL);
        CHECK(wubu_ngram_pool_add(q, ctx, 2, 12) == 0);
        for (int i = 0; i < 6; ++i) {
            int k[2] = {100 + i, 200 + i};
            CHECK(wubu_ngram_pool_add(q, k, 2, 300 + i) == 0);
        }
        CHECK(wubu_ngram_pool_resets(q) == 1);
        CHECK(wubu_ngram_pool_entries(q) <= 6);
        CHECK(wubu_ngram_pool_get(q, ctx, 2) == -1);
        printf("EG-71 occupancy reset=%d entries=%d after 7 adds\n",
               wubu_ngram_pool_resets(q), wubu_ngram_pool_entries(q));
        wubu_ngram_pool_free(q);
    }

    /* Two consecutive rejections reset the pool; an accept clears the streak. */
    {
        wubu_ngram_pool_t *r = wubu_ngram_pool_create(8, 2, 0.75f, 2);
        int ctx[2] = {10, 11};
        CHECK(r != NULL);
        CHECK(wubu_ngram_pool_add(r, ctx, 2, 12) == 0);
        CHECK(wubu_ngram_pool_observe(r, 0) == 0);
        CHECK(wubu_ngram_pool_observe(r, 1) == 0);
        CHECK(wubu_ngram_pool_get(r, ctx, 2) == 12);
        CHECK(wubu_ngram_pool_observe(r, 0) == 0);
        CHECK(wubu_ngram_pool_observe(r, 0) == 0);
        CHECK(wubu_ngram_pool_resets(r) == 1);
        CHECK(wubu_ngram_pool_entries(r) == 0);
        CHECK(wubu_ngram_pool_get(r, ctx, 2) == -1);
        printf("EG-71 low-acceptance reset=%d entries=0\n", wubu_ngram_pool_resets(r));
        CHECK(wubu_ngram_pool_create(1, 2, 0.75f, 2) == NULL);
        CHECK(wubu_ngram_pool_create(8, 0, 0.75f, 2) == NULL);
        CHECK(wubu_ngram_pool_create(8, 2, 0.0f, 2) == NULL);
        CHECK(wubu_ngram_pool_add(NULL, ctx, 2, 12) == -1);
        CHECK(wubu_ngram_pool_get(NULL, ctx, 2) == -1);
        CHECK(wubu_ngram_pool_observe(NULL, 0) == -1);
        CHECK(wubu_ngram_pool_observe(r, 2) == -1);
        wubu_ngram_pool_free(r);
    }

    wubu_ngram_pool_free(p);
    if (failures) { fprintf(stderr, "test_eg71_ngram_pool: FAIL (%d)\n", failures); return 1; }
    printf("test_eg71_ngram_pool: PASS\n");
    return 0;
}
