/* test_eg94_accept_table.c -- EG-94 gate: empirical tree-ordering table. */
#include <stdio.h>
#include <math.h>

#include "wubu_accept_table.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_accept_table_t *p = wubu_accept_table_create();
    CHECK(p != NULL);

    /* Slot 2 / rank 1 / p~0.55 accepts 9 of 10; another cell accepts 1 of 10. */
    for (int i = 0; i < 10; ++i) {
        CHECK(wubu_accept_table_observe(p, 2, 1, 0.55, i < 9) == 0);
        CHECK(wubu_accept_table_observe(p, 2, 2, 0.55, i == 0) == 0);
    }
    double good = wubu_accept_table_logq(p, 2, 1, 0.55);
    double bad = wubu_accept_table_logq(p, 2, 2, 0.55);
    CHECK(good > bad);
    printf("EG-94 logq good=%.4f bad=%.4f (9/10 vs 1/10, prior weight 5)\n", good, bad);
    CHECK(good < 0.0 && bad < 0.0);

    /* A calibrated table can flip best-first ordering against raw log p. */
    {
        wubu_accept_table_t *q = wubu_accept_table_create();
        wubu_accept_table_t *empty = wubu_accept_table_create();
        int slots[2] = {2, 2}, ranks[2] = {1, 2};
        double probs[2] = {0.5, 0.3};
        CHECK(q && empty);
        for (int i = 0; i < 10; ++i) {
            CHECK(wubu_accept_table_observe(q, 2, 1, 0.5, 0) == 0);
            CHECK(wubu_accept_table_observe(q, 2, 2, 0.3, 1) == 0);
        }
        CHECK(wubu_accept_table_best(empty, slots, ranks, probs, 2) == 0);
        CHECK(wubu_accept_table_best(q, slots, ranks, probs, 2) == 1);
        printf("EG-94 best-first empty=0 calibrated=1 (table flips ordering)\n");
        CHECK(wubu_accept_table_best(NULL, slots, ranks, probs, 2) == -1);
        CHECK(wubu_accept_table_best(q, slots, ranks, probs, 0) == -1);
        CHECK(wubu_accept_table_best(q, NULL, ranks, probs, 2) == -1);
        CHECK(wubu_accept_table_observe(q, 0, 1, 0.5, 1) == -1);
        CHECK(wubu_accept_table_observe(q, 2, 1, 1.5, 1) == -1);
        wubu_accept_table_free(q);
        wubu_accept_table_free(empty);
    }

    wubu_accept_table_free(p);
    if (failures) { fprintf(stderr, "test_eg94_accept_table: FAIL (%d)\n", failures); return 1; }
    printf("test_eg94_accept_table: PASS\n");
    return 0;
}
