/* test_eg39_surgery.c -- EG-39b gate: model surgery under a memory budget. */
#include <stdio.h>

#include "wubu_model_surgery.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_surgery_layer_t layers[4] = {
        {400, 10.0}, {300, 1.0}, {200, 2.0}, {100, 8.0}
    };
    int keep[4];
    int rc = wubu_surgery_plan(layers, 4, 700, 0.10, keep);
    CHECK(rc == 0);
    CHECK(keep[0] == 1 && keep[1] == 0 && keep[2] == 1 && keep[3] == 1);
    CHECK(wubu_surgery_bytes_saved(layers, keep, 4) == 300);
    CHECK(wubu_surgery_remaining_bytes(layers, keep, 4) == 700);
    CHECK(wubu_surgery_loss_fraction(layers, keep, 4) < 0.10);
    printf("EG-39b keep=[%d %d %d %d] saved=300 remaining=700 loss=%.4f\n",
           keep[0], keep[1], keep[2], keep[3],
           wubu_surgery_loss_fraction(layers, keep, 4));

    /* Tight loss cap forbids the same cut. */
    CHECK(wubu_surgery_plan(layers, 4, 700, 0.01, keep) == -1);
    CHECK(wubu_surgery_plan(NULL, 4, 700, 0.10, keep) == -1);
    CHECK(wubu_surgery_plan(layers, 4, 700, 0.10, NULL) == -1);
    CHECK(wubu_surgery_plan(layers, 0, 700, 0.10, keep) == -1);
    CHECK(wubu_surgery_plan(layers, 4, 0, 0.10, keep) == -1);
    CHECK(wubu_surgery_plan(layers, 4, 700, 1.1, keep) == -1);
    {
        wubu_surgery_layer_t bad[1] = {{0, 1.0}};
        CHECK(wubu_surgery_plan(bad, 1, 700, 0.10, keep) == -1);
    }
    CHECK(wubu_surgery_bytes_saved(layers, NULL, 4) == 0);
    CHECK(wubu_surgery_loss_fraction(layers, NULL, 4) < 0.0);
    printf("EG-39b infeasible budgets and invalid layers rejected\n");

    if (failures) { fprintf(stderr, "test_eg39_surgery: FAIL (%d)\n", failures); return 1; }
    printf("test_eg39_surgery: PASS\n");
    return 0;
}
