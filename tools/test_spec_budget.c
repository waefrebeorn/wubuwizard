/* test_spec_budget.c -- gate for LP-07 (speculative verification budget). */
#include <stdio.h>
#include "wubu_spec_budget.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

static int sum_of(const int *a, int n)
{
    int s = 0;
    for (int i = 0; i < n; ++i) s += a[i];
    return s;
}

int main(void)
{
    wubu_spec_budget_t *p = wubu_spec_budget_create();
    CHECK(p != NULL);
    if (!p) return 1;

    /* 1. ranking follows cumulative confidence; budget is exact; clamped to
     *    nextn; longer windows never go to a lower-ranked sequence. */
    {
        float c[12] = { .9f,.9f,.9f,  .8f,.8f,.8f,
                        .5f,.5f,.5f,  .1f,.1f,.1f };
        int out[4] = { -1, -1, -1, -1 };
        int order[4];
        CHECK(wubu_spec_budget_plan(p, c, 4, 3, 7, out) == 0);
        CHECK(sum_of(out, 4) == 7);                 /* budget exact */
        CHECK(out[0] == 3 && out[1] == 3 && out[2] == 1 && out[3] == 0);
        CHECK(out[0] <= 3 && out[1] <= 3 && out[2] <= 3 && out[3] <= 3);
        CHECK(wubu_spec_budget_last_order(p, order, 4) == 0);
        CHECK(order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3);
        for (int i = 1; i < 4; ++i)
            CHECK(out[order[i - 1]] >= out[order[i]]);
        CHECK(wubu_spec_budget_cum(p, 0) > wubu_spec_budget_cum(p, 3));
    }

    /* 2. ranking is by confidence, not by index; over-budget request clamps. */
    {
        float c[12] = { .1f,.1f,.1f,  .2f,.2f,.2f,
                        .99f,.99f,.99f,  .7f,.7f,.7f };
        int out[4], order[4];
        CHECK(wubu_spec_budget_plan(p, c, 4, 3, 5, out) == 0);
        CHECK(sum_of(out, 4) == 5);
        CHECK(out[2] == 3 && out[3] == 2 && out[1] == 0 && out[0] == 0);
        CHECK(wubu_spec_budget_last_order(p, order, 4) == 0);
        CHECK(order[0] == 2 && order[1] == 3 && order[2] == 1 && order[3] == 0);
        CHECK(wubu_spec_budget_plan(p, c, 4, 3, 100, out) == 0);
        CHECK(sum_of(out, 4) == 12);                /* capacity clamp */
        CHECK(out[0] == 3 && out[1] == 3 && out[2] == 3 && out[3] == 3);
    }

    /* 3. invalid inputs are refused with a negative status value. */
    {
        float c[3] = { 1.0f, 1.0f, 1.0f };
        int out[2];
        CHECK(wubu_spec_budget_plan(NULL, c, 1, 3, 1, out) == -1);
        CHECK(wubu_spec_budget_plan(p, NULL, 1, 3, 1, out) == -1);
        CHECK(wubu_spec_budget_plan(p, c, 0, 3, 1, out) == -1);
    }

    wubu_spec_budget_free(p);

    if (failures == 0) {
        printf("test_spec_budget: PASS\n");
        printf("LP-07: PASS\n");
        return 0;
    }
    return 1;
}
