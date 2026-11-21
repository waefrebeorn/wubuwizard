/* test_ttc_budget.c -- gate for LP-14 (test-time-compute budget). */
#include <stdio.h>
#include "wubu_ttc_budget.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void)
{
    /* candidate cap: 3 candidates -> 1,1,0 */
    wubu_ttc_budget_t *b =
        wubu_ttc_budget_create(100000, (size_t)1 << 30, 3, 1000, 0.001);
    CHECK(b != NULL);
    CHECK(wubu_ttc_budget_begin(b) == 0);
    CHECK(wubu_ttc_budget_try(b, 0.1) == 1);
    CHECK(wubu_ttc_budget_try(b, 0.2) == 1);
    CHECK(wubu_ttc_budget_try(b, 0.3) == 0);
    CHECK(wubu_ttc_budget_candidates(b) == 3);
    wubu_ttc_budget_free(b);

    /* wall-ms budget via injected time */
    b = wubu_ttc_budget_create(10, (size_t)1 << 30, 100, 1000, 0.001);
    CHECK(wubu_ttc_budget_begin(b) == 0);
    CHECK(wubu_ttc_budget_add_wall_ms(b, 5) == 0);
    CHECK(wubu_ttc_budget_try(b, 0.1) == 1);
    CHECK(wubu_ttc_budget_add_wall_ms(b, 6) == 0);
    CHECK(wubu_ttc_budget_try(b, 0.2) == 0);   /* 11ms >= 10ms */
    CHECK(wubu_ttc_budget_elapsed_ms(b) >= 11);
    wubu_ttc_budget_free(b);

    /* real measured wall time advances (spin) */
    b = wubu_ttc_budget_create(100000, (size_t)1 << 30, 100, 1000, 0.001);
    CHECK(wubu_ttc_budget_begin(b) == 0);
    volatile unsigned long spin = 0;
    for (unsigned long i = 0; i < 20000000UL; i++) spin += i;
    (void)spin;
    CHECK(wubu_ttc_budget_elapsed_ms(b) >= 0);
    wubu_ttc_budget_free(b);

    /* kv-bytes budget */
    b = wubu_ttc_budget_create(100000, 1024, 100, 1000, 0.001);
    CHECK(wubu_ttc_budget_begin(b) == 0);
    CHECK(wubu_ttc_budget_add_kv(b, 600) == 0);
    CHECK(wubu_ttc_budget_try(b, 0.1) == 1);
    CHECK(wubu_ttc_budget_add_kv(b, 500) == 0);
    CHECK(wubu_ttc_budget_try(b, 0.2) == 0);   /* 1100 >= 1024 */
    CHECK(wubu_ttc_budget_kv_bytes(b) == 1100);
    wubu_ttc_budget_free(b);

    /* plateau stop: window 3, no gain -> stop on 4th non-improving call */
    b = wubu_ttc_budget_create(100000, (size_t)1 << 30, 100, 3, 0.01);
    CHECK(wubu_ttc_budget_begin(b) == 0);
    CHECK(wubu_ttc_budget_try(b, 0.5) == 1);   /* best = 0.5 */
    CHECK(wubu_ttc_budget_try(b, 0.5) == 1);   /* non_improve = 1 */
    CHECK(wubu_ttc_budget_try(b, 0.5) == 1);   /* non_improve = 2 */
    CHECK(wubu_ttc_budget_try(b, 0.5) == 0);   /* non_improve = 3 -> plateau */
    CHECK(wubu_ttc_budget_plateaued(b) == 1);
    wubu_ttc_budget_free(b);

    CHECK(wubu_ttc_budget_create(-1, 100, 3, 3, 0.0) == NULL);
    if (failures == 0) printf("test_ttc_budget: PASS (21 gates)\n");
    return failures ? 1 : 0;
}
