/* test_eg80_fly.c -- EG-80 gate: entropy-gated rejection deferral. */
#include <stdio.h>
#include <math.h>

#include "wubu_fly_defer.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    double uniform[4] = {1.0, 1.0, 1.0, 1.0};
    double peaked[4] = {100.0, 1.0, 1.0, 1.0};
    double e1 = wubu_fly_normalized_entropy(uniform, 4);
    double e2 = wubu_fly_normalized_entropy(peaked, 4);
    CHECK(fabs(e1 - 1.0) < 1e-6);
    CHECK(e2 >= 0.0 && e2 < 1.0);
    printf("EG-80 normalized entropy: uniform=%.6f peaked=%.6f\n", e1, e2);

    /* High-entropy rejection is deferred only if the whole window aligns. */
    {
        wubu_fly_defer_t *p = wubu_fly_defer_create(0.7, 3, 1);
        wubu_fly_defer_t *off = wubu_fly_defer_create(0.7, 3, 0);
        unsigned char aligned[3] = {1, 1, 1}, broken[3] = {1, 0, 1};
        CHECK(p && off);
        CHECK(wubu_fly_defer_decide(p, 1, 0.9, aligned, 3) == 1);
        CHECK(wubu_fly_defer_deferred(p) == 1);
        CHECK(wubu_fly_defer_decide(p, 1, 0.5, aligned, 3) == 0);
        CHECK(wubu_fly_defer_decide(p, 1, 0.9, broken, 3) == 0);
        CHECK(wubu_fly_defer_decide(p, 0, 0.9, aligned, 3) == 0);
        CHECK(wubu_fly_defer_deferred(p) == 1);
        CHECK(wubu_fly_defer_rejections(p) == 2);
        CHECK(wubu_fly_defer_decide(off, 1, 0.9, aligned, 3) == 0);
        printf("EG-80 deferred=%d rejections=%d (opt-in off => no deferral)\n",
               wubu_fly_defer_deferred(p), wubu_fly_defer_rejections(p));
        CHECK(wubu_fly_defer_create(1.5, 3, 1) == NULL);
        CHECK(wubu_fly_defer_create(0.7, 0, 1) == NULL);
        CHECK(wubu_fly_defer_decide(NULL, 1, 0.9, aligned, 3) == -1);
        CHECK(wubu_fly_defer_decide(p, 1, 0.9, NULL, 3) == -1);
        CHECK(wubu_fly_normalized_entropy(NULL, 4) == -1.0);
        CHECK(wubu_fly_normalized_entropy(uniform, 1) == -1.0);
        wubu_fly_defer_free(p);
        wubu_fly_defer_free(off);
    }

    if (failures) { fprintf(stderr, "test_eg80_fly: FAIL (%d)\n", failures); return 1; }
    printf("test_eg80_fly: PASS\n");
    return 0;
}
