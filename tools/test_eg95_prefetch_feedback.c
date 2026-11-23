/* test_eg95_prefetch_feedback.c -- EG-95 gate: closed-loop prefetch depth. */
#include <stdio.h>
#include <math.h>

#include "wubu_prefetch_feedback.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

/* window(consumed, useful, late, evictions, skips, latency_us, peak, budget) */
int main(void) {
    wubu_prefetch_feedback_t *p = wubu_prefetch_feedback_create(3);
    CHECK(p != NULL);
    CHECK(wubu_prefetch_feedback_depth(p) == 3);

    /* Low usefulness drops depth by one. */
    CHECK(wubu_prefetch_feedback_observe(p, 10, 5, 0, 0, 0, 100.0, 400.0, 1000.0) == 0);
    CHECK(wubu_prefetch_feedback_depth(p) == 2);
    printf("EG-95 low usefulness (5/10) depth 3 -> %d\n",
           wubu_prefetch_feedback_depth(p));

    /* Budget skips drop again. */
    CHECK(wubu_prefetch_feedback_observe(p, 10, 10, 1, 0, 2, 101.0, 400.0, 1000.0) == 0);
    CHECK(wubu_prefetch_feedback_depth(p) == 1);
    printf("EG-95 budget_skips=2 depth 2 -> %d\n",
           wubu_prefetch_feedback_depth(p));

    wubu_prefetch_feedback_free(p);

    /* Two late, stable, highly-useful windows with free budget start a probe. */
    {
        wubu_prefetch_feedback_t *q = wubu_prefetch_feedback_create(2);
        CHECK(q != NULL);
        for (int i = 0; i < 2; i++)
            CHECK(wubu_prefetch_feedback_observe(q, 10, 10, 1, 0, 0,
                                                 100.0, 500.0, 1000.0) == 0);
        CHECK(wubu_prefetch_feedback_depth(q) == 3);
        CHECK(wubu_prefetch_feedback_probe_active(q) == 1);
        /* 0.2% latency drift is within the 0.5% probe tolerance. */
        CHECK(wubu_prefetch_feedback_observe(q, 10, 10, 0, 0, 0,
                                             100.2, 500.0, 1000.0) == 0);
        CHECK(wubu_prefetch_feedback_depth(q) == 3);
        CHECK(wubu_prefetch_feedback_probe_active(q) == 0);
        CHECK(wubu_prefetch_feedback_rejections(q) == 0);
        printf("EG-95 probe 2->3 accepted, depth=%d rejections=0\n",
               wubu_prefetch_feedback_depth(q));
        wubu_prefetch_feedback_free(q);
    }

    /* A regressing probe is rejected, remembered, then reopened by phase change. */
    {
        wubu_prefetch_feedback_t *r = wubu_prefetch_feedback_create(2);
        CHECK(r != NULL);
        for (int i = 0; i < 2; i++)
            CHECK(wubu_prefetch_feedback_observe(r, 10, 10, 1, 0, 0,
                                                 100.0, 500.0, 1000.0) == 0);
        CHECK(wubu_prefetch_feedback_depth(r) == 3);
        CHECK(wubu_prefetch_feedback_observe(r, 10, 10, 0, 0, 0,
                                             110.0, 500.0, 1000.0) == 0);
        CHECK(wubu_prefetch_feedback_depth(r) == 2);
        CHECK(wubu_prefetch_feedback_rejections(r) == 1);
        CHECK(wubu_prefetch_feedback_rejected(r, 2) == 1);
        /* Four stable windows: cooldown first, then remembered rejection. */
        for (int i = 0; i < 4; i++)
            CHECK(wubu_prefetch_feedback_observe(r, 10, 10, 1, 0, 0,
                                                 110.0, 500.0, 1000.0) == 0);
        CHECK(wubu_prefetch_feedback_depth(r) == 2);
        printf("EG-95 rejected 2->3 remembered, depth stays %d\n",
               wubu_prefetch_feedback_depth(r));
        /* >5% baseline shift is a phase change: memory clears, probe reopens. */
        CHECK(wubu_prefetch_feedback_observe(r, 10, 10, 0, 0, 0,
                                             130.0, 500.0, 1000.0) == 0);
        CHECK(wubu_prefetch_feedback_rejected(r, 2) == 0);
        for (int i = 0; i < 2; i++)
            CHECK(wubu_prefetch_feedback_observe(r, 10, 10, 1, 0, 0,
                                                 130.0, 500.0, 1000.0) == 0);
        CHECK(wubu_prefetch_feedback_depth(r) == 3);
        printf("EG-95 phase change 110->130us reopens probe, depth=%d\n",
               wubu_prefetch_feedback_depth(r));
        wubu_prefetch_feedback_free(r);
    }

    /* Sequential-stream evictions at high usefulness are not pressure. */
    {
        wubu_prefetch_feedback_t *s = wubu_prefetch_feedback_create(2);
        CHECK(s != NULL);
        CHECK(wubu_prefetch_feedback_observe(s, 10, 10, 0, 3, 0,
                                             100.0, 500.0, 1000.0) == 0);
        CHECK(wubu_prefetch_feedback_depth(s) == 2);
        wubu_prefetch_feedback_free(s);
    }

    /* Invalid telemetry and buffers are rejected. */
    {
        wubu_prefetch_feedback_t *t = wubu_prefetch_feedback_create(2);
        CHECK(wubu_prefetch_feedback_create(0) == NULL);
        CHECK(wubu_prefetch_feedback_create(5) == NULL);
        CHECK(wubu_prefetch_feedback_observe(t, 10, 11, 0, 0, 0,
                                             100.0, 500.0, 1000.0) == -1);
        CHECK(wubu_prefetch_feedback_observe(t, 10, 10, 0, 0, 0,
                                             -1.0, 500.0, 1000.0) == -1);
        CHECK(wubu_prefetch_feedback_observe(t, 10, 10, 0, 0, 0,
                                             100.0, 500.0, 0.0) == -1);
        CHECK(wubu_prefetch_feedback_observe(NULL, 10, 10, 0, 0, 0,
                                             100.0, 500.0, 1000.0) == -1);
        CHECK(wubu_prefetch_feedback_depth(NULL) == -1);
        CHECK(wubu_prefetch_feedback_rejected(NULL, 2) == -1);
        CHECK(wubu_prefetch_feedback_rejected(t, 0) == -1);
        wubu_prefetch_feedback_free(t);
    }

    if (failures) { fprintf(stderr, "test_eg95_prefetch_feedback: FAIL (%d)\n", failures); return 1; }
    printf("test_eg95_prefetch_feedback: PASS\n");
    return 0;
}
