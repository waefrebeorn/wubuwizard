/* test_eg87_ev_depth.c -- EG-87 gate: expected-value speculative depth. */
#include <stdio.h>
#include <math.h>

#include "wubu_ev_depth.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_ev_depth_t *p = wubu_ev_depth_create(3, 20.0f, 0.005f);
    CHECK(p != NULL);

    /* Warmup: four full-depth cycles regardless of acceptance evidence. */
    for (int i = 0; i < 4; i++) CHECK(wubu_ev_depth_decide(p) == 3);

    /* Configured priors: P1=.92, P2=.5888, P3=.1884; cost threshold .11. */
    CHECK(fabs(wubu_ev_depth_extra_prob(p, 1) - 0.92f) < 1e-3);
    CHECK(fabs(wubu_ev_depth_extra_prob(p, 2) - 0.5888f) < 1e-3);
    CHECK(fabs(wubu_ev_depth_extra_prob(p, 3) - 0.1884f) < 1e-3);
    CHECK(wubu_ev_depth_decide(p) == 3);
    printf("EG-87 warmup=4 priors P1=%.4f P2=%.4f P3=%.4f threshold=%.3f depth=3\n",
           wubu_ev_depth_extra_prob(p, 1), wubu_ev_depth_extra_prob(p, 2),
           wubu_ev_depth_extra_prob(p, 3), 20.0f * 0.005f * 1.10f);

    /* 20 immediate rejects collapse the depth-1 EWMA below the 0.18 floor. */
    for (int i = 0; i < 20; i++) CHECK(wubu_ev_depth_observe(p, 0, 0.0f, 0.0f) == 0);
    CHECK(wubu_ev_depth_extra_prob(p, 1) < 0.2f);
    CHECK(wubu_ev_depth_decide(p) == 0);
    printf("EG-87 after 20 rejects P1=%.4f depth=0\n", wubu_ev_depth_extra_prob(p, 1));

    /* 20 full accepts restore the EWMAs and full depth. */
    for (int i = 0; i < 20; i++) CHECK(wubu_ev_depth_observe(p, 3, 0.0f, 0.0f) == 0);
    CHECK(wubu_ev_depth_decide(p) == 3);

    /* Floor holds even with a zero configured cost. */
    {
        wubu_ev_depth_t *q = wubu_ev_depth_create(3, 20.0f, 0.0f);
        CHECK(q != NULL);
        for (int i = 0; i < 4; i++) CHECK(wubu_ev_depth_decide(q) == 3);
        for (int i = 0; i < 20; i++) CHECK(wubu_ev_depth_observe(q, 0, 0.0f, 0.0f) == 0);
        CHECK(wubu_ev_depth_decide(q) == 0);
        wubu_ev_depth_free(q);
    }

    /* Observed draft+verify cost replaces the configured cost after 4 samples. */
    {
        wubu_ev_depth_t *r = wubu_ev_depth_create(3, 20.0f, 0.0f);
        CHECK(r != NULL);
        for (int i = 0; i < 4; i++)
            CHECK(wubu_ev_depth_observe(r, 3, 0.05f, 0.10f) == 0);
        CHECK(wubu_ev_depth_cost_source(r) == 1);
        CHECK(wubu_ev_depth_decide(r) == 0); /* 0.05s*20tok/s*1.1 = 1.1 */
        printf("EG-87 observed cost source=%d extra=0.050 depth=0\n",
               wubu_ev_depth_cost_source(r));
        wubu_ev_depth_free(r);
    }

    /* Every 32nd cycle probes full depth even with collapsed EWMAs. */
    {
        wubu_ev_depth_t *s = wubu_ev_depth_create(3, 20.0f, 0.005f);
        CHECK(s != NULL);
        for (int i = 0; i < 32; i++) CHECK(wubu_ev_depth_observe(s, 0, 0.0f, 0.0f) == 0);
        CHECK(wubu_ev_depth_decide(s) == 3);
        CHECK(wubu_ev_depth_observe(s, 0, 0.0f, 0.0f) == 0);
        CHECK(wubu_ev_depth_decide(s) == 0);
        printf("EG-87 probe at cycle 32 -> depth=3, next cycle -> depth=0\n");
        wubu_ev_depth_free(s);
    }

    /* Confidence factor scales the expected probability. */
    wubu_ev_depth_set_confidence(p, 0.1f);
    CHECK(wubu_ev_depth_decide(p) == 0);
    wubu_ev_depth_set_confidence(p, 2.0f);
    CHECK(wubu_ev_depth_decide(p) == 3); /* clamped to 1.0 */

    /* Bad input is rejected by status, not by crash. */
    CHECK(wubu_ev_depth_create(0, 20.0f, 0.0f) == NULL);
    CHECK(wubu_ev_depth_create(5, 20.0f, 0.0f) == NULL);
    CHECK(wubu_ev_depth_create(3, 0.0f, 0.0f) == NULL);
    CHECK(wubu_ev_depth_observe(p, 4, 0.0f, 0.0f) == -1);
    CHECK(wubu_ev_depth_observe(p, 3, -1.0f, 0.0f) == -1);
    CHECK(wubu_ev_depth_decide(NULL) == -1);
    CHECK(wubu_ev_depth_extra_prob(p, 0) == -1.0f);
    CHECK(wubu_ev_depth_extra_prob(p, 4) == -1.0f);

    wubu_ev_depth_free(p);
    if (failures) { fprintf(stderr, "test_eg87_ev_depth: FAIL (%d)\n", failures); return 1; }
    printf("test_eg87_ev_depth: PASS\n");
    return 0;
}
