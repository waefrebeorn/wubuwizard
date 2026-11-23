/* test_eg103_zero_logits.c -- EG-103 gate: sampled-token-only D2H handoff. */
#include <stdio.h>
#include <math.h>

#include "wubu_zero_logits.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define VOCAB 8
#define ROWS 2

int main(void) {
    float logits[ROWS*VOCAB] = {
        1.0f, 3.0f, 0.5f, -1.0f, 2.0f, 0.0f, -2.0f, 4.0f,
        0.1f, -0.5f, 2.5f, 1.0f, 0.3f, -1.2f, 3.5f, 0.7f
    };
    int ids[ROWS] = { 7, 6 };
    float lp[ROWS], bad[ROWS];
    wubu_zero_logits_plan_t plan;
    int i;

    CHECK(wubu_zero_logits_plan(32000, 16, &plan) == 0);
    printf("EG-103 full=%zu bytes handoff=%zu bytes ratio=%.8f\n",
           plan.full_bytes, plan.handoff_bytes, plan.reduction);
    CHECK(plan.full_bytes == 2048000u && plan.handoff_bytes == 128u);
    CHECK(plan.reduction < 0.001);

    for (i = 0; i < ROWS; i++) {
        const float *row = logits + i*VOCAB;
        float mx = row[0], denom = 0.0f;
        for (int v = 1; v < VOCAB; v++) if (row[v] > mx) mx = row[v];
        for (int v = 0; v < VOCAB; v++) denom += expf(row[v] - mx);
        lp[i] = row[ids[i]] - mx - logf(denom);
    }
    CHECK(wubu_zero_logits_verify(logits, VOCAB, ids, lp, ROWS, 1e-5f) == 0);
    printf("EG-103 verify lp=[%.5f %.5f]\n", lp[0], lp[1]);

    bad[0] = lp[0] + 0.5f; bad[1] = lp[1];
    CHECK(wubu_zero_logits_verify(logits, VOCAB, ids, bad, ROWS, 1e-5f) == -1);
    {
        int wrong[ROWS] = { 7, 99 };
        CHECK(wubu_zero_logits_verify(logits, VOCAB, wrong, lp, ROWS, 1e-5f) == -1);
    }

    CHECK(wubu_zero_logits_plan(0, 16, &plan) == -1);
    CHECK(wubu_zero_logits_plan(8, 0, &plan) == -1);
    CHECK(wubu_zero_logits_verify(NULL, VOCAB, ids, lp, ROWS, 1e-5f) == -1);
    CHECK(wubu_zero_logits_verify(logits, VOCAB, ids, lp, 0, 1e-5f) == -1);

    if (failures) {
        fprintf(stderr, "test_eg103_zero_logits: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg103_zero_logits: PASS\n");
    return 0;
}
