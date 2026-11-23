/* test_eg101_sq_agreement.c -- EG-101 gate: SQ build acceptance contract. */
#include <stdio.h>
#include <math.h>

#include "wubu_sq_agreement.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    float ref[6] = { 0.7f, 0.2f, 0.1f,  0.6f, 0.3f, 0.1f };
    float close[6] = { 0.68f, 0.21f, 0.11f,  0.59f, 0.31f, 0.10f };
    float flipped[6] = { 0.2f, 0.7f, 0.1f,  0.1f, 0.2f, 0.7f };
    wubu_sq_report_t r;

    CHECK(wubu_sq_evaluate(ref, close, 2, 3, 0.5f, 0.01f, &r) == 0);
    printf("EG-101 close agreement=%.2f kl=%.6f exact=%d accepted=%d\n",
           r.top1_agreement, r.kl, r.bit_exact, r.accepted);
    CHECK(r.top1_agreement == 1.0f && r.kl < 0.01f && r.bit_exact == 0);
    CHECK(r.accepted == 1);

    CHECK(wubu_sq_evaluate(ref, ref, 2, 3, 0.99f, 0.0f, &r) == 0);
    printf("EG-101 exact agreement=%.2f kl=%.6f exact=%d accepted=%d\n",
           r.top1_agreement, r.kl, r.bit_exact, r.accepted);
    CHECK(r.bit_exact == 1 && fabsf(r.kl) < 1e-6f && r.accepted == 1);

    CHECK(wubu_sq_evaluate(ref, flipped, 2, 3, 0.5f, 1.0f, &r) == 0);
    printf("EG-101 flipped agreement=%.2f kl=%.6f accepted=%d\n",
           r.top1_agreement, r.kl, r.accepted);
    CHECK(r.top1_agreement == 0.0f && r.accepted == 0);

    CHECK(wubu_sq_evaluate(NULL, close, 2, 3, 0.5f, 0.01f, &r) == -1);
    CHECK(wubu_sq_evaluate(ref, close, 0, 3, 0.5f, 0.01f, &r) == -1);
    CHECK(wubu_sq_evaluate(ref, close, 2, 0, 0.5f, 0.01f, &r) == -1);
    CHECK(wubu_sq_evaluate(ref, close, 2, 3, 2.0f, 0.01f, &r) == -1);
    CHECK(wubu_sq_evaluate(ref, close, 2, 3, 0.5f, -1.0f, &r) == -1);

    if (failures) {
        fprintf(stderr, "test_eg101_sq_agreement: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg101_sq_agreement: PASS\n");
    return 0;
}
