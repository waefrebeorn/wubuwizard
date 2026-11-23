/* test_eg45_expert_stage.c -- EG-45c gate: coordinated expert staging. */
#include <stdio.h>

#include "wubu_expert_stage.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    int bytes[3] = {1000, 1000, 1000};
    double compute[3] = {2.0, 2.0, 2.0};
    double stall = -1.0;
    CHECK(wubu_expert_stage_plan(bytes, compute, 3, 1000.0, &stall) == 0);
    CHECK(stall < 1e-9);
    CHECK(wubu_expert_stage_overlapped_bytes(bytes, 3) == 2000);
    printf("EG-45c fast stage stall=%.3fs overlapped=2000 bytes\n", stall);

    CHECK(wubu_expert_stage_plan(bytes, compute, 3, 250.0, &stall) == 0);
    CHECK(stall > 3.9 && stall < 4.1);
    printf("EG-45c slow stage stall=%.3fs (2 handoffs)\n", stall);

    CHECK(wubu_expert_stage_plan(NULL, compute, 3, 1000.0, &stall) == -1);
    CHECK(wubu_expert_stage_plan(bytes, compute, 0, 1000.0, &stall) == -1);
    CHECK(wubu_expert_stage_plan(bytes, compute, 3, 0.0, &stall) == -1);
    CHECK(wubu_expert_stage_plan(bytes, compute, 3, 1000.0, NULL) == -1);
    CHECK(wubu_expert_stage_overlapped_bytes(NULL, 3) == 0);

    if (failures) { fprintf(stderr, "test_eg45_expert_stage: FAIL (%d)\n", failures); return 1; }
    printf("test_eg45_expert_stage: PASS\n");
    return 0;
}
