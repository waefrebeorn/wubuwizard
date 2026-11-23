/* test_eg32_capacity.c -- EG-32a gate: capacity + micro-batch scheduler. */
#include <stdio.h>

#include "wubu_capacity_sched.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_cap_req_t reqs[5] = {
        { 0, 3, 1 }, { 1, 4, 2 }, { 2, 2, 1 }, { 3, 5, 3 }, { 4, 1, 1 }
    };
    wubu_cap_plan_t p;
    CHECK(wubu_capacity_plan(reqs, 5, 12, 6, &p) == 0);
    printf("EG-32 admitted=%d rejected=%d batches=%d [%d %d]\n",
           p.tokens_admitted, p.tokens_rejected, p.n_batches,
           p.batch_tokens[0], p.batch_tokens[1]);
    CHECK(p.tokens_admitted == 12 && p.tokens_rejected == 3);
    CHECK(p.n_batches == 2 && p.batch_tokens[0] == 6 && p.batch_tokens[1] == 6);
    CHECK(p.accepted[3] == 1 && p.accepted[1] == 1 && p.accepted[4] == 1);
    CHECK(p.accepted[2] == 1 && p.accepted[0] == 0);
    CHECK(p.batch_of[0] == -1);

    /* Tight capacity: highest priority still wins, smaller companion fits. */
    CHECK(wubu_capacity_plan(reqs, 5, 6, 6, &p) == 0);
    printf("EG-32 tight admitted=%d rejected=%d batches=%d\n",
           p.tokens_admitted, p.tokens_rejected, p.n_batches);
    CHECK(p.tokens_admitted == 6 && p.tokens_rejected == 9);
    CHECK(p.accepted[3] == 1 && p.accepted[4] == 1 && p.accepted[1] == 0);

    /* A request larger than the batch ceiling is rejected, never split. */
    {
        wubu_cap_req_t big[1] = { { 0, 7, 9 } };
        CHECK(wubu_capacity_plan(big, 1, 20, 6, &p) == 0);
        CHECK(p.tokens_admitted == 0 && p.tokens_rejected == 7);
        CHECK(p.n_batches == 0 && p.accepted[0] == 0);
    }

    CHECK(wubu_capacity_plan(NULL, 5, 12, 6, &p) == -1);
    CHECK(wubu_capacity_plan(reqs, 0, 12, 6, &p) == -1);
    CHECK(wubu_capacity_plan(reqs, 5, 0, 6, &p) == -1);
    CHECK(wubu_capacity_plan(reqs, 5, 12, 0, &p) == -1);
    CHECK(wubu_capacity_plan(reqs, 5, 12, 6, NULL) == -1);

    if (failures) {
        fprintf(stderr, "test_eg32_capacity: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg32_capacity: PASS\n");
    return 0;
}
