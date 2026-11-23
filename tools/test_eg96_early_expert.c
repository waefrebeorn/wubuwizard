/* test_eg96_early_expert.c -- EG-96 gate: lossless cross-layer expert pread. */
#include <stdio.h>

#include "wubu_early_expert.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_early_expert_t *p = wubu_early_expert_create(2, 4);
    int ids[4] = {3, 7, 2, 9};
    double scores[4] = {0.9, 0.8, 0.7, 0.1};
    unsigned char cached[16] = {0};
    int staged;
    CHECK(p != NULL);
    cached[3] = 1;                       /* expert 3 already in the slot pool */

    /* Highest-scoring uncached experts are staged outside the slot pool. */
    staged = wubu_early_expert_stage(p, ids, scores, 4, cached, 16);
    CHECK(staged == 2);
    CHECK(wubu_early_expert_staged(p) == 2);
    CHECK(wubu_early_expert_is_staged(p, 7) == 1);
    CHECK(wubu_early_expert_is_staged(p, 2) == 1);
    CHECK(wubu_early_expert_is_staged(p, 3) == 0); /* cached, not staged */
    CHECK(wubu_early_expert_is_staged(p, 9) == 0); /* low score */
    printf("EG-96 staged=%d experts=[7 2] budget=2 cached_expert=3 skipped\n", staged);

    /* A real plan with misses adopts staged buffers; the plan is unchanged. */
    {
        int plan[3] = {7, 5, 2}, out[3], got = 0;
        unsigned char used[3] = {0};
        CHECK(wubu_early_expert_adopt(p, plan, 3, out, used) == 2);
        CHECK(out[0] == 7 && out[1] == 5 && out[2] == 2);
        CHECK(used[0] == 1 && used[1] == 0 && used[2] == 1);
        CHECK(wubu_early_expert_adopted(p) == 2);
        CHECK(wubu_early_expert_wasted(p) == 0);
        got = wubu_early_expert_adopted(p);
        printf("EG-96 adopted=%d plan unchanged, normal read for expert 5\n", got);
    }

    /* No plan hit means both staged reads were wasted. */
    {
        wubu_early_expert_t *q = wubu_early_expert_create(2, 4);
        int ids[2] = {7, 2}, plan[2] = {4, 5}, out[2];
        double scores[2] = {0.8, 0.7};
        unsigned char cached[8] = {0}, used[2] = {0};
        CHECK(q != NULL);
        CHECK(wubu_early_expert_stage(q, ids, scores, 2, cached, 8) == 2);
        CHECK(wubu_early_expert_adopt(q, plan, 2, out, used) == 0);
        CHECK(wubu_early_expert_wasted(q) == 2);
        CHECK(out[0] == 4 && out[1] == 5);
        CHECK(wubu_early_expert_adopt(NULL, plan, 2, out, used) == -1);
        CHECK(wubu_early_expert_adopt(q, NULL, 2, out, used) == -1);
        CHECK(wubu_early_expert_create(0, 4) == NULL);
        CHECK(wubu_early_expert_create(2, 0) == NULL);
        printf("EG-96 all misses: adopted=0 wasted=2\n");
        wubu_early_expert_free(q);
    }

    wubu_early_expert_free(p);
    if (failures) { fprintf(stderr, "test_eg96_early_expert: FAIL (%d)\n", failures); return 1; }
    printf("test_eg96_early_expert: PASS\n");
    return 0;
}
