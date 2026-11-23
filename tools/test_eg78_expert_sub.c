/* test_eg78_expert_sub.c -- EG-78 gate: constant-vector novice experts. */
#include <stdio.h>

#include "wubu_expert_sub.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_expert_sub_t *p = wubu_expert_sub_create(8, 4, 1024);
    float c[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    CHECK(p != NULL);

    /* Replacing two experts frees their whole MLP allocation. */
    CHECK(wubu_expert_sub_mark(p, 3, c) == 1024);
    CHECK(wubu_expert_sub_mark(p, 5, c) == 1024);
    CHECK(wubu_expert_sub_bytes_saved(p) == 2048);
    CHECK(wubu_expert_sub_is_substituted(p, 3) == 1);
    CHECK(wubu_expert_sub_is_substituted(p, 1) == 0);
    printf("EG-78 substituted=2 bytes_saved=%zu (zero MLP bytes)\n",
           wubu_expert_sub_bytes_saved(p));

    /* Constant vectors are added with unchanged router weights and ids. */
    {
        double weights[8] = {0, 0, 0, 0.3, 0, 0.2, 0, 0};
        int sel[3] = {1, 3, 5};
        float out[4];
        CHECK(wubu_expert_sub_contrib(p, weights, sel, 3, out) == 0);
        CHECK(out[0] == 0.5f && out[1] == 1.0f && out[2] == 1.5f && out[3] == 2.0f);
        CHECK(sel[0] == 1 && sel[1] == 3 && sel[2] == 5);
        CHECK(weights[3] == 0.3 && weights[5] == 0.2); /* no renorm */
        printf("EG-78 contrib=[%.1f %.1f %.1f %.1f] logical ids preserved\n",
               out[0], out[1], out[2], out[3]);
        CHECK(wubu_expert_sub_contrib(NULL, weights, sel, 3, out) == -1);
        CHECK(wubu_expert_sub_contrib(p, NULL, sel, 3, out) == -1);
        CHECK(wubu_expert_sub_mark(p, 9, c) == 0);
        CHECK(wubu_expert_sub_mark(p, 3, c) == 0);
        CHECK(wubu_expert_sub_is_substituted(NULL, 3) == -1);
        CHECK(wubu_expert_sub_create(0, 4, 1024) == NULL);
    }

    wubu_expert_sub_free(p);
    if (failures) { fprintf(stderr, "test_eg78_expert_sub: FAIL (%d)\n", failures); return 1; }
    printf("test_eg78_expert_sub: PASS\n");
    return 0;
}
