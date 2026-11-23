/* test_eg74_selector.c -- EG-74 gate: DFlash2/LiliCorr p-min selector. */
#include <stdio.h>

#include "wubu_candidate_selector.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_candidate_selector_t *p = wubu_candidate_selector_create(8);
    float probs[6] = {0.5f, 0.3f, 0.1f, 0.05f, 0.04f, 0.01f};
    int out[8], n;
    CHECK(p != NULL);

    /* p_min=0.05 keeps candidates 0..3, ranked by probability desc. */
    n = wubu_candidate_selector_select(p, probs, NULL, 6, 0.05f, out);
    CHECK(n == 4);
    CHECK(out[0] == 0 && out[1] == 1 && out[2] == 2 && out[3] == 3);
    printf("EG-74 p_min=0.05 selected=[%d %d %d %d] of 6\n",
           out[0], out[1], out[2], out[3]);

    /* A hidden-state-conditioned score can reorder the survivors. */
    {
        double scores[6] = {0.0, 0.0, 0.5, -0.5, 0.0, 0.0};
        CHECK(wubu_candidate_selector_select(p, probs, scores, 6, 0.05f, out) == 4);
        CHECK(out[0] == 2 && out[1] == 0 && out[2] == 1 && out[3] == 3);
        printf("EG-74 scored selected=[%d %d %d %d] (candidate 2 promoted)\n",
               out[0], out[1], out[2], out[3]);
    }

    /* Even an over-high p_min keeps the single best candidate. */
    CHECK(wubu_candidate_selector_select(p, probs, NULL, 6, 0.9f, out) == 1);
    CHECK(out[0] == 0);
    CHECK(wubu_candidate_selector_select(p, probs, NULL, 6, 0.0f, out) == 6);
    CHECK(wubu_candidate_selector_select(NULL, probs, NULL, 6, 0.1f, out) == -1);
    CHECK(wubu_candidate_selector_select(p, NULL, NULL, 6, 0.1f, out) == -1);
    CHECK(wubu_candidate_selector_select(p, probs, NULL, 0, 0.1f, out) == -1);
    CHECK(wubu_candidate_selector_select(p, probs, NULL, 6, 1.5f, out) == -1);
    CHECK(wubu_candidate_selector_create(0) == NULL);
    CHECK(wubu_candidate_selector_create(65) == NULL);

    wubu_candidate_selector_free(p);
    if (failures) { fprintf(stderr, "test_eg74_selector: FAIL (%d)\n", failures); return 1; }
    printf("test_eg74_selector: PASS\n");
    return 0;
}
