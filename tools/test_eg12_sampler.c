/* test_eg12_sampler.c -- EG-12 gate: ordered sampler chain with transforms
 * and filters. Prints: test_eg12_sampler: PASS. */
#include <stdio.h>
#include <string.h>

#include "wubu_sampler_chain.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

static void set_logits(float *v) { v[0] = 1.0f; v[1] = 2.0f; v[2] = 3.0f; v[3] = 4.0f; }

int main(void) {
    float logits[4];
    unsigned char mask[4];
    int prev[1] = { 3 };

    wubu_sampler_chain_t *c = wubu_sampler_chain_create(8);
    CHECK(c != NULL);
    if (!c) return 1;

    /* top-k=2 keeps only the two largest logits. */
    CHECK(wubu_sampler_chain_add(c, WUBU_SAMPLE_TOP_K, 2.0) == 0);
    set_logits(logits);
    CHECK(wubu_sampler_chain_apply(c, logits, 4, NULL, 0, mask) == 2);
    CHECK(mask[3] && mask[2] && !mask[1] && !mask[0]);
    CHECK(wubu_sampler_chain_pick(c, logits, mask, 4, 0) == 3);
    printf("EG-12 top_k=2 allowed=%d%d%d%d pick=3\n",
           mask[0], mask[1], mask[2], mask[3]);

    /* Repetition penalty before top-k changes the candidate set. */
    wubu_sampler_chain_t *r = wubu_sampler_chain_create(8);
    CHECK(r != NULL);
    CHECK(wubu_sampler_chain_add(r, WUBU_SAMPLE_REPETITION, 4.0) == 0);
    CHECK(wubu_sampler_chain_add(r, WUBU_SAMPLE_TOP_K, 2.0) == 0);
    set_logits(logits);
    CHECK(wubu_sampler_chain_apply(r, logits, 4, prev, 1, mask) == 2);
    CHECK(logits[3] == 1.0f);
    CHECK(mask[2] && mask[1] && !mask[3]);
    printf("EG-12 repetition+top_k: logit3=%.1f allowed=1,2\n", logits[3]);

    /* Temperature transforms logits without filtering. */
    wubu_sampler_chain_t *t = wubu_sampler_chain_create(4);
    CHECK(wubu_sampler_chain_add(t, WUBU_SAMPLE_TEMPERATURE, 0.5) == 0);
    set_logits(logits);
    CHECK(wubu_sampler_chain_apply(t, logits, 4, NULL, 0, mask) == 4);
    CHECK(logits[3] == 8.0f);
    int picked = wubu_sampler_chain_pick(t, logits, mask, 4, 12345);
    CHECK(picked >= 0 && picked < 4 && mask[picked]);
    printf("EG-12 temperature=0.5 logit3=%.1f seeded_pick=%d\n", logits[3], picked);

    /* min-p and top-p are strict filters. */
    wubu_sampler_chain_t *m = wubu_sampler_chain_create(4);
    CHECK(wubu_sampler_chain_add(m, WUBU_SAMPLE_MIN_P, 0.9) == 0);
    set_logits(logits);
    CHECK(wubu_sampler_chain_apply(m, logits, 4, NULL, 0, mask) == 1);
    CHECK(mask[3]);
    wubu_sampler_chain_t *p = wubu_sampler_chain_create(4);
    CHECK(wubu_sampler_chain_add(p, WUBU_SAMPLE_TOP_P, 0.7) == 0);
    set_logits(logits);
    CHECK(wubu_sampler_chain_apply(p, logits, 4, NULL, 0, mask) == 1);
    CHECK(mask[3]);
    printf("EG-12 min_p=0.9 and top_p=0.7 each keep 1 candidate\n");

    /* Chained filters only shrink the candidate set. */
    wubu_sampler_chain_t *both = wubu_sampler_chain_create(4);
    CHECK(wubu_sampler_chain_add(both, WUBU_SAMPLE_TOP_K, 3.0) == 0);
    CHECK(wubu_sampler_chain_add(both, WUBU_SAMPLE_MIN_P, 0.9) == 0);
    set_logits(logits);
    CHECK(wubu_sampler_chain_apply(both, logits, 4, NULL, 0, mask) == 1);
    CHECK(wubu_sampler_chain_stage_count(both) == 2);
    CHECK(wubu_sampler_chain_stage_count(NULL) == -1);

    CHECK(wubu_sampler_chain_create(0) == NULL);
    CHECK(wubu_sampler_chain_add(c, (wubu_sampler_stage_t)99, 1.0) == -1);
    CHECK(wubu_sampler_chain_add(c, WUBU_SAMPLE_TOP_K, 0.0) == -1);
    CHECK(wubu_sampler_chain_apply(c, NULL, 4, NULL, 0, mask) == -1);
    CHECK(wubu_sampler_chain_pick(c, logits, NULL, 4, 0) == -1);

    wubu_sampler_chain_free(c);
    wubu_sampler_chain_free(r);
    wubu_sampler_chain_free(t);
    wubu_sampler_chain_free(m);
    wubu_sampler_chain_free(p);
    wubu_sampler_chain_free(both);
    if (failures) {
        fprintf(stderr, "test_eg12_sampler: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg12_sampler: PASS\n");
    return 0;
}
