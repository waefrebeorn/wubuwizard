/* test_eg89_keyed_sample.c -- EG-89 gate: position-keyed exponential race. */
#include <stdio.h>

#include "wubu_keyed_sample.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define TRIALS 20000

int main(void) {
    wubu_keyed_sample_t *p = wubu_keyed_sample_create(0x12345678ULL);
    float probs[3] = {0.6f, 0.3f, 0.1f};
    int id, counts[3] = {0, 0, 0};
    CHECK(p != NULL);
    id = wubu_keyed_sample(p, 5, probs, 3);
    CHECK(id >= 0 && id < 3);
    CHECK(id == wubu_keyed_sample(p, 5, probs, 3)); /* deterministic key */
    printf("EG-89 seed=0x12345678 pos=5 id=%d (repeatable)\n", id);
    wubu_keyed_sample_free(p);

    /* The noise field is keyed by seed, so a fresh seed is a fresh draw. */
    for (unsigned long long s = 0; s < TRIALS; ++s) {
        wubu_keyed_sample_t *q = wubu_keyed_sample_create(s);
        int k = wubu_keyed_sample(q, 7, probs, 3);
        if (k >= 0 && k < 3) ++counts[k];
        wubu_keyed_sample_free(q);
    }
    printf("EG-89 trials=%d counts=[%d %d %d] expected=[12000 6000 2000]\n",
           TRIALS, counts[0], counts[1], counts[2]);
    CHECK(counts[0] > 11640 && counts[0] < 12360);
    CHECK(counts[1] > 5820 && counts[1] < 6180);
    CHECK(counts[2] > 1940 && counts[2] < 2060);

    /* Two tree shapes placing nodes at the same absolute positions must agree. */
    {
        wubu_keyed_sample_t *t = wubu_keyed_sample_create(0xabcdef01ULL);
        float tp[3] = {0.5f, 0.3f, 0.2f};
        CHECK(t != NULL);
        for (int depth = 0; depth < 6; ++depth) {
            int a = wubu_keyed_sample_at(t, 10, depth, tp, 3);
            int b = wubu_keyed_sample(t, 10 + depth, tp, 3);
            int c = wubu_keyed_sample_at(t, 10 + depth, 0, tp, 3);
            CHECK(a == b && b == c);
        }
        printf("EG-89 tree-shape invariance: 6 absolute positions, 0 mismatches\n");
        /* Bad input is rejected by status. */
        CHECK(wubu_keyed_sample(NULL, 1, tp, 3) == -1);
        CHECK(wubu_keyed_sample(t, -1, tp, 3) == -1);
        CHECK(wubu_keyed_sample(t, 1, tp, 0) == -1);
        CHECK(wubu_keyed_sample(t, 1, NULL, 3) == -1);
        {
            float zero[3] = {0.0f, 0.0f, 0.0f};
            CHECK(wubu_keyed_sample(t, 1, zero, 3) == -1);
        }
        CHECK(wubu_keyed_sample_at(t, -1, 2, tp, 3) == -1);
        wubu_keyed_sample_free(t);
    }

    if (failures) { fprintf(stderr, "test_eg89_keyed_sample: FAIL (%d)\n", failures); return 1; }
    printf("test_eg89_keyed_sample: PASS\n");
    return 0;
}
