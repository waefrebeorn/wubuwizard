/* test_eg76_index_reuse.c -- EG-76 gate: cross-layer DSA index reuse. */
#include <stdio.h>

#include "wubu_index_reuse.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_index_reuse_t *p = wubu_index_reuse_create(4, 3, 8, 2);
    double scores[6] = {0.1, 0.9, 0.4, 0.8, 0.2, 0.7};
    int idx[3];
    CHECK(p != NULL);

    /* Selector layer computes top-3 and publishes one index-key KV set. */
    CHECK(wubu_index_reuse_compute(p, scores, 6, idx) == 3);
    CHECK(idx[0] == 1 && idx[1] == 3 && idx[2] == 5);
    CHECK(wubu_index_reuse_publish(p, 0, idx, 3) == 48); /* 3*8*2 bytes */
    CHECK(wubu_index_reuse_bytes_allocated(p) == 48);
    printf("EG-76 top3=[%d %d %d] published_bytes=%zu (one set for 4 layers)\n",
           idx[0], idx[1], idx[2], wubu_index_reuse_bytes_allocated(p));

    /* Matching layers reuse the set; a divergence must recompute. */
    {
        int same[3] = {1, 3, 5}, diff[3] = {0, 2, 4};
        CHECK(wubu_index_reuse_reuse(p, 1, same, 3) == 0);
        CHECK(wubu_index_reuse_reuse(p, 2, same, 3) == 0);
        CHECK(wubu_index_reuse_reuse(p, 3, diff, 3) == -1);
        CHECK(wubu_index_reuse_bytes_saved(p) == 96);
        CHECK(wubu_index_reuse_reuse(p, 0, same, 3) == -1);
        printf("EG-76 reused=2 saved_bytes=%zu divergent_layer_recomputed\n",
               wubu_index_reuse_bytes_saved(p));
        CHECK(wubu_index_reuse_create(0, 3, 8, 2) == NULL);
        CHECK(wubu_index_reuse_create(4, 0, 8, 2) == NULL);
        CHECK(wubu_index_reuse_compute(NULL, scores, 6, idx) == -1);
        CHECK(wubu_index_reuse_publish(p, 9, same, 3) == 0);
        CHECK(wubu_index_reuse_publish(p, 1, same, 2) == 0);
        CHECK(wubu_index_reuse_reuse(NULL, 1, same, 3) == -1);
    }

    wubu_index_reuse_free(p);
    if (failures) { fprintf(stderr, "test_eg76_index_reuse: FAIL (%d)\n", failures); return 1; }
    printf("test_eg76_index_reuse: PASS\n");
    return 0;
}
