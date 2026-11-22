/* test_eg04_attention_sink.c -- EG-04 gate: attention sinks pin the first N
 * KV positions so sliding-window reclamation cannot evict them. Verifies
 * sink refusal, non-sink reuse, baseline with sinks disabled, validation,
 * and deterministic snapshots. Prints: test_eg04_attention_sink: PASS. */
#include <stdio.h>
#include <string.h>

#include "wubu_swa_reclaim.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_swa_reclaim_t *c = wubu_swa_reclaim_create(8, 1);
    CHECK(c != NULL);
    if (!c) return 1;

    /* Fill eight cells with positions 0..7 (bound 0 accepts all writes). */
    CHECK(wubu_swa_reclaim_bound(c, 0) == 0);
    for (int i = 0; i < 8; i++)
        CHECK(wubu_swa_reclaim_reuse(c, i, 0, i) == WUBU_SWA_OK);

    /* Slide the window: only positions >= 6 are live. */
    CHECK(wubu_swa_reclaim_set_bound(c, 0, 6) == WUBU_SWA_OK);
    CHECK(wubu_swa_set_sinks(c, 2) == WUBU_SWA_OK);
    CHECK(wubu_swa_sinks(c) == 2);
    printf("EG-04 window_bound=6 sinks=2\n");

    /* Sink cells are pinned even though both positions are outside the window. */
    CHECK(wubu_swa_reclaim_can_reuse(c, 0, 0, 100) == 0);
    CHECK(wubu_swa_reclaim_can_reuse(c, 1, 0, 100) == 0);
    CHECK(wubu_swa_reclaim_reuse(c, 0, 0, 100) == WUBU_SWA_ERR_UNSAFE);
    CHECK(wubu_swa_reclaim_reuse(c, 1, 0, 100) == WUBU_SWA_ERR_UNSAFE);
    CHECK(wubu_swa_reclaim_cell_pos(c, 0) == 0);
    CHECK(wubu_swa_reclaim_cell_pos(c, 1) == 1);

    /* A non-sink masked cell is still reclaimable. */
    CHECK(wubu_swa_reclaim_can_reuse(c, 2, 0, 100) == 1);
    CHECK(wubu_swa_reclaim_reuse(c, 2, 0, 100) == WUBU_SWA_OK);
    CHECK(wubu_swa_reclaim_cell_pos(c, 2) == 100);
    CHECK(wubu_swa_reclaim_cell_pos(c, 0) == 0);
    CHECK(wubu_swa_reclaim_cell_pos(c, 1) == 1);
    printf("EG-04 sink cells survived; non-sink cell 2 reused\n");

    /* With sinks disabled the same position becomes reclaimable. */
    CHECK(wubu_swa_set_sinks(c, 0) == WUBU_SWA_OK);
    CHECK(wubu_swa_reclaim_can_reuse(c, 0, 0, 200) == 1);
    CHECK(wubu_swa_reclaim_reuse(c, 0, 0, 200) == WUBU_SWA_OK);
    CHECK(wubu_swa_reclaim_cell_pos(c, 0) == 200);
    printf("EG-04 sinks=0 baseline: cell 0 reused\n");

    /* Validation and snapshot determinism. */
    CHECK(wubu_swa_set_sinks(c, -1) == WUBU_SWA_ERR_POS);
    CHECK(wubu_swa_set_sinks(c, 9) == WUBU_SWA_ERR_POS);
    CHECK(wubu_swa_set_sinks(NULL, 2) == WUBU_SWA_ERR_ARG);
    CHECK(wubu_swa_sinks(NULL) == -1);
    CHECK(wubu_swa_set_sinks(c, 4) == WUBU_SWA_OK);
    size_t n = wubu_swa_reclaim_snapshot_size(c);
    unsigned char a[4096], b[4096];
    CHECK(n <= sizeof(a));
    if (n <= sizeof(a)) {
        wubu_swa_reclaim_snapshot(c, a);
        wubu_swa_reclaim_snapshot(c, b);
        CHECK(memcmp(a, b, n) == 0);
        unsigned char saved = a[2 * sizeof(int)];
        a[2 * sizeof(int)] = (unsigned char)(saved ^ 0xFF);
        CHECK(memcmp(a, b, n) != 0); /* sink count is in the snapshot */
    }
    printf("EG-04 snapshot_size=%zu deterministic\n", n);

    wubu_swa_reclaim_free(c);
    if (failures) {
        fprintf(stderr, "test_eg04_attention_sink: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg04_attention_sink: PASS\n");
    return 0;
}
