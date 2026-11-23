/* test_eg20_expert.c -- EG-20 gate: MoE expert trace cache + layer decay. */
#include <stdio.h>

#include "wubu_expert_trace.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_expert_trace_t *t = wubu_expert_trace_create(2, 4, 2, 2, 4096);
    wubu_expert_trace_stats_t st;

    CHECK(t != NULL);
    if (!t) return 1;

    for (int i = 0; i < 3; i++) CHECK(wubu_expert_trace_observe(t, 0, 2) == 0);
    CHECK(wubu_expert_trace_observe(t, 0, 3) == 0);
    CHECK(wubu_expert_trace_predict(t, 0) == 2);

    CHECK(wubu_expert_trace_prefetch(t, 0, 2) == 0); /* cold load */
    CHECK(wubu_expert_trace_prefetch(t, 0, 2) == 1); /* cache hit */
    CHECK(wubu_expert_trace_prefetch(t, 0, 1) == 0);
    CHECK(wubu_expert_trace_prefetch(t, 0, 3) == 0); /* evicts lowest-score */

    CHECK(wubu_expert_trace_tick(t) == 0);
    CHECK(wubu_expert_trace_tick(t) == 1); /* one decay step */
    for (int i = 0; i < 4; i++) CHECK(wubu_expert_trace_observe(t, 0, 1) == 0);
    CHECK(wubu_expert_trace_predict(t, 0) == 1); /* routing drift followed */

    wubu_expert_trace_stats(t, &st);
    printf("EG-20 hits=%d misses=%d loads=%d evictions=%d decay=%d bytes=%llu\n",
           st.hits, st.misses, st.loads, st.evictions, st.decay_steps,
           (unsigned long long)st.bytes_moved);
    CHECK(st.hits == 1 && st.misses == 3);
    CHECK(st.loads == 3 && st.evictions == 1);
    CHECK(st.decay_steps == 1 && st.bytes_moved == 3u * 4096u);

    CHECK(wubu_expert_trace_observe(t, 2, 0) == -1);
    CHECK(wubu_expert_trace_observe(t, 0, 9) == -1);
    CHECK(wubu_expert_trace_prefetch(t, -1, 0) == -1);
    CHECK(wubu_expert_trace_predict(t, 99) == -1);
    CHECK(wubu_expert_trace_create(0, 4, 2, 2, 4096) == NULL);

    wubu_expert_trace_free(t);
    if (failures) {
        fprintf(stderr, "test_eg20_expert: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg20_expert: PASS\n");
    return 0;
}
