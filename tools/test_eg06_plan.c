/* test_eg06_plan.c -- EG-06 gate: placement solver + zigzag block schedule.
 * Compares the greedy placement against an exhaustive optimum for 6 layers,
 * checks capacity, and verifies the forward/reverse schedule.
 * Prints: test_eg06_plan: PASS. */
#include <stdio.h>
#include <string.h>

#include "wubu_offload_plan.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define GB (1024ULL * 1024ULL * 1024ULL)

int main(void) {
    wubu_offload_hw_t hw;
    hw.bandwidth_bps[WUBU_OFFLOAD_GPU] = 400ULL * 1024 * 1024 * 1024ULL;
    hw.bandwidth_bps[WUBU_OFFLOAD_RAM] = 50ULL * 1024 * 1024 * 1024ULL;
    hw.bandwidth_bps[WUBU_OFFLOAD_SSD] = 2ULL * 1024 * 1024 * 1024ULL;
    hw.capacity[WUBU_OFFLOAD_GPU] = 4 * GB;
    hw.capacity[WUBU_OFFLOAD_RAM] = 16 * GB;
    hw.capacity[WUBU_OFFLOAD_SSD] = 100 * GB;

    enum { N = 6 };
    wubu_offload_layer_t layers[N] = {
        { 1 * GB, 100 }, { 3 * GB / 2, 90 }, { GB / 2, 80 },
        { 2 * GB, 70 },  { 1 * GB, 60 },      { 1 * GB, 50 }
    };
    int assign[N];
    CHECK(wubu_offload_solve(&hw, layers, N, assign) == 0);
    printf("EG-06 assignment:");
    for (int i = 0; i < N; i++) printf(" %d", assign[i]);
    printf("\n");

    /* Capacity is never exceeded. */
    uint64_t used[WUBU_OFFLOAD_TIERS] = {0, 0, 0};
    for (int i = 0; i < N; i++) {
        CHECK(assign[i] >= 0 && assign[i] < WUBU_OFFLOAD_TIERS);
        used[assign[i]] += layers[i].bytes;
        CHECK(used[assign[i]] <= hw.capacity[assign[i]]);
    }

    double greedy = wubu_offload_cost(&hw, layers, N, assign);
    printf("EG-06 greedy cost=%.6fs\n", greedy);
    CHECK(greedy > 0.0);

    /* Exhaustive optimum over 3^6 assignments; keep the best feasible. */
    double best = 1e300;
    int choice[N];
    for (int code = 0; code < 729; code++) {
        int v = code, ok = 1;
        uint64_t cap[WUBU_OFFLOAD_TIERS] = {0, 0, 0};
        for (int i = 0; i < N; i++) {
            choice[i] = v % 3;
            v /= 3;
            cap[choice[i]] += layers[i].bytes;
            if (cap[choice[i]] > hw.capacity[choice[i]]) { ok = 0; break; }
        }
        if (!ok) continue;
        double c = wubu_offload_cost(&hw, layers, N, choice);
        if (c < best) best = c;
    }
    printf("EG-06 exhaustive optimum=%.6fs ratio=%.3f\n", best, greedy / best);
    CHECK(best < 1e299);
    CHECK(greedy <= 2.0 * best);
    CHECK(greedy <= best * 1.000001 + 1e-12); /* this instance is optimal */

    /* All-SSD baseline for context. */
    int all_ssd[N];
    for (int i = 0; i < N; i++) all_ssd[i] = WUBU_OFFLOAD_SSD;
    double baseline = wubu_offload_cost(&hw, layers, N, all_ssd);
    printf("EG-06 all-SSD baseline=%.6fs\n", baseline);
    CHECK(greedy < baseline);

    /* Zigzag schedule: forward 0..N-1 then reverse N-1..0, twice. */
    int sched[4 * N];
    CHECK(wubu_offload_zigzag(N, 2, sched, 4 * N) == 4 * N);
    for (int pass = 0; pass < 4; pass++) {
        int seen[N];
        memset(seen, 0, sizeof(seen));
        for (int i = 0; i < N; i++) {
            int expect = (pass % 2 == 0) ? i : (N - 1 - i);
            CHECK(sched[pass * N + i] == expect);
            CHECK(seen[expect] == 0);
            seen[expect] = 1;
        }
    }
    printf("EG-06 zigzag: %d entries, 4 alternating passes\n", 4 * N);

    /* Fail-closed validation and infeasible capacity. */
    CHECK(wubu_offload_solve(NULL, layers, N, assign) == -1);
    CHECK(wubu_offload_cost(&hw, layers, N, NULL) == -1.0);
    CHECK(wubu_offload_zigzag(N, 2, sched, 4 * N - 1) == -1);
    wubu_offload_hw_t tiny = hw;
    tiny.capacity[WUBU_OFFLOAD_GPU] = 0;
    tiny.capacity[WUBU_OFFLOAD_RAM] = 0;
    tiny.capacity[WUBU_OFFLOAD_SSD] = 0;
    CHECK(wubu_offload_solve(&tiny, layers, N, assign) == -1);

    if (failures) {
        fprintf(stderr, "test_eg06_plan: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg06_plan: PASS\n");
    return 0;
}
