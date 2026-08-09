/*
 * test_resources.c — the RESOURCE LEDGER gate (post-AN54 A6).
 *
 * Asserts:
 *   1. the snapshot reads real RSS + CPU (nonzero on the host)
 *   2. the windowed update computes a throughput + soft fitness
 *   3. the budget blows -> the soft fitness drops (the metadiag can
 *      penalize a mutation that blows memory)
 */
#include <stdio.h>
#include <string.h>

#include "wubu_resources.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_resources (the resource ledger) ===\n");

    /* 1. the snapshot reads real values */
    wubu_res_t r;
    wubu_res_snapshot(&r);
    printf("  snapshot: rss=%llu KB cpu=%.2fs\n",
           (unsigned long long)r.rss_kb, r.cpu_sec);
    if (r.rss_kb == 0) FAIL("RSS is 0 (procfs read failed)");

    /* 2. the windowed update: throughput + soft fitness */
    wubu_res_update(&r, 100);   /* 100 events since the last call */
    printf("  window: cpu_delta=%.3fs wall=%.3fs throughput=%.1f ev/s "
           "soft=%.2f\n", r.cpu_delta, r.wall_delta, r.throughput, r.soft_fitness);
    if (r.soft_fitness < 0.0f || r.soft_fitness > 1.0f)
        FAIL("the soft fitness is out of range");

    /* 3. the budget: RSS over the budget -> the soft fitness drops */
    wubu_res_set_budget(&r, 1);   /* a 1KB budget (RSS is way over) */
    wubu_res_update(&r, 10);
    printf("  blown budget: soft=%.2f (1KB budget vs %llu KB RSS)\n",
           r.soft_fitness, (unsigned long long)r.rss_kb);
    if (r.soft_fitness > 0.5f) FAIL("the blown budget did not drop the fitness");

    printf("=== ALL RESOURCES TESTS PASSED (the metadiag gets soft fitness) ===\n");
    return 0;
}
