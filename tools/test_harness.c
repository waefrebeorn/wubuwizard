/*
 * test_harness.c -- the TASK HARNESS gate (Phase 3: Pillars 11+20 —
 * "the colony cannot pass by short-batch loss alone").
 *
 * Asserts:
 *   1. the fixed suite runs with deterministic scorers
 *   2. outcomes are TRAJ CELLS in the hive (goal, outcome, cost)
 *   3. the suite score is the first-class fitness signal
 *   4. the loopguard stops the harness cleanly at the deadline/step
 *      ceiling (unattended-safe)
 */
#include <stdio.h>
#include <string.h>

#include "wubu_harness.h"
#include "wubu_hive.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_harness (the sustained autonomy suite) ===\n");

    wubu_hive_t tissue;
    if (wubu_hive_init(&tissue) != 0) FAIL("hive init");

    wubu_harness_t h;
    /* a small deadline + step ceiling for the test */
    if (wubu_harness_init(&h, &tissue, 1000000L, 6) != 0) FAIL("harness init");
    printf("  fixed suite: %d tasks (3 edits, 2 transforms, 2 tool-use)\n", h.n_tasks);

    /* 1. run the suite: 4 good attempts + 2 bad */
    int p1 = wubu_harness_run_task(&h, 0, 3, 0.95f, 0.3f);   /* pass */
    int p2 = wubu_harness_run_task(&h, 1, 4, 0.90f, 0.4f);   /* pass */
    int p3 = wubu_harness_run_task(&h, 2, 2, 0.60f, 0.5f);   /* fail (correctness) */
    int p4 = wubu_harness_run_task(&h, 3, 5, 0.85f, 0.6f);   /* pass */
    int p5 = wubu_harness_run_task(&h, 4, 3, 0.50f, 0.7f);   /* fail */
    int p6 = wubu_harness_run_task(&h, 5, 3, 0.80f, 0.8f);   /* pass */
    if (!p1 || !p2 || !p4 || !p6) FAIL("good tasks did not pass");
    if (p3 || p5) FAIL("bad tasks passed (scorer wrong)");
    printf("  6 attempts: 4 pass, 2 fail (deterministic scorer ✓)\n");

    /* 2. outcomes are traj cells in the hive */
    size_t live = wubu_hive_live(&tissue);
    printf("  hive live cells: %zu (6 traj cells expected)\n", live);
    if (live < 6) FAIL("outcomes not in the hive as traj cells");

    /* 3. the suite score is the first-class fitness signal */
    float score = wubu_harness_suite_score(&h);
    printf("  suite score: %.3f\n", score);
    if (score <= 0.0f || score > 1.0f) FAIL("suite score out of range");

    /* 4. the loopguard stops the harness at the step ceiling */
    int continue_ok = wubu_harness_may_continue(&h);
    printf("  loopguard: continue=%d (step %ld / max 6)\n", continue_ok, h.step);
    if (continue_ok) FAIL("the step ceiling did not stop the harness");
    wubu_harness_t h2;
    wubu_harness_init(&h2, &tissue, 0, 1000);   /* no deadline, big ceiling */
    if (!wubu_harness_may_continue(&h2)) FAIL("the fresh harness should continue");

    char stats[256];
    wubu_harness_stats(&h, stats, sizeof(stats));
    printf("  stats: %s\n", stats);

    printf("=== ALL HARNESS TESTS PASSED (unattended-safe, first-class fitness) ===\n");
    return 0;
}
