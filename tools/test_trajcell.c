/*
 * test_trajcell.c -- the TOOL-TRAJECTORY CELL gate (roadmap #2:
 * "closes the gap between the model talked and the colony learned
 * how to act").
 *
 * Asserts:
 *   1. trajectories are typed hive cells (goal, steps, outcome,
 *      cost, hash, provenance)
 *   2. the trajectory hash is stable + compact (dedup identity)
 *   3. "trajectories that solved similar goals" is queryable
 *   4. FAILED trajectories with high partial credit are mutation
 *      seeds (not garbage)
 */
#include <stdio.h>
#include <string.h>

#include "wubu_trajcell.h"
#include "wubu_hive.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_trajcell (tool trajectories join the hive) ===\n");

    wubu_hive_t tissue;
    if (wubu_hive_init(&tissue) != 0) FAIL("hive init");

    wubu_traj_tracker_t tt;
    if (wubu_traj_init(&tt, &tissue) != 0) FAIL("tracker init");

    /* 1. record trajectories: two successes near goal 42, one failure
     * with high partial credit, one far success */
    uint32_t stepsA[3] = { 11, 22, 33 };
    uint32_t stepsB[3] = { 11, 22, 34 };   /* near A */
    uint32_t stepsF[4] = { 5, 50, 60, 70 }; /* the failure */
    int64_t a = wubu_traj_record(&tt, 42, stepsA, 3, 1, 6.0f, 1.0f, 10.0f, 1);
    int64_t b = wubu_traj_record(&tt, 42, stepsB, 3, 1, 6.5f, 1.0f, 12.0f, 1);
    int64_t f = wubu_traj_record(&tt, 42, stepsF, 4, 0, 9.0f, 0.8f, 20.0f, 1);
    int64_t far = wubu_traj_record(&tt, 99, stepsF, 4, 1, 5.0f, 1.0f, 8.0f, 1);
    if (a <= 0 || b <= 0 || f <= 0 || far <= 0) FAIL("record failed");
    size_t live = wubu_hive_live(&tissue);
    if (live != 4) FAIL("trajectories not in the hive (%zu)", live);
    printf("  4 trajectory cells in the hive (2 success, 1 fail, 1 far)\n");

    /* 2. the hash is stable + compact */
    uint32_t h1 = wubu_traj_hash(stepsA, 3, 42);
    uint32_t h1b = wubu_traj_hash(stepsA, 3, 42);
    uint32_t h2 = wubu_traj_hash(stepsB, 3, 42);
    if (h1 != h1b) FAIL("hash not stable");
    if (h1 == h2) FAIL("different trajectories collided");
    printf("  hashes: A=%08x B=%08x (stable, distinct)\n", h1, h2);

    /* 3. similar-goal query: goal 42 successes = 2 */
    int64_t sim[8];
    int n_sim = wubu_traj_similar(&tt, 42, 0, sim, 8);
    printf("  similar to goal 42: %d successes\n", n_sim);
    if (n_sim != 2) FAIL("similar query wrong (%d)", n_sim);

    /* 4. mutation seeds: the failed trajectory with credit 0.8 */
    int64_t seeds[8];
    int n_seeds = wubu_traj_seeds(&tt, 0.7f, seeds, 8);
    printf("  mutation seeds (failed, credit>=0.7): %d\n", n_seeds);
    if (n_seeds != 1) FAIL("seed query wrong (%d)", n_seeds);
    if (seeds[0] != f) FAIL("the failure with partial credit was not the seed");

    char stats[256];
    wubu_traj_stats(&tt, stats, sizeof(stats));
    printf("  stats: %s\n", stats);

    printf("=== ALL TRAJCELL TESTS PASSED (the colony learns how to act) ===\n");
    return 0;
}
