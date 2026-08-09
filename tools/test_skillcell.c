/*
 * test_skillcell.c -- the SKILL CURRICULUM gate (Phase 4: Pillar 17 —
 * "the colony can add a skill from experience, reuse it on a similar
 * goal, and shrink it if it stops helping").
 *
 * Asserts:
 *   1. successful traj patterns become draft skill cells in the hive
 *   2. accepted drafts become versioned skills (the gate passed)
 *   3. the orchestrator's match finds the best skill BEFORE spawning
 *      a new specialist
 *   4. unused + low-fitness skills get pruned (extinction)
 */
#include <stdio.h>
#include <string.h>

#include "wubu_skillcell.h"
#include "wubu_hive.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_skillcell (the skill curriculum) ===\n");

    wubu_hive_t tissue;
    if (wubu_hive_init(&tissue) != 0) FAIL("hive init");

    wubu_skill_tracker_t sk;
    if (wubu_skill_init(&sk, &tissue) != 0) FAIL("skill init");

    /* 1. successful traj patterns -> drafts */
    int64_t d1 = wubu_skill_propose(&sk, 42, 0, 0xABCD, 0.9f, 1);  /* code, goal 42 */
    int64_t d2 = wubu_skill_propose(&sk, 42, 1, 0x1234, 0.8f, 1);  /* math, goal 42 */
    int64_t d3 = wubu_skill_propose(&sk, 99, 0, 0xDEAD, 0.6f, 1);  /* code, goal 99 */
    if (d1 <= 0 || d2 <= 0 || d3 <= 0) FAIL("draft proposal failed");
    size_t live = wubu_hive_live(&tissue);
    if (live != 3) FAIL("drafts not in the hive (%zu)", live);
    printf("  3 drafts in the hive (code/math goals)\n");

    /* 2. accept: the drafts pass the gate -> versioned skills */
    uint32_t v1 = wubu_skill_accept(&sk, d1);
    uint32_t v2 = wubu_skill_accept(&sk, d2);
    printf("  accepted versions: %u, %u\n", v1, v2);
    if (v1 == 0 || v2 == 0) FAIL("acceptance failed");
    if (v1 >= v2) FAIL("versions did not grow");

    /* 3. the match: a code specialist for goal 42 finds d1 (fitness 0.9) */
    int64_t m = wubu_skill_match(&sk, 42, 0, 2);
    printf("  match for (goal 42, code): %lld (expect the 0.9-fitness draft)\n",
           (long long)m);
    if (m != d1) FAIL("the match did not find the best skill");
    /* a goal with no match -> -1 (the orchestrator must spawn new) */
    int64_t none = wubu_skill_match(&sk, 200, 0, 2);
    if (none != -1) FAIL("no-match should return -1");

    /* 4. record uses + prune: d3 (unused, low fitness) gets pruned */
    wubu_skill_accept(&sk, d3);          /* d3 becomes a real skill */
    wubu_skill_use(&sk, d1);
    wubu_skill_use(&sk, d2);
    int pruned = wubu_skill_prune(&sk, 1, 0.8f);
    printf("  pruned: %d (unused + low-fitness skills)\n", pruned);
    if (pruned < 1) FAIL("the unused low-fitness skill survived");

    char stats[256];
    wubu_skill_stats(&sk, stats, sizeof(stats));
    printf("  stats: %s\n", stats);
    if (sk.n_accepted != 3) FAIL("acceptance count wrong");

    printf("=== ALL SKILLCELL TESTS PASSED (the colony learns skills) ===\n");
    return 0;
}
