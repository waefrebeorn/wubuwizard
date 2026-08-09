/*
 * test_pref.c -- the RLHF ORACLE gate (the user directive #4: "give
 * the oracle teeth").
 *
 * Asserts:
 *   1. preference pairs from the mutation outcomes (accepted vs
 *      rejected) build correctly
 *   2. the Bradley-Terry update moves the survival scores: the
 *      winning lineage's survival RISES, the loser's FALLS
 *   3. credit assignment: repeated wins push a cell's survival up,
 *      repeated losses push it down (the survival probability
 *      becomes the mutation gate's prior)
 */
#include <stdio.h>
#include <string.h>

#include "wubu_pref.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_pref (the RLHF oracle) ===\n");

    wubu_pref_t pf;
    if (wubu_pref_init(&pf, 4, 32, 0.1f) != 0) FAIL("pref init");
    printf("  initial survivals: %.2f %.2f %.2f %.2f\n",
           wubu_pref_survival(&pf, 0), wubu_pref_survival(&pf, 1),
           wubu_pref_survival(&pf, 2), wubu_pref_survival(&pf, 3));
    if (wubu_pref_survival(&pf, 0) != 0.5f) FAIL("survival should start at 0.5");

    /* 1. pairs from outcomes: cell 0's lineage won (its loss improved),
     * cell 1's lineage lost (another outcome was better). The pair is
     * (better_fitness, worse_fitness, cell_idx) — cell_idx is the cell
     * whose survival the pair updates. */
    wubu_pref_pair_from_outcome(&pf, 1, 5.0f, 8.0f, 0, 0xFF, 1);  /* cell 0 won */
    wubu_pref_pair_from_outcome(&pf, 2, 6.0f, 7.5f, 0, 0xFF, 1);  /* cell 0 won again */
    wubu_pref_pair_from_outcome(&pf, 3, 4.0f, 9.0f, 1, 0xFF, 0);  /* cell 1 lost */
    if (pf.pairs_n != 3) FAIL("pairs not recorded (%d)", pf.pairs_n);

    /* 2. the update: cell 0's survival rises (it won twice) */
    float s0_before = wubu_pref_survival(&pf, 0);
    wubu_pref_update(&pf);
    float s0_after = wubu_pref_survival(&pf, 0);
    float s1_after = wubu_pref_survival(&pf, 1);
    printf("  after 3 pairs: cell0 %.3f -> %.3f, cell1 -> %.3f\n",
           s0_before, s0_after, s1_after);
    if (s0_after <= s0_before) FAIL("cell 0's survival did not rise (it won)");
    if (s1_after >= 0.5f) FAIL("cell 1's survival did not fall (it lost)");

    /* 3. credit assignment: more wins push higher */
    for (int i = 0; i < 10; i++)
        wubu_pref_pair_from_outcome(&pf, (uint64_t)(10 + i), 5.0f, 8.0f, 0, 0xFF, 1);
    wubu_pref_update(&pf);
    float s0_final = wubu_pref_survival(&pf, 0);
    printf("  after 13 wins: cell0 survival %.3f\n", s0_final);
    if (s0_final <= s0_after) FAIL("credit assignment did not push survival up");
    if (s0_final > 1.0f) FAIL("survival overflowed");

    char stats[256];
    wubu_pref_stats(&pf, stats, sizeof(stats));
    printf("  stats: %s\n", stats);
    if (pf.updates == 0) FAIL("no updates applied");

    wubu_pref_free(&pf);
    printf("=== ALL PREF TESTS PASSED (the oracle has teeth) ===\n");
    return 0;
}
