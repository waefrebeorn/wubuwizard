/*
 * test_lineage.c -- the LINEAGE + EXTINCTION gate (roadmap #1:
 * evolutionary pressure with memory, not just per-batch fitness).
 *
 * Asserts:
 *   1. a lineage records generations + best fitness (never falls)
 *   2. survival rises on improvement, decays on stagnation
 *   3. the extinction pass soft-extincts a stagnant lineage but keeps
 *      a surviving one alive
 *   4. near-miss cousins are queryable (the archive keeps them so
 *      diagnose avoids rediscovering dead ends)
 */
#include <stdio.h>
#include <string.h>

#include "wubu_lineage.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_lineage (evolutionary pressure with memory) ===\n");

    wubu_lineage_tracker_t lt;
    if (wubu_lineage_init(&lt, 16, 5, 0.2f) != 0) FAIL("init");

    /* 1. the winning lineage: improvements (lower loss) */
    uint64_t win = 0;
    for (int g = 0; g < 8; g++)
        win = wubu_lineage_record(&lt, win, 10.0f - (float)g, 10.0f - (float)g + 0.5f);
    if (win == 0) FAIL("no lineage id");
    if (lt.lines[0].generations != 8) FAIL("generations not counted");
    if (lt.lines[0].best_fitness != 3.0f) FAIL("best fitness wrong (%.2f)", lt.lines[0].best_fitness);
    if (lt.lines[0].stagnant_for != 0) FAIL("improving lineage flagged stagnant");
    printf("  winning lineage %llu: %llu gens, best %.2f, survival %.2f\n",
           (unsigned long long)win,
           (unsigned long long)lt.lines[0].generations,
           lt.lines[0].best_fitness, lt.lines[0].survival);
    if (lt.lines[0].survival <= 0.5f) FAIL("winner survival too low");

    /* 2. the stagnant lineage: no improvement for many cycles */
    uint64_t stag = wubu_lineage_record(&lt, 0, 9.0f, 9.0f);
    for (int g = 0; g < 7; g++)
        wubu_lineage_record(&lt, stag, 9.0f, 9.0f);   /* no improvement */
    if (lt.lines[1].stagnant_for != 7) FAIL("stagnation not counted");
    printf("  stagnant lineage: stagnant_for=%d survival %.2f\n",
           lt.lines[1].stagnant_for, lt.lines[1].survival);
    if (lt.lines[1].survival >= lt.lines[0].survival)
        FAIL("stagnant survival should be below the winner's");

    /* 3. the extinction pass: the stagnant lineage fades, the winner lives */
    int extinct = wubu_lineage_extinction_pass(&lt);
    printf("  extinction pass: %d soft-extinct (winner alive=%d)\n",
           extinct, lt.lines[0].extinct == 0);
    if (lt.lines[0].extinct) FAIL("the winning lineage was killed");
    if (!lt.lines[1].extinct) FAIL("the stagnant lineage survived (no pressure)");

    /* 4. near-miss cousins: a lineage close to the winner is kept */
    uint64_t cousin = wubu_lineage_record(&lt, 0, 3.5f, 3.0f);   /* close to 3.0 */
    uint64_t far = wubu_lineage_record(&lt, 0, 8.0f, 8.0f);      /* far */
    uint64_t near_ids[4];
    int n_near = wubu_lineage_nearmiss(&lt, 3.0f, 1.0f, near_ids, 4);
    printf("  near-miss cousins: %d (win=%.2f tol=1.0)\n", n_near, 3.0f);
    int found = 0;
    for (int i = 0; i < n_near; i++)
        if (near_ids[i] == cousin) found = 1;
    if (!found) FAIL("the near-miss cousin was not kept");

    char stats[256];
    wubu_lineage_stats(&lt, stats, sizeof(stats));
    printf("  stats: %s\n", stats);
    if (lt.n_extinctions == 0) FAIL("no extinctions recorded");

    wubu_lineage_free(&lt);
    printf("=== ALL LINEAGE TESTS PASSED (the colony has evolutionary memory) ===\n");
    return 0;
}
