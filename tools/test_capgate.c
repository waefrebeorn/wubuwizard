/*
 * test_capgate.c -- the CAPABILITY GATE gate (roadmap #4: "prevents
 * unbounded agent spawning and makes missing skills visible").
 *
 * Asserts:
 *   1. a well-equipped spawn passes
 *   2. low headroom -> decompose (split into cheaper sub-goals)
 *   3. missing tool / verifier -> fallback to a generalist + a gap
 *   4. a redundant similar lineage -> deny (no wasteful spawn)
 *   5. capability gaps are first-class hive cells (visible)
 */
#include <stdio.h>
#include <string.h>

#include "wubu_capgate.h"
#include "wubu_hive.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_capgate (capability-gated specialist spawning) ===\n");

    wubu_hive_t tissue;
    if (wubu_hive_init(&tissue) != 0) FAIL("hive init");

    wubu_capgate_t cg;
    if (wubu_capgate_init(&cg, &tissue, 0.3f, 0.15f, 50.0f) != 0) FAIL("gate init");

    /* 1. the well-equipped spawn passes */
    wubu_cap_input_t good = { 0.8f, 1, 1, -1.0f, 0.0f };
    wubu_cap_verdict_t v = wubu_capgate_check(&cg, 0, 42, &good);
    printf("  well-equipped: verdict %d (0=pass)\n", v);
    if (v != WUBU_CAP_PASS) FAIL("well-equipped spawn did not pass");

    /* 2. low headroom -> decompose */
    wubu_cap_input_t hot = { 0.1f, 1, 1, -1.0f, 0.0f };
    v = wubu_capgate_check(&cg, 0, 42, &hot);
    printf("  low headroom (0.1): verdict %d (1=decompose)\n", v);
    if (v != WUBU_CAP_DECOMPOSE) FAIL("low headroom did not decompose");

    /* 3. missing tool -> fallback + record the gap */
    wubu_cap_input_t notool = { 0.8f, 0, 1, -1.0f, 0.0f };
    v = wubu_capgate_check(&cg, 1, 42, &notool);
    printf("  missing tool: verdict %d (2=fallback)\n", v);
    if (v != WUBU_CAP_FALLBACK) FAIL("missing tool did not fallback");
    wubu_capgate_gap(&cg, 1, 42, 1, 1);   /* reason 1 = tool */

    /* missing verifier -> fallback + gap */
    wubu_cap_input_t nover = { 0.8f, 1, 0, -1.0f, 0.0f };
    v = wubu_capgate_check(&cg, 2, 42, &nover);
    if (v != WUBU_CAP_FALLBACK) FAIL("missing verifier did not fallback");
    wubu_capgate_gap(&cg, 2, 42, 2, 1);   /* reason 2 = verifier */

    /* 4. a redundant similar lineage -> deny */
    wubu_cap_input_t dup = { 0.8f, 1, 1, 0.9f, 5.0f };  /* similar fitness 0.9, 5 batches ago */
    v = wubu_capgate_check(&cg, 3, 42, &dup);
    printf("  redundant lineage: verdict %d (3=deny)\n", v);
    if (v != WUBU_CAP_DENY) FAIL("redundant similar lineage was not denied");

    /* 5. the gaps are first-class hive cells */
    size_t n = wubu_capgate_gap_count(&cg);
    printf("  capability gaps: %zu (hive live %zu)\n", n, wubu_hive_live(&tissue));
    if (n != 2) FAIL("gaps not recorded (%zu)", n);

    char stats[256];
    wubu_capgate_stats(&cg, stats, sizeof(stats));
    printf("  stats: %s\n", stats);

    printf("=== ALL CAPGATE TESTS PASSED (no unbounded spawning) ===\n");
    return 0;
}
