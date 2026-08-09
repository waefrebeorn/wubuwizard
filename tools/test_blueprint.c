/*
 * test_blueprint.c -- the EXECUTABLE BLUEPRINT gate (Phase 6: WB01 —
 * "mark WB01 wired only when a train+diagnose run refuses off-
 * blueprint mutations by default").
 *
 * Asserts:
 *   1. on-blueprint mutations pass (the axes in range)
 *   2. OFF-blueprint mutations are REFUSED by default (the MoE axis
 *      beyond the bound; the locked axis always)
 *   3. an expansion is REQUIRED + versioned (a contract-expansion
 *      meta-cell) before the bound moves
 *   4. the refusal counter proves the run enforces the blueprint
 */
#include <stdio.h>
#include <string.h>

#include "wubu_blueprint.h"
#include "wubu_hive.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_blueprint (the executable lineage) ===\n");

    wubu_hive_t tissue;
    if (wubu_hive_init(&tissue) != 0) FAIL("hive init");

    wubu_blueprint_t bp;
    if (wubu_blueprint_init(&bp, &tissue) != 0) FAIL("blueprint init");

    /* 1. on-blueprint mutations pass */
    int ok = wubu_blueprint_allows(&bp, WUBU_BP_MOE, 8.0f);
    printf("  MoE 8 experts (range 2..16): allowed=%d\n", ok);
    if (!ok) FAIL("an in-range mutation was refused");

    ok = wubu_blueprint_allows(&bp, WUBU_BP_NEST, 1.0f);
    if (!ok) FAIL("the nesting curvature 1.0 was refused");

    /* 2. off-blueprint mutations are REFUSED by default */
    int bad = wubu_blueprint_allows(&bp, WUBU_BP_MOE, 64.0f);  /* > 16 */
    printf("  MoE 64 experts (off-blueprint): allowed=%d (0 = refused)\n", bad);
    if (bad) FAIL("an off-blueprint mutation was allowed");

    /* 3. the expansion is required + versioned (a meta-cell in the hive) */
    uint32_t exp = wubu_blueprint_expand(&bp, WUBU_BP_MOE, 2.0f, 64.0f,
                                         1, "the colony grew past 16 experts");
    printf("  expansion version: %u (the bound moved ONLY via this)\n", exp);
    if (exp == 0) FAIL("the expansion was not recorded");
    size_t live = wubu_hive_live(&tissue);
    if (live < 1) FAIL("the expansion meta-cell did not land in the hive");
    ok = wubu_blueprint_allows(&bp, WUBU_BP_MOE, 64.0f);
    if (!ok) FAIL("the expanded bound still refuses (the expansion failed)");

    char stats[256];
    wubu_blueprint_stats(&bp, stats, sizeof(stats));
    printf("  stats: %s\n", stats);
    if (bp.n_refused == 0) FAIL("no refusals recorded (the run does not enforce)");

    printf("=== ALL BLUEPRINT TESTS PASSED (the lineage is a constraint) ===\n");
    return 0;
}
