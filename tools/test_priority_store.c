/*
 * test_priority_store.c -- the PRIORITY STORE gate (Phase 2
 * completion: "one checkpoint + one profile produces the right
 * precision morph, and diagnose refuses mutations that ignore
 * priority-store evidence").
 *
 * Asserts:
 *   1. cells register with BI importance + family
 *   2. the online Fisher/EWC update accumulates the gradient diagonal
 *   3. the DIAGNOSE GATE: a protected cell (high Fisher + recent
 *      rejection) is REFUSED; an unregistered/low-importance cell is
 *      allowed (the mutation evidence gates the morph)
 *   4. the sidecar saves + loads (the checkpoint's priority ledger)
 */
#include <stdio.h>
#include <string.h>

#include "wubu_priority_store.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_priority_store (the diagnose gate's evidence) ===\n");

    wubu_priority_store_t ps;
    if (wubu_prio_init(&ps, 0.5f, 0.3f) != 0) FAIL("init");

    /* 1. register cells: cell 0 (dense, important), cell 1 (gate_up) */
    wubu_prio_register(&ps, 0, 2, 0.9f, 1);   /* WUBU_FAM_DENSE */
    wubu_prio_register(&ps, 1, 0, 0.2f, 1);   /* WUBU_FAM_GATE_UP, low */
    if (ps.n != 2) FAIL("cells not registered");
    printf("  cells: [0]=dense imp 0.90, [1]=gate_up imp 0.20\n");

    /* 2. the Fisher update: cell 0 accumulates the grad diagonal
     * (the online EWC: the EMA over many batches) */
    for (int i = 0; i < 20; i++)
        wubu_prio_update_fisher(&ps, 0, 0.8f, 0.1f);
    printf("  cell 0 fisher after 20 updates: %.3f (>= 0.5 = the loss cares)\n",
           ps.cells[0].fisher);
    if (ps.cells[0].fisher < 0.5f) FAIL("the Fisher diagonal did not accumulate");

    /* 3. the diagnose gate:
     *   - cell 0 with a recent REJECTION -> protected -> REFUSED */
    wubu_prio_record_mutation(&ps, 0, 0);   /* rejected */
    int g = wubu_prio_gate(&ps, 0);
    printf("  gate cell 0 (high Fisher + rejected): %d (0 = refused)\n", g);
    if (g != 0) FAIL("the protected cell was not refused");
    /*   - cell 1 (low importance, no rejects) -> allowed */
    g = wubu_prio_gate(&ps, 1);
    printf("  gate cell 1 (low importance): %d (1 = allowed)\n", g);
    if (g != 1) FAIL("the low-importance cell was refused");
    /*   - an unregistered cell -> allowed (no evidence yet) */
    g = wubu_prio_gate(&ps, 7);
    printf("  gate unregistered cell 7: %d (1 = allowed)\n", g);
    if (g != 1) FAIL("the unregistered cell was refused");

    /* 4. the sidecar: save + load (the checkpoint's priority ledger) */
    wubu_prio_set_delta(&ps, 0, -1.0f);   /* the ladder moved cell 0 */
    static char ck[4096];
    long n = wubu_prio_save(&ps, ck, (long)sizeof(ck));
    if (n <= 0) FAIL("sidecar save failed");
    wubu_priority_store_t ps2;
    wubu_prio_init(&ps2, 0.5f, 0.3f);
    int restored = wubu_prio_load(&ps2, ck, n);
    if (restored != ps.n) FAIL("sidecar restore count mismatch (%d vs %d)", restored, ps.n);
    if (ps2.cells[0].fisher != ps.cells[0].fisher) FAIL("fisher not restored");
    printf("  sidecar: %d cells saved/restored (%ld bytes)\n", restored, n);

    char stats[256];
    wubu_prio_stats(&ps, stats, sizeof(stats));
    printf("  stats: %s\n", stats);
    if (ps.n_refused == 0) FAIL("no refusals recorded (the gate does not enforce)");

    printf("=== ALL PRIORITY STORE TESTS PASSED (the evidence gates the morph) ===\n");
    return 0;
}
