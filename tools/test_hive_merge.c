/*
 * test_hive_merge.c — the MULTI-CHECKPOINT LINEAGE MERGE gate (AN47
 * #6: "the colony can federate without forking fitness history").
 *
 * Asserts:
 *   1. the union: a cell only in one archive is kept
 *   2. the conflict rule: the better-fitness cell wins
 *   3. the PRIORITY STORE overrides: a protected cell wins even with
 *      worse fitness (the Fisher evidence beats the loss gate)
 *   4. the graveyards union (negative examples never dropped)
 */
#include <stdio.h>
#include <string.h>

#include "wubu_hive_merge.h"
#include "wubu_diagnosis.h"
#include "wubu_priority_store.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

static void add_cell(wubu_diag_loop_t *l, uint64_t batch, uint8_t cell,
                     float fitness, int verdict)
{
    wubu_fitness_cell_t c;
    memset(&c, 0, sizeof(c));
    c.batch = batch; c.cell_idx = cell; c.fitness = fitness;
    c.verdict = (wubu_diag_verdict_t)verdict;
    l->ledger[l->ledger_n++] = c;
}

int main(void)
{
    printf("=== test_hive_merge (federate without forking) ===\n");

    wubu_diag_loop_t L, R, OUT;
    memset(&L, 0, sizeof(L)); memset(&R, 0, sizeof(R));
    memset(&OUT, 0, sizeof(OUT));
    L.ledger_cap = R.ledger_cap = OUT.ledger_cap = 16;
    L.grave_cap = R.grave_cap = OUT.grave_cap = 16;
    L.ledger = (wubu_fitness_cell_t *)calloc(16, sizeof(wubu_fitness_cell_t));
    R.ledger = (wubu_fitness_cell_t *)calloc(16, sizeof(wubu_fitness_cell_t));
    OUT.ledger = (wubu_fitness_cell_t *)calloc(16, sizeof(wubu_fitness_cell_t));
    L.graveyard = (wubu_fitness_cell_t *)calloc(16, sizeof(wubu_fitness_cell_t));
    R.graveyard = (wubu_fitness_cell_t *)calloc(16, sizeof(wubu_fitness_cell_t));
    OUT.graveyard = (wubu_fitness_cell_t *)calloc(16, sizeof(wubu_fitness_cell_t));

    wubu_priority_store_t prio;
    wubu_prio_init(&prio, 0.5f, 0.3f);

    /* L: cell 0 batch 1 (fitness 5.0 — worse), cell 2 batch 3 (only L) */
    add_cell(&L, 1, 0, 5.0f, 1);
    add_cell(&L, 3, 2, 4.0f, 1);
    /* R: cell 0 batch 1 (fitness 3.0 — better), cell 4 batch 5 (only R) */
    add_cell(&R, 1, 0, 3.0f, 1);
    add_cell(&R, 5, 4, 6.0f, 1);
    /* graveyards: L rejected batch 2, R rejected batch 6 */
    wubu_fitness_cell_t g1 = { 0 }, g2 = { 0 };
    g1.batch = 2; g1.verdict = 0; g1.graveyard = 1;
    g2.batch = 6; g2.verdict = 0; g2.graveyard = 1;
    L.graveyard[L.grave_n++] = g1;
    R.graveyard[R.grave_n++] = g2;

    /* 1+2: the union + the conflict rule (R's cell 0 wins: 3.0 < 5.0) */
    wubu_merge_stats_t st;
    wubu_hive_merge(&L, &R, NULL, NULL, &OUT, &st);
    printf("  merge: %s\n", st.kept_left ? "see stats" : "");
    char sb[256];
    wubu_merge_stats_str(&st, sb, sizeof(sb));
    printf("  stats: %s\n", sb);
    if (OUT.ledger_n != 3) FAIL("the union is wrong (%d cells)", OUT.ledger_n);
    /* cell 0 = 3.0 (the better fitness won) */
    int found_better = 0;
    for (int i = 0; i < OUT.ledger_n; i++)
        if (OUT.ledger[i].cell_idx == 0 && OUT.ledger[i].fitness == 3.0f)
            found_better = 1;
    if (!found_better) FAIL("the better-fitness cell did not win the conflict");

    /* 3. the PRIORITY EVIDENCE override: the LEFT checkpoint protects
     * its cell 0 (Fisher 0.56 + recent rejection — the loss cares)
     * while the RIGHT checkpoint does NOT protect its cell 9. The
     * protected lineage survives the conflict even with worse fitness:
     * the federation keeps what the evidence says is valuable. */
    add_cell(&R, 9, 9, 3.0f, 1);   /* a different cell, better fitness */
    wubu_prio_register(&prio, 0, 2, 0.9f, 1);
    for (int i = 0; i < 20; i++)
        wubu_prio_update_fisher(&prio, 0, 0.8f, 0.1f);
    wubu_prio_record_mutation(&prio, 0, 0);   /* rejected -> protected */
    wubu_priority_store_t prio_r;
    wubu_prio_init(&prio_r, 0.5f, 0.3f);      /* the right store: no protection */
    memset(&OUT, 0, sizeof(OUT));
    OUT.ledger_cap = 16; OUT.grave_cap = 16;
    OUT.ledger = (wubu_fitness_cell_t *)calloc(16, sizeof(wubu_fitness_cell_t));
    OUT.graveyard = (wubu_fitness_cell_t *)calloc(16, sizeof(wubu_fitness_cell_t));
    wubu_merge_stats_t st2;
    wubu_hive_merge(&L, &R, &prio, &prio_r, &OUT, &st2);
    /* L's cell 0 (protected, fit 5.0) must survive over R's cell 9 (3.0) */
    int found_protected = 0;
    for (int i = 0; i < OUT.ledger_n; i++)
        if (OUT.ledger[i].cell_idx == 0 && OUT.ledger[i].fitness == 5.0f)
            found_protected = 1;
    printf("  protected cell 0 (Fisher 0.56 + rejected) survives: %d\n", found_protected);
    if (!found_protected) FAIL("the protected cell did not override the loss gate");
    if (st2.protected_wins < 1) FAIL("the protected-win was not counted");

    /* 4. the graveyards union (negative examples never dropped) */
    if (OUT.grave_n != 2) FAIL("the graveyards did not union (%d)", OUT.grave_n);
    printf("  graveyards union: %d rejections preserved\n", OUT.grave_n);

    printf("=== ALL HIVE-MERGE TESTS PASSED (the colony federates) ===\n");
    return 0;
}
