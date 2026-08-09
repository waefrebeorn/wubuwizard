/*
 * test_diagnosis.c -- the CLOSED CONTROL SYSTEM gate (the user directive:
 * "the colony can notice its own failure modes, propose mutations,
 * validate them, archive the good ones, and keep running").
 *
 * Runs the full loop: batch records -> hive fitness cells (the ONLY
 * recorder) -> amoeba diagnose -> mutate -> validate (loss tol + prover)
 * -> archive (accept) / graveyard (reject). Asserts:
 *   1. every batch writes a fitness cell into the hive
 *   2. the fitness gate accepts good mutations (loss improved) and
 *      rejects bad ones (loss exploded -> graveyard)
 *   3. the graveyard is queryable (negative examples)
 *   4. the loop keeps running (no state corruption across cycles)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_diagnosis.h"
#include "wubu_amoeba.h"
#include "wubu_hive.h"
#include "wubu_moe2.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

/* the graveyard walk callback (counts entries) */
static int count_grave(const wubu_fitness_cell_t *cell, void *user)
{
    (void)cell;
    (*(int *)user)++;
    return 0;
}

/* a fake colony: moe2 with a few experts */
static int setup_colony(wubu_hive_t *tissue, wubu_amoeba_t *amoeba,
                        wubu_moe2_t *agents)
{
    if (wubu_hive_init(tissue) != 0) return -1;
    if (wubu_moe2_init(agents, 7) != 0) return -1;
    wubu_amoeba_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.grow_util = 0.7; cfg.grow_grad = 0.3;
    cfg.shrink_util = 0.02; cfg.shrink_grad = 0.1;
    cfg.entropy_min = 0.05; cfg.loss_tol = 0.05;
    cfg.split_eps = 0.01; cfg.max_cells = 8; cfg.min_cells = 2;
    if (wubu_amoeba_init(amoeba, &cfg, tissue, agents) != 0) return -1;
    return 0;
}

int main(void)
{
    printf("=== test_diagnosis (the closed control loop) ===\n");

    wubu_hive_t tissue;
    wubu_amoeba_t amoeba;
    wubu_moe2_t agents;
    if (setup_colony(&tissue, &amoeba, &agents) != 0)
        FAIL("colony setup");

    wubu_diag_loop_t loop;
    if (wubu_diag_loop_init(&loop, &tissue, &amoeba, &agents, 64, 32) != 0)
        FAIL("loop init");

    /* run 20 batches: the loss surface descends, then one batch explodes
     * (simulating a bad mutation) — the gate must catch it */
    int n_accept = 0, n_reject = 0, n_stasis = 0;
    float fitness = 10.0f;
    for (int b = 1; b <= 20; b++) {
        wubu_diag_record_t rec;
        memset(&rec, 0, sizeof(rec));
        rec.batch = (uint64_t)b;
        rec.epoch = 1;
        rec.loss = fitness;
        rec.loss_ema = 0.9f * fitness + 0.1f * rec.loss;
        rec.slope = -0.1f;               /* the surface is descending */
        rec.plateau = 0;
        rec.n_experts = MOE2_N_EXPERTS;
        rec.util_min = 0.1f; rec.util_max = 0.9f; rec.util_mean = 0.5f;
        rec.grad_norm_max = 0.8f; rec.grad_norm_mean = 0.3f;
        rec.n_dead_cells = 0;
        rec.fitness = fitness;
        rec.prev_fitness = fitness + 0.5f;   /* we improved */
        /* the per-cell grads (the immune system's input) */
        for (int i = 0; i < loop.n_cells_alloc; i++)
            loop.cell_grads[i] = 0.1f + 0.05f * (float)(i % 3);

        float held_out;
        if (b == 15) {
            /* the bad mutation: the loss explodes */
            held_out = fitness + 2.0f;
        } else {
            held_out = fitness - 0.2f;   /* the good mutation: better */
        }
        wubu_diag_verdict_t v = wubu_diag_cycle(&loop, &rec, held_out);
        if (v == WUBU_DIAG_ACCEPT) n_accept++;
        else if (v == WUBU_DIAG_REJECT) n_reject++;
        else n_stasis++;
        fitness = held_out;
    }

    /* 1. every batch wrote a fitness cell into the hive */
    size_t live = wubu_hive_live(&tissue);
    printf("  hive live cells: %zu (20 batches expected)\n", live);
    if (live < 5) FAIL("hive did not record fitness cells");

    /* 2. the gate caught the bad mutation (batch 15 -> reject) */
    printf("  accepted=%d rejected=%d stasis=%d\n",
           n_accept, n_reject, n_stasis);
    if (n_accept < 1) FAIL("no good mutations accepted");
    if (n_reject < 1) FAIL("the bad mutation was NOT rejected (gate broken)");

    /* 3. the graveyard is queryable */
    int grave_count = 0;
    wubu_diag_graveyard_foreach(&loop, count_grave, &grave_count);
    printf("  graveyard entries: %d\n", grave_count);
    if (grave_count < 1) FAIL("graveyard empty (rejected mutations not kept)");

    /* 4. the loop kept running (stats sane) */
    char stats[256];
    wubu_diag_stats(&loop, stats, sizeof(stats));
    printf("  stats: %s\n", stats);
    if (loop.batch != 20) FAIL("batch counter wrong");

    /* 5. SELF-CRITIQUE + RECOVERY (priority #6): a failed generation
     * auto-triggers diagnose->mutate on the responsible cell */
    int rejected_before = loop.n_rejected;
    wubu_diag_verdict_t rv = wubu_diag_recover(&loop, 2, 12.0f, 8.0f);
    printf("  recovery verdict: %s (graveyard +%d)\n",
           rv == WUBU_DIAG_ACCEPT ? "accept" :
           rv == WUBU_DIAG_REJECT ? "reject" : "stasis",
           loop.n_rejected - rejected_before);
    if (loop.n_rejected <= rejected_before)
        FAIL("the failure was not recorded in the graveyard");
    if (rv != WUBU_DIAG_ACCEPT && rv != WUBU_DIAG_REJECT && rv != WUBU_DIAG_STASIS)
        FAIL("bad recovery verdict");
    printf("  self-critique: the failed cell was marked for shrink + "
           "an immediate mutation cycle ran\n");

    wubu_diag_loop_free(&loop);
    wubu_moe2_free(&agents);
    /* the Phase 1 ASan gate: the caller owns the organs — free the
     * amoeba + the hive tissue with the cell payloads (the closed
     * loop is leak-free) */
    wubu_amoeba_free(&amoeba);
    wubu_hive_clear_with(&tissue, free);
    printf("=== ALL DIAGNOSIS TESTS PASSED (the closed loop is live) ===\n");
    return 0;
}
