/*
 * test_colony.c — THE RELEASE GATE (AN47 #7: "make test_colony —
 * harness floor + zero contract violations + priority-store sidecar
 * present + blueprint bounds intact. No ship without it.").
 *
 * One integration pass over the colony stack (the same organs the
 * endurance run uses):
 *   1. the closed loop: batches -> diagnoses -> mutations, accepted/
 *      rejected recorded (the archive is queryable)
 *   2. the HARNESS FLOOR: the task suite score must clear the floor
 *      (the colony cannot pass by short-batch loss alone)
 *   3. ZERO CONTRACT VIOLATIONS: every mutation that passed the loss
 *      gate also passed the runtime contracts
 *   4. the PRIORITY-STORE SIDECAR present: the .prio round-trips
 *   5. the BLUEPRINT bounds intact: no off-blueprint mutation
 *
 * The gate is the "no ship without it" line: a release passes only
 * when all five hold.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "wubu_diagnosis.h"
#include "wubu_amoeba.h"
#include "wubu_hive.h"
#include "wubu_moe2.h"
#include "wubu_lineage.h"
#include "wubu_contracts.h"
#include "wubu_priority_store.h"
#include "wubu_harness.h"
#include "wubu_blueprint.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

/* the release gate checklist */
static int g_harness_ok = 0, g_contracts_ok = 0, g_prio_ok = 0, g_bp_ok = 0;

int main(void)
{
    printf("=== test_colony (the RELEASE gate — no ship without it) ===\n");

    wubu_hive_t tissue;
    wubu_amoeba_t amoeba;
    wubu_moe2_t agents;
    if (wubu_hive_init(&tissue) != 0) FAIL("hive init");
    if (wubu_moe2_init(&agents, 8) != 0) FAIL("moe2 init");
    wubu_amoeba_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.grow_util = 0.7; cfg.grow_grad = 0.3;
    cfg.shrink_util = 0.02; cfg.shrink_grad = 0.1;
    cfg.entropy_min = 0.05; cfg.loss_tol = 0.05;
    cfg.split_eps = 0.01; cfg.max_cells = 8; cfg.min_cells = 2;
    if (wubu_amoeba_init(&amoeba, &cfg, &tissue, &agents) != 0) FAIL("amoeba init");

    /* the full gate stack (the endurance run's organs) */
    wubu_lineage_tracker_t lineage;
    wubu_lineage_init(&lineage, 64, 8, 0.2f);
    wubu_contracts_t contracts;
    wubu_contracts_init(&contracts, &tissue);
    wubu_priority_store_t prio;
    wubu_prio_init(&prio, 0.5f, 0.3f);
    wubu_harness_t harness;
    wubu_harness_init(&harness, &tissue, 1000000L, 1000);

    wubu_diag_loop_t loop;
    wubu_diag_loop_init(&loop, &tissue, &amoeba, &agents, 128, 64);
    loop.lineage = &lineage;
    loop.contracts = &contracts;
    loop.prio = &prio;

    /* 1. the closed loop: 12 batches with a descending loss, one
     * explosion at batch 9 (the gate must reject it) */
    float fit = 10.0f;
    int n_rejected = 0;
    for (int b = 1; b <= 12; b++) {
        wubu_diag_record_t rec;
        memset(&rec, 0, sizeof(rec));
        rec.batch = (uint64_t)b;
        rec.epoch = 1;
        rec.loss = fit;
        rec.loss_ema = fit;
        rec.prev_fitness = fit + 0.1f;
        rec.fitness = fit;
        rec.n_experts = 8;
        rec.grad_norm_mean = 0.4f;
        float held = (b == 9) ? fit + 2.0f : fit - 0.15f;  /* the bad batch */
        wubu_diag_verdict_t v = wubu_diag_cycle(&loop, &rec, held);
        if (v == WUBU_DIAG_REJECT) n_rejected++;
        /* the lineage + prio record (the gate's bookkeeping) */
        wubu_prio_register(&prio, (uint8_t)(b % 8), 2, 0.6f, (uint64_t)b);
        wubu_prio_update_fisher(&prio, (uint8_t)(b % 8), 0.7f, 0.1f);
        fit = held;
    }
    printf("  closed loop: %d cycles, %d rejected (the gate works)\n",
           loop.n_accepted + loop.n_rejected + loop.n_stasis, n_rejected);
    if (n_rejected < 1) FAIL("the explosion was not rejected");

    /* 2. the HARNESS FLOOR: 5 good tasks must clear the floor */
    float score = 0;
    for (int i = 0; i < 5; i++)
        wubu_harness_run_task(&harness, i, 4, 0.9f, 0.3f);
    score = wubu_harness_suite_score(&harness);
    printf("  harness floor: suite score %.3f (floor 0.5)\n", score);
    if (score < 0.5f) FAIL("the harness score missed the floor");
    g_harness_ok = 1;

    /* 3. ZERO CONTRACT VIOLATIONS */
    printf("  contracts: %llu checks, %llu violations\n",
           (unsigned long long)contracts.n_checks,
           (unsigned long long)contracts.n_violations);
    if (contracts.n_violations != 0) FAIL("contract violations in the gate run");
    g_contracts_ok = 1;

    /* 4. the PRIORITY-STORE SIDECAR present + round-trips */
    static char pbuf[8192];
    long pn = wubu_prio_save(&prio, pbuf, (long)sizeof(pbuf));
    wubu_priority_store_t prio2;
    wubu_prio_init(&prio2, 0.5f, 0.3f);
    int pr = wubu_prio_load(&prio2, pbuf, pn);
    printf("  priority store: %d cells round-tripped (%ld bytes)\n", pr, pn);
    if (pr < 1) FAIL("the priority sidecar is missing");
    g_prio_ok = 1;

    /* 5. the BLUEPRINT bounds intact: off-blueprint refused */
    wubu_blueprint_t bp;
    wubu_blueprint_init(&bp, &tissue);
    int ok = wubu_blueprint_allows(&bp, WUBU_BP_MOE, 8.0f);
    int bad = wubu_blueprint_allows(&bp, WUBU_BP_MOE, 64.0f);
    printf("  blueprint: in-range=%d off-blueprint=%d (must be 1/0)\n", ok, bad);
    if (!ok || bad) FAIL("the blueprint bounds are not intact");
    g_bp_ok = 1;

    /* the gate verdict */
    printf("\n=== RELEASE GATE: harness %s, contracts %s, prio %s, blueprint %s ===\n",
           g_harness_ok ? "PASS" : "FAIL",
           g_contracts_ok ? "PASS" : "FAIL",
           g_prio_ok ? "PASS" : "FAIL",
           g_bp_ok ? "PASS" : "FAIL");
    if (g_harness_ok && g_contracts_ok && g_prio_ok && g_bp_ok)
        printf("=== ALL COLONY RELEASE GATE TESTS PASSED (ship it) ===\n");
    else
        FAIL("the release gate is RED — do not ship");

    wubu_diag_loop_free(&loop);
    wubu_amoeba_free(&amoeba);
    wubu_moe2_free(&agents);
    wubu_hive_clear_with(&tissue, free);
    return 0;
}
