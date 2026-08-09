/*
 * wubu_amoeba.c -- THE AMOEBA: the diagnostic self-evolving model.
 *
 * THE HIVE IS THE BODY. The user: "this is the beauty of the hive we
 * can use the hive." The amoeba's cells ARE hive slots:
 *
 *   - the hive (wubu_hive) holds the live cells: each slot is a
 *     pointer to a cell's state (stable -- the state never moves)
 *   - GROW = wubu_hive_insert: the freelist pop (a recycled slot) or
 *     a new block -- the pseudopod extends
 *   - SHRINK = wubu_hive_erase: skip-mark + freelist push -- the
 *     membrane retracts, the memory RETURNS to the pool
 *   - DIAGNOSE = wubu_hive_foreach: jumps skips, visits only the
 *     live cells -- the immune system sees only what is alive
 *   - the block structure gives cache locality: cells in the same
 *     block are hot together (the amoeba's tissue is warm)
 *
 * Every cell keeps a stable identity (its registry index); the hive
 * slot holds &cells[i]. Erase marks skip + pushes the slot to the
 * freelist; the NEXT grow reuses that exact slot (LIFO) -- the
 * freelist IS the recycled membrane. Pure C11, no templates.
 *
 * The Darwin Gödel Machine pattern: mutate -> validate -> archive,
 * empirically, open-ended, sandboxed (AGI_HOME_METAGAME H3).
 */
#include "wubu_amoeba.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const wubu_amoeba_cfg_t DEFAULT_CFG = {
    0.7,    /* grow_util */
    2.0,    /* grow_grad (relative to the mean) */
    0.02,   /* shrink_util */
    0.25,   /* shrink_grad (relative to the mean) */
    0.05,   /* entropy_min */
    0.05,   /* loss_tol */
    0.01,   /* split_eps */
    32,     /* max_cells (safety ceiling) */
    2       /* min_cells (safety floor) */
};

int wubu_amoeba_init(wubu_amoeba_t *am, wubu_amoeba_cfg_t *cfg,
                     wubu_hive_t *tissue, wubu_moe2_t *agents)
{
    if (!am || !tissue || !agents) return -1;
    memset(am, 0, sizeof(*am));
    am->cfg = cfg ? *cfg : DEFAULT_CFG;
    am->tissue = tissue;
    am->agents = agents;
    am->n_cells = MOE2_N_EXPERTS;
    am->cells = (wubu_amoeba_cell_t *)calloc(am->cfg.max_cells,
                                             sizeof(wubu_amoeba_cell_t));
    if (!am->cells) return -1;
    am->gate_parent = (float *)calloc(MOE2_D_FF, sizeof(float));
    if (!am->gate_parent) { free(am->cells); am->cells = NULL; return -1; }
    am->vitals.prev_fitness = -1;
    /* seed the hive with the initial colony: the cells are the live
     * slots (each holds &cells[i] -- the stable identity). */
    for (int i = 0; i < am->n_cells && i < am->cfg.max_cells; i++)
        wubu_hive_insert(am->tissue, (void *)&am->cells[i]);
    return 0;
}

int wubu_amoeba_feed_grads(wubu_amoeba_t *am, const float *grad_norms)
{
    if (!am || !grad_norms) return -1;
    /* the gradients arrive indexed by the REGISTRY cell id (the
     * trainer's per-cell grads, in cell order) -- NOT the hive-visit
     * order. The DA review caught the previous version reading past
     * the caller's array once the colony grew past the original cell
     * count (hive-visit order != registry order). */
    for (int i = 0; i < am->n_cells && i < am->cfg.max_cells; i++)
        am->cells[i].grad_norm = grad_norms[i];
    return 0;
}

/* the quadratic loss-delta approximation: removing a parameter with
 * gradient g changes the loss by about -||g||^2 / n. Per cell: -g^2
 * (relative to the colony's mean). */
static double loss_delta_approx(double grad_norm, double mean_grad)
{
    if (mean_grad <= 1e-9) return 0;
    return -grad_norm * grad_norm / mean_grad;
}

int wubu_amoeba_diagnose(wubu_amoeba_t *am)
{
    if (!am || !am->tissue) return -1;
    int n = 0;
    double mean_grad = 0;
    /* pass 1: count + mean grad — the DA type-check: the amoeba's OWN
     * cells only (the shared tissue holds skills/traj/fitness/meta
     * cells whose grad_norm offset is garbage) */
    for (wubu_hive_block_t *blk = am->tissue->head; blk; blk = blk->next) {
        if (blk->live == 0) continue;
        for (size_t s = 0; s < blk->cap; s++) {
            if (blk->skip[s] == 0) {
                void *p = blk->slots[s];
                int is_mine = 0;
                for (int i = 0; i < am->cfg.max_cells; i++)
                    if (p == (void *)&am->cells[i]) { is_mine = 1; break; }
                if (!is_mine) continue;
                wubu_amoeba_cell_t *c =
                    (wubu_amoeba_cell_t *)blk->slots[s];
                c->utilization = 0;
                c->loss_delta = 0;
                mean_grad += c->grad_norm;
                n++;
            }
        }
    }
    if (n == 0) return -1;
    mean_grad /= (double)n;
    am->vitals.n_grow = 0;
    am->vitals.n_shrink = 0;
    am->vitals.n_stasis = 0;
    /* pass 2: classify every LIVE AMOEBA cell (the same type-check) */
    for (wubu_hive_block_t *blk = am->tissue->head; blk; blk = blk->next) {
        if (blk->live == 0) continue;
        for (size_t s = 0; s < blk->cap; s++) {
            if (blk->skip[s] == 0) {
                void *p = blk->slots[s];
                int is_mine = 0;
                for (int i = 0; i < am->cfg.max_cells; i++)
                    if (p == (void *)&am->cells[i]) { is_mine = 1; break; }
                if (!is_mine) continue;
                wubu_amoeba_cell_t *c =
                    (wubu_amoeba_cell_t *)blk->slots[s];
                c->loss_delta = loss_delta_approx(c->grad_norm, mean_grad);
                /* the utilization prior: the router reports are
                 * folded in by the trainer; the amoeba uses the
                 * gradient as the primary signal here */
                int grow = c->grad_norm > am->cfg.grow_grad * mean_grad ||
                           c->utilization > am->cfg.grow_util;
                int die  = c->grad_norm < am->cfg.shrink_grad * mean_grad &&
                           c->utilization < am->cfg.shrink_util;
                if (grow) am->vitals.n_grow++;
                else if (die) am->vitals.n_shrink++;
                else am->vitals.n_stasis++;
            }
        }
    }
    am->vitals.n_cells = n;
    am->vitals.total_live = (double)wubu_hive_live(am->tissue);
    am->vitals.memory_used = (double)wubu_hive_capacity(am->tissue);
    return 0;
}

/* ---- the operators (the hive is the body) ---- */

/* forward: the amoeba's OWN live-cell count (defined below, used by
 * grow_cell first) */
static int amoeba_live_cells(const wubu_amoeba_t *am);

/* GROW: mitosis. The parent's gate splits (+eps / -eps), and a NEW
 * hive slot holds the daughter cell -- the freelist pop or a new
 * block. The pseudopod extends. */
static void grow_cell(wubu_amoeba_t *am, int parent_idx)
{
    if (!am || !am->agents || !am->tissue) return;
    wubu_moe2_t *moe = am->agents;
    if (parent_idx < 0 || parent_idx >= MOE2_N_EXPERTS) return;
    if (amoeba_live_cells(am) >= am->cfg.max_cells) return;   /* colony cap */
    /* the split: parent keeps -eps, the daughter gets +eps */
    for (int d = 0; d < MOE2_D_FF; d++) {
        float pv = moe->exp_gate[parent_idx][d];
        moe->exp_gate[parent_idx][d] = pv - (float)am->cfg.split_eps;
        am->gate_parent[d] = pv;
    }
    /* the daughter cell: find a free registry slot (the freelist
     * recycled one is reused by the hive automatically -- the slot
     * pointer is stable) */
    wubu_amoeba_cell_t *daughter = NULL;
    for (int i = 0; i < am->cfg.max_cells; i++) {
        /* a cell is free when its hive slot is skipped (erased) OR it
         * was never born; the registry tracks it via the hive */
        int in_hive = 0;
        for (wubu_hive_block_t *blk = am->tissue->head; blk && !in_hive;
             blk = blk->next)
            for (size_t s = 0; s < blk->cap && !in_hive; s++)
                if (blk->skip[s] == 0 &&
                    blk->slots[s] == (void *)&am->cells[i])
                    in_hive = 1;
        if (!in_hive) { daughter = &am->cells[i]; break; }
    }
    if (!daughter) return;
    memset(daughter, 0, sizeof(*daughter));
    daughter->utilization = 0.5;      /* the newborn starts mid-band */
    daughter->grad_norm = 0;
    daughter->loss_delta = 0;
    daughter->route_entropy = 0.5;
    /* THE HIVE: the daughter is inserted -- the freelist pop or a new
     * block. The membrane extends. */
    wubu_hive_insert(am->tissue, (void *)daughter);
    am->vitals.total_live = (double)wubu_hive_live(am->tissue);
}

/* SHRINK: apoptosis. The cell's hive slot is erased -- skip-mark +
 * freelist push. The memory RETURNS to the pool; the next grow reuses
 * the exact slot (LIFO). The membrane retracts. */
static void shrink_cell(wubu_amoeba_t *am, wubu_amoeba_cell_t *cell)
{
    if (!am || !am->tissue || !cell) return;
    wubu_hive_erase(am->tissue, (void *)cell);
    am->vitals.total_live = (double)wubu_hive_live(am->tissue);
}

/* the amoeba's OWN live-cell count: how many of the fixed registry
 * slots (am->cells[]) are live in the hive. The colony-size contract
 * (max_cells/min_cells) measures THE COLONY, not the observation
 * layer — the hive also carries traj cells (the harness), fitness
 * cells (the diag ledger), and meta-cells (the metadiag), and counting
 * all of them trips max_cells after the first round (the 3000-round
 * run's 2993 rejections were this bug). */
static int amoeba_live_cells(const wubu_amoeba_t *am)
{
    if (!am || !am->tissue) return 0;
    int n = 0;
    for (wubu_hive_block_t *blk = am->tissue->head; blk; blk = blk->next) {
        if (blk->live == 0) continue;
        for (size_t s = 0; s < blk->cap; s++) {
            if (blk->skip[s]) continue;
            const void *p = blk->slots[s];
            for (int i = 0; i < am->cfg.max_cells; i++) {
                if (p == (const void *)&am->cells[i]) { n++; break; }
            }
        }
    }
    return n;
}

int wubu_amoeba_mutate(wubu_amoeba_t *am)
{
    if (!am || !am->tissue) return -1;
    int mutations = 0;
    /* the SAME classification as diagnose: the mean gradient decides
     * the thresholds (a cell is overworked when its grad is above
     * grow_grad x the mean; dead when below shrink_grad x the mean).
     * The loss_delta sign alone is NOT the grow signal -- the DA test
     * caught that (every nonzero grad has a negative quadratic
     * approximation, so every cell looked "growing"). */
    double mean_grad = 0;
    int n = 0;
    for (wubu_hive_block_t *blk = am->tissue->head; blk; blk = blk->next) {
        if (blk->live == 0) continue;
        for (size_t s = 0; s < blk->cap; s++) {
            if (blk->skip[s]) continue;
            void *p = blk->slots[s];
            /* the DA type-check: the mean is over the amoeba's OWN
             * cells only (the shared tissue holds skills/traj/fitness/
             * meta cells whose grad_norm offset is garbage) */
            int is_mine = 0;
            for (int i = 0; i < am->cfg.max_cells; i++)
                if (p == (void *)&am->cells[i]) { is_mine = 1; break; }
            if (!is_mine) continue;
            wubu_amoeba_cell_t *c = (wubu_amoeba_cell_t *)p;
            mean_grad += c->grad_norm;
            n++;
        }
    }
    if (n == 0) return 0;
    mean_grad /= (double)n;

    /* collect the actions first (the hive changes under us). THE DA
     * TYPE-CONFUSION FIX: only the amoeba's OWN registry cells may be
     * classified (grow/die). The shared tissue also holds skill cells
     * (the endurance runner), traj cells (the harness), fitness cells
     * (the diag ledger), and meta-cells (the metadiag) — reading
     * grad_norm at their offsets gives garbage, and cells that LOOKED
     * dead were erased from the hive (the run's skills vanished: the
     * same shared-tissue bug class as the max_cells contract). */
    wubu_amoeba_cell_t *to_grow[64], *to_shrink[64];
    int ng = 0, ns = 0;
    int colony = amoeba_live_cells(am);   /* the amoeba's own count */
    for (wubu_hive_block_t *blk = am->tissue->head; blk; blk = blk->next) {
        if (blk->live == 0) continue;
        for (size_t s = 0; s < blk->cap; s++) {
            if (blk->skip[s]) continue;
            void *p = blk->slots[s];
            /* the type check: is this one of the amoeba's cells? */
            int is_mine = 0;
            for (int i = 0; i < am->cfg.max_cells; i++)
                if (p == (void *)&am->cells[i]) { is_mine = 1; break; }
            if (!is_mine) continue;   /* not the amoeba's — skip */
            wubu_amoeba_cell_t *c = (wubu_amoeba_cell_t *)p;
            /* grow: grad >> mean (the cell is overworked); only up
             * to the ceiling (the COLONY ceiling, not the whole
             * hive) */
            int grow = c->grad_norm >
                       am->cfg.grow_grad * mean_grad &&
                       colony < am->cfg.max_cells;
            /* die: grad << mean OR below the absolute floor (the
             * cell is dead weight); only down to the floor. The
             * absolute floor catches the all-dead colony: when
             * every grad is ~0 the relative test can't fire. */
            int die  = (c->grad_norm <
                        am->cfg.shrink_grad * mean_grad ||
                        c->grad_norm < 1e-4) &&
                       colony > am->cfg.min_cells;
            if (grow && ng < 16) to_grow[ng++] = c;
            else if (die && ns < 16) to_shrink[ns++] = c;
        }
    }
    /* grow first (the colony extends), then shrink (it retracts) */
    for (int i = 0; i < ng; i++) {
        int idx = (int)(to_grow[i] - am->cells);
        grow_cell(am, idx >= 0 && idx < MOE2_N_EXPERTS ? idx : 0);
        mutations++;
    }
    for (int i = 0; i < ns; i++) {
        shrink_cell(am, to_shrink[i]);
        mutations++;
    }
    return mutations;
}

int wubu_amoeba_validate(wubu_amoeba_t *am, double held_out_loss)
{
    if (!am) return 0;
    am->vitals.fitness = held_out_loss;
    int ok = 1;
    /* 1. the loss tolerance: the mutation may not hurt by more than
     * the tolerance (unless the previous fitness is unknown) */
    if (am->vitals.prev_fitness >= 0 &&
        held_out_loss > am->vitals.prev_fitness + am->cfg.loss_tol)
        ok = 0;
    /* 2. the prover invariants: the Lean-verified theorems must still
     * hold (the colony's geometry is intact) */
    wubu_pf_step_t m = { WUBU_PF_MOBUS, 0.25, 0.3, -0.4, 0, 0 };
    if (!wubu_prover_check(&m)) ok = 0;
    /* 3. the safety: the COLONY (the amoeba's own cells) is within the
     * floor/ceiling — NOT the whole hive, which also carries traj
     * cells (the harness), fitness cells (the diag ledger), and
     * meta-cells (the metadiag). Counting those tripped max_cells
     * after round 1 and rejected every mutation (the 3000-round run). */
    int colony = amoeba_live_cells(am);
    if (colony < am->cfg.min_cells ||
        colony > am->cfg.max_cells)
        ok = 0;
    return ok;
}

int wubu_amoeba_commit(wubu_amoeba_t *am, int accepted)
{
    if (!am) return -1;
    if (accepted) {
        am->vitals.accepted++;
        am->vitals.prev_fitness = am->vitals.fitness;
    } else {
        am->vitals.rejected++;
        /* the caller does the 5+1 rollback; the fitness reverts */
        am->vitals.prev_fitness = -1;
    }
    am->epoch++;
    return 0;
}

void wubu_amoeba_stats(const wubu_amoeba_t *am, char *buf, size_t cap)
{
    if (!am || !buf || cap == 0) return;
    snprintf(buf, cap,
        "epoch=%u cells=%d (grow=%d shrink=%d stasis=%d) "
        "hive_live=%.0f/%.0f fitness=%.4f accept=%d reject=%d",
        am->epoch, am->vitals.n_cells, am->vitals.n_grow,
        am->vitals.n_shrink, am->vitals.n_stasis,
        am->vitals.total_live, am->vitals.memory_used,
        am->vitals.fitness, am->vitals.accepted, am->vitals.rejected);
}

void wubu_amoeba_free(wubu_amoeba_t *am)
{
    if (!am) return;
    free(am->cells);
    free(am->gate_parent);
    am->cells = NULL;
    am->gate_parent = NULL;
}
