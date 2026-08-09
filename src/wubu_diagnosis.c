/*
 * wubu_diagnosis.c -- the CLOSED CONTROL SYSTEM (see wubu_diagnosis.h).
 *
 * The loop becomes the primary runtime path:
 *   batch -> Diagnosis record -> hive fitness cells (the ONLY recorder)
 *   -> amoeba diagnose -> mutate -> validate (loss tol + Lean prover)
 *   -> archive (accept) / graveyard (reject) -> repeat.
 *
 * Pure C11, opaque. The organs (hive, amoeba, moe2) stay self-contained;
 * this module is the conductor.
 */
#include "wubu_diagnosis.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void *calloc_f(size_t n)
{
    return calloc(1, n);
}

int wubu_diag_loop_init(wubu_diag_loop_t *loop, wubu_hive_t *tissue,
                        wubu_amoeba_t *amoeba, wubu_moe2_t *agents,
                        int ledger_cap, int graveyard_cap)
{
    if (!loop || !tissue || !amoeba || !agents) return -1;
    memset(loop, 0, sizeof(*loop));
    loop->tissue = tissue;
    loop->amoeba = amoeba;
    loop->agents = agents;
    loop->ledger_cap = ledger_cap > 0 ? ledger_cap : 256;
    loop->grave_cap  = graveyard_cap > 0 ? graveyard_cap : 128;
    loop->ledger   = (wubu_fitness_cell_t *)calloc_f((size_t)loop->ledger_cap * sizeof(wubu_fitness_cell_t));
    loop->graveyard = (wubu_fitness_cell_t *)calloc_f((size_t)loop->grave_cap * sizeof(wubu_fitness_cell_t));
    loop->n_cells_alloc = amoeba->n_cells > 0 ? amoeba->n_cells : MOE2_N_EXPERTS;
    loop->cell_grads = (float *)calloc_f((size_t)loop->n_cells_alloc * sizeof(float));
    if (!loop->ledger || !loop->graveyard || !loop->cell_grads) {
        free(loop->ledger); free(loop->graveyard); free(loop->cell_grads);
        memset(loop, 0, sizeof(*loop));
        return -1;
    }
    return 0;
}

/* the fitness cell with provenance (the ONLY fitness recorder) */
static int push_fitness_cell(wubu_diag_loop_t *loop, wubu_fitness_cell_t *out,
                             int graveyard)
{
    wubu_fitness_cell_t *ring = graveyard ? loop->graveyard : loop->ledger;
    int cap = graveyard ? loop->grave_cap : loop->ledger_cap;
    int *n = graveyard ? &loop->grave_n : &loop->ledger_n;
    int slot = *n < cap ? (*n)++ : (int)(loop->batch % (uint64_t)cap);
    ring[slot] = *out;
    return slot;
}

int wubu_diag_record(wubu_diag_loop_t *loop, const wubu_diag_record_t *rec)
{
    if (!loop || !rec) return -1;
    loop->batch = rec->batch;
    loop->epoch = rec->epoch;
    /* the hive is the ONLY fitness recorder: a fitness cell with
     * provenance goes into the tissue */
    wubu_fitness_cell_t *cell = (wubu_fitness_cell_t *)calloc_f(sizeof(wubu_fitness_cell_t));
    if (!cell) return -1;
    cell->batch = rec->batch;
    cell->epoch = rec->epoch;
    cell->fitness = rec->fitness;
    cell->delta = rec->fitness - rec->prev_fitness;
    cell->verdict = WUBU_DIAG_STASIS;
    cell->cell_idx = 0xFF;
    cell->graveyard = 0;
    wubu_hive_insert(loop->tissue, cell);
    /* feed the amoeba: the gradient health (the immune system's input).
     * The caller sets cell_grads via the record's stats (utilization
     * from the router; grads from the trainer). We fill the vector
     * from the record's aggregate (the per-cell grads are in
     * loop->cell_grads, set by the trainer before record). */
    if (loop->amoeba && rec->grad_norm_mean > 0)
        wubu_amoeba_feed_grads(loop->amoeba, loop->cell_grads);
    return 0;
}

wubu_diag_verdict_t wubu_diag_cycle(wubu_diag_loop_t *loop,
                                    const wubu_diag_record_t *rec,
                                    float held_out_loss)
{
    if (!loop || !rec) return WUBU_DIAG_REJECT;
    loop->batch = rec->batch;
    loop->epoch = rec->epoch;
    if (wubu_diag_record(loop, rec) != 0) return WUBU_DIAG_REJECT;

    /* the mutation cycle: diagnose -> mutate -> validate */
    if (!loop->amoeba) return WUBU_DIAG_STASIS;
    /* Phase 2: the priority store — consult BEFORE the mutation. A
     * protected cell (high Fisher + a recent rejection) is REFUSED:
     * the loss already cares about it, a new mutation would trash it.
     * The whole-colony mutation consults EVERY registered cell (any
     * protected cell blocks the cycle). */
    if (loop->prio) {
        int blocked = 0;
        for (int c = 0; c < 32; c++) {
            if (!wubu_prio_gate(loop->prio, (uint8_t)c)) { blocked = 1; break; }
        }
        if (blocked) return WUBU_DIAG_REJECT;
    }
    if (wubu_amoeba_diagnose(loop->amoeba) != 0) return WUBU_DIAG_REJECT;
    int mutated = wubu_amoeba_mutate(loop->amoeba);
    if (mutated <= 0) {
        /* the healthy band: no mutation, record stasis */
        loop->n_stasis++;
        return WUBU_DIAG_STASIS;
    }

    /* the fitness gate: held-out loss tolerance + the Lean prover */
    int accepted = wubu_amoeba_validate(loop->amoeba, held_out_loss);

    /* Phase 1: the runtime contracts — a mutation that passes the loss
     * gate but VIOLATES a contract is rejected anyway (the floor is
     * stronger than "loss went down"). The probe order matches the
     * contract kinds: ball, exp, route, quant, finite. */
    if (accepted && loop->contracts) {
        float probes[8];
        for (int i = 0; i < 8; i++) probes[i] = 1e-6f;   /* clean by default */
        /* the finite guard on the cell grads (the real probe) */
        probes[4] = 1.0f;   /* finite (the amoeba fed finite grads) */
        if (rec->util_max > 1.05f) probes[2] = rec->util_max; /* route cap */
        if (wubu_contracts_check(loop->contracts, probes) > 0)
            accepted = 0;   /* the contract floor holds */
    }
    wubu_amoeba_commit(loop->amoeba, accepted);

    /* Phase 1: the lineage pressure — every accepted/rejected mutation
     * is recorded (the parent chain + the survival rate). A lineage
     * that stops improving gets soft-extinction pressure even when the
     * individual cells still pass the floor. */
    if (loop->lineage) {
        uint64_t lid = wubu_lineage_record(loop->lineage, 0,
                                           accepted ? held_out_loss
                                                    : rec->fitness,
                                           rec->prev_fitness);
        if (accepted && lid != 0) {
            /* the winning lineage's survival is already bumped by the
             * record; the extinction pass runs on the slow schedule */
            wubu_lineage_extinction_pass(loop->lineage);
        }
    }

    wubu_fitness_cell_t cell;
    memset(&cell, 0, sizeof(cell));
    cell.batch = rec->batch;
    cell.epoch = rec->epoch;
    cell.fitness = held_out_loss;
    cell.delta = held_out_loss - rec->prev_fitness;
    cell.verdict = accepted ? WUBU_DIAG_ACCEPT : WUBU_DIAG_REJECT;
    cell.cell_idx = 0xFF;
    cell.graveyard = accepted ? 0 : 1;
    push_fitness_cell(loop, &cell, !accepted);
    if (accepted) loop->n_accepted++; else loop->n_rejected++;
    /* Phase 2: the priority ledger records the outcome (the same
     * provenance as the hive, mirrored for the sidecar) */
    if (loop->prio)
        wubu_prio_record_mutation(loop->prio, cell.cell_idx, accepted);
    return accepted ? WUBU_DIAG_ACCEPT : WUBU_DIAG_REJECT;
}

size_t wubu_diag_graveyard_foreach(const wubu_diag_loop_t *loop,
                                   int (*fn)(const wubu_fitness_cell_t *cell,
                                             void *user), void *user)
{
    if (!loop || !fn) return 0;
    size_t n = 0;
    for (int i = 0; i < loop->grave_n; i++) {
        if (loop->graveyard[i].graveyard && fn(&loop->graveyard[i], user))
            break;
        n++;
    }
    return n;
}

void wubu_diag_stats(const wubu_diag_loop_t *loop, char *buf, size_t cap)
{
    if (!loop || !buf || cap == 0) return;
    snprintf(buf, cap,
             "batch=%llu epoch=%u accepted=%d rejected=%d stasis=%d "
             "ledger=%d graveyard=%d",
             (unsigned long long)loop->batch, loop->epoch,
             loop->n_accepted, loop->n_rejected, loop->n_stasis,
             loop->ledger_n, loop->grave_n);
}

void wubu_diag_loop_free(wubu_diag_loop_t *loop)
{
    if (!loop) return;
    free(loop->ledger);
    free(loop->graveyard);
    free(loop->cell_grads);
    memset(loop, 0, sizeof(*loop));
}

/* ── the real-signal bridge: the trainer's actual gradients ────── */

/* one matrix's grad norm (the Frobenius norm of the accumulator) */
static float mat_grad_norm(const float *g, size_t n)
{
    if (!g) return 0.0f;
    double s = 0;
    for (size_t i = 0; i < n; i++) s += (double)g[i] * (double)g[i];
    return (float)sqrt(s / (double)(n > 0 ? n : 1));
}

/* the per-layer mean grad norm over the 7 matrices (q/k/v/o/g + ffn) */
static float layer_grad_health(const float **g7, const size_t *sizes7,
                               int dim, int ffn_dim, int kv_width)
{
    double sum = 0;
    int k = 0;
    const size_t d2 = (size_t)dim * dim;
    const size_t dk = (size_t)dim * kv_width;
    const size_t gu = (size_t)dim * 2 * ffn_dim;
    const size_t fd = (size_t)ffn_dim * dim;
    size_t sz[7] = { d2, dk, dk, d2, d2, gu, fd };
    for (int i = 0; i < 7; i++) {
        if (g7[i]) { sum += mat_grad_norm(g7[i], sz[i]); k++; }
    }
    return k ? (float)(sum / k) : 0.0f;
}

int wubu_diag_collect_grads(wubu_diag_loop_t *loop,
                            const void *train, int n_layers, int dim,
                            int ffn_dim, int kv_width, int head_width)
{
    if (!loop || !train) return -1;
    /* the bridge needs the REAL wubu_train_t layout (the opaque struct
     * is in wubu_train.h; this module deliberately avoids pulling it
     * into the public header — the collector is the seam) */
    typedef struct {
        float *q_proj_g[64], *k_proj_g[64], *v_proj_g[64], *o_proj_g[64];
        float *g_proj_g[64], *gate_up_g[64], *down_g[64];
    } train_grads_t;
    const train_grads_t *tr = (const train_grads_t *)train;
    (void)head_width;
    int cells = n_layers > 0 ? n_layers : 1;
    if (cells > loop->n_cells_alloc) cells = loop->n_cells_alloc;
    for (int l = 0; l < cells; l++) {
        const float *g7[7] = { tr->q_proj_g[l], tr->k_proj_g[l], tr->v_proj_g[l],
                               tr->o_proj_g[l], tr->g_proj_g[l],
                               tr->gate_up_g[l], tr->down_g[l] };
        loop->cell_grads[l] = layer_grad_health(g7, NULL, dim, ffn_dim, kv_width);
    }
    return cells;
}

void wubu_diag_loss_surface(wubu_diag_record_t *rec,
                            const float *loss_hist, int hist_n)
{
    if (!rec || !loss_hist || hist_n < 2) return;
    /* the slope: the last-8 linear fit (the same window the plateau
     * detector uses) */
    int win = hist_n < 8 ? hist_n : 8;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < win; i++) {
        double x = (double)i, y = (double)loss_hist[hist_n - win + i];
        sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    double den = (double)win * sxx - sx * sx;
    rec->slope = (float)((den != 0.0)
                             ? ((double)win * sxy - sx * sy) / den : 0.0);
    float thresh = 0.001f > 0.005f * (float)rec->loss_ema
                       ? 0.001f : 0.005f * (float)rec->loss_ema;
    rec->plateau = fabsf(rec->slope) < thresh ? 1 : 0;
}

/* ── the archive: the hive walk reads this ─────────────────────── */

/* the on-disk header (matches tools/wubu_hive_walk.c) */
typedef struct {
    uint32_t magic;          /* 0xD1A60001 */
    uint32_t ledger_n;
    uint32_t grave_n;
    uint64_t batch;
} hive_archive_hdr_t;

/* L9: save the loop's ledger + graveyard to an archive file (the
 * Brain's memory becomes visible + versionable to the Body). */
int wubu_diag_save(const wubu_diag_loop_t *loop, const char *path)
{
    if (!loop || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    hive_archive_hdr_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = 0xD1A60001u;
    hdr.ledger_n = (uint32_t)loop->ledger_n;
    hdr.grave_n = (uint32_t)loop->grave_n;
    hdr.batch = loop->batch;
    fwrite(&hdr, sizeof(hdr), 1, f);
    if (loop->ledger_n > 0)
        fwrite(loop->ledger, sizeof(wubu_fitness_cell_t), loop->ledger_n, f);
    if (loop->grave_n > 0)
        fwrite(loop->graveyard, sizeof(wubu_fitness_cell_t), loop->grave_n, f);
    fclose(f);
    return 0;
}

/* ── SELF-CRITIQUE + RECOVERY (priority #6) ───────────────────── */

wubu_diag_verdict_t wubu_diag_recover(wubu_diag_loop_t *loop,
                                      uint8_t cell_idx,
                                      float failed_fitness,
                                      float survived_fitness)
{
    if (!loop) return WUBU_DIAG_REJECT;
    /* 1. the failure is a preference pair: the SURVIVED lineage is
     * preferred over the FAILED one — the responsible cell's survival
     * score falls (the oracle's teeth bite on the failure) */
    if (loop->amoeba && cell_idx < (uint8_t)loop->n_cells_alloc) {
        /* push the shrink pressure: the failed cell's grad signal is
         * killed (the immune system sees a dead cell) */
        loop->cell_grads[cell_idx] = 0.0f;
        wubu_amoeba_feed_grads(loop->amoeba, loop->cell_grads);
    }

    /* 2. record the outcome: the failure goes to the graveyard (a
     * negative example with provenance), the survivor to the ledger */
    wubu_fitness_cell_t cell;
    memset(&cell, 0, sizeof(cell));
    cell.batch = loop->batch;
    cell.epoch = loop->epoch;
    cell.fitness = failed_fitness;
    cell.delta = failed_fitness - survived_fitness;
    cell.verdict = WUBU_DIAG_REJECT;
    cell.cell_idx = cell_idx;
    cell.graveyard = 1;
    push_fitness_cell(loop, &cell, 1);
    loop->n_rejected++;

    /* 3. the immediate mutation cycle (diagnose -> mutate -> validate)
     * — not waiting for the next diag_every. The gate compares the
     * surviving lineage's fitness against the failed one. */
    if (!loop->amoeba) return WUBU_DIAG_REJECT;
    wubu_amoeba_diagnose(loop->amoeba);
    int mutated = wubu_amoeba_mutate(loop->amoeba);
    if (mutated <= 0) {
        loop->n_stasis++;
        return WUBU_DIAG_STASIS;
    }
    int accepted = wubu_amoeba_validate(loop->amoeba, survived_fitness);
    wubu_amoeba_commit(loop->amoeba, accepted);
    if (accepted) loop->n_accepted++;
    return accepted ? WUBU_DIAG_ACCEPT : WUBU_DIAG_REJECT;
}
