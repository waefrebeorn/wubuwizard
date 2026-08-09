/*
 * wubu_diagnosis.h -- the CLOSED CONTROL SYSTEM (the user directive,
 * 2026-08-09: "make the colony actually live as a self-modifying AGI,
 * not just an inference engine with cool modules").
 *
 * The standing loop becomes the PRIMARY runtime path:
 *
 *   every batch
 *     -> structured Diagnosis record (loss surface, expert utilization,
 *        gradient health, cell fitness vector)
 *     -> the hive is the ONLY fitness recorder (fitness cells live in
 *        the same tissue as the model cells)
 *     -> the amoeba mutates (grow/shrink) gated by the SAME fitness +
 *        Lean prover path
 *     -> accepted mutations archive (the DGM branch tree); rejected
 *        ones go to the GRAVEYARD (still queryable: negative examples)
 *     -> repeat
 *
 * Pure C11, opaque, no third party. The organs stay self-contained;
 * this module is the conductor that closes the loop.
 */
#ifndef WUBU_DIAGNOSIS_H
#define WUBU_DIAGNOSIS_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_hive.h"
#include "wubu_amoeba.h"
#include "wubu_moe2.h"

/* the structured batch Diagnosis (every batch produces one) */
typedef struct {
    /* loss surface */
    float loss;            /* this batch's loss */
    float loss_ema;        /* the exponential moving average */
    float slope;           /* the recent ema slope (per batch) */
    int   plateau;         /* 1 = plateau detector fired */
    /* expert utilization (the colony's work distribution) */
    int   n_experts;       /* live expert count */
    float util_min, util_max, util_mean;  /* route utilization stats */
    /* gradient health */
    float grad_norm_max;   /* the largest per-cell grad norm */
    float grad_norm_mean;  /* the mean per-cell grad norm */
    int   n_dead_cells;    /* cells below the shrink threshold */
    /* the cell fitness vector (the hive is the ONLY recorder) */
    float fitness;         /* the held-out fitness (loss-based) */
    float prev_fitness;    /* the pre-mutation fitness */
    /* provenance */
    uint64_t batch;        /* the global batch counter */
    uint32_t epoch;
} wubu_diag_record_t;

/* the mutation outcome (what the gate decided) */
typedef enum {
    WUBU_DIAG_ACCEPT = 1,   /* archived: passed fitness + prover */
    WUBU_DIAG_REJECT = 0,   /* graveyard: failed the gate */
    WUBU_DIAG_STASIS = 2    /* no mutation attempted (healthy band) */
} wubu_diag_verdict_t;

/* a hive cell: one fitness observation with provenance. The hive is
 * the ONLY place fitness is recorded (no side-channel stats). */
typedef struct {
    uint64_t batch;         /* when */
    uint32_t epoch;
    float    fitness;       /* the held-out fitness */
    float    delta;         /* change vs the previous fitness */
    wubu_diag_verdict_t verdict;  /* what the gate decided */
    uint8_t  cell_idx;      /* which cell mutated (0xFF = whole colony) */
    /* the graveyard flag: rejected mutations stay queryable here */
    uint8_t  graveyard;
} wubu_fitness_cell_t;

/* the closed loop state */
typedef struct {
    wubu_hive_t   *tissue;      /* the hive (cells + fitness) */
    wubu_amoeba_t *amoeba;      /* the immune system */
    wubu_moe2_t   *agents;      /* the colony */
    /* the fitness ledger */
    wubu_fitness_cell_t *ledger;  /* the archive ring */
    int ledger_n, ledger_cap;
    /* the graveyard ring (rejected mutations — still queryable) */
    wubu_fitness_cell_t *graveyard;
    int grave_n, grave_cap;
    /* the loop counters */
    uint64_t batch;
    uint32_t epoch;
    int      n_accepted, n_rejected, n_stasis;
    /* the diagnosis scratch (reused per batch) */
    float *cell_grads;          /* [n_experts] */
    int    n_cells_alloc;
    /* Phase 1 (the default path): the lineage tracker + the runtime
     * contracts are part of the gate — a mutation survives only if it
     * passes fitness + prover + contracts + lineage pressure. */
    struct wubu_lineage_tracker_t *lineage;   /* optional (owned by caller) */
    struct wubu_contracts_t       *contracts; /* optional (owned by caller) */
    /* Phase 2: the priority store (BI + Fisher/EWC + precision deltas
     * + the mutation ledger) — the diagnose consults it before the
     * next mutation; a protected cell (the loss cares + recent
     * rejection) is REFUSED. */
    struct wubu_priority_store_t  *prio;      /* optional (owned by caller) */
} wubu_diag_loop_t;

/* L1: init the closed loop. The caller owns the organs. */
int wubu_diag_loop_init(wubu_diag_loop_t *loop, wubu_hive_t *tissue,
                        wubu_amoeba_t *amoeba, wubu_moe2_t *agents,
                        int ledger_cap, int graveyard_cap);

/* L2: record one batch's diagnosis. The hive is the ONLY fitness
 * recorder: this writes a fitness cell + feeds the amoeba. */
int wubu_diag_record(wubu_diag_loop_t *loop, const wubu_diag_record_t *rec);

/* L3: run one closed-loop iteration (the mutation cycle):
 *   record -> amoeba diagnose -> mutate -> validate (loss tol + prover)
 *   -> archive (accept) / graveyard (reject).
 * Returns the verdict. The caller provides the held-out fitness AFTER
 * the mutation (the gate compares it to the pre-mutation fitness). */
wubu_diag_verdict_t wubu_diag_cycle(wubu_diag_loop_t *loop,
                                    const wubu_diag_record_t *rec,
                                    float held_out_loss);

/* L4: the graveyard query (negative examples are useful): walk the
 * rejected mutations. Returns the count visited. */
size_t wubu_diag_graveyard_foreach(const wubu_diag_loop_t *loop,
                                   int (*fn)(const wubu_fitness_cell_t *cell,
                                             void *user), void *user);

/* L5: the loop stats. */
void wubu_diag_stats(const wubu_diag_loop_t *loop, char *buf, size_t cap);

/* L6: free the loop's own allocations (NOT the organs). */
void wubu_diag_loop_free(wubu_diag_loop_t *loop);

/* L7: collect the per-cell gradient health from the REAL trainer's
 * accumulators (wubu_train_t). The colony cells map to the layer
 * groups: cell i gets the mean grad norm of layer i's matrices
 * (q/k/v/o/g projections + gate_up/down). Fills loop->cell_grads
 * and returns the number of cells measured. This is what closes the
 * loop with REAL training signal (not a toy task). */
int wubu_diag_collect_grads(wubu_diag_loop_t *loop,
                            const void *train, int n_layers, int dim,
                            int ffn_dim, int kv_width, int head_width);

/* L8: compute the loss surface (ema slope + plateau) from a loss
 * history window — the trainer feeds the ema array; this fills the
 * record's slope/plateau fields. The same adaptive threshold as the
 * CLI's plateau detector (abs floor OR 0.5% of the loss). */
void wubu_diag_loss_surface(wubu_diag_record_t *rec,
                            const float *loss_hist, int hist_n);

/* L9: save the loop's ledger + graveyard to an archive file (the
 * hive walk reads it; the Brain's memory becomes visible + versionable
 * to the Body). */
int wubu_diag_save(const wubu_diag_loop_t *loop, const char *path);

/* L10: SELF-CRITIQUE + RECOVERY (the user directive #6: failed
 * generations auto-trigger diagnose->mutate on the responsible
 * lineage). Called when an execution fails (the fitness gate rejects,
 * the Body reports a failed action, the oracle rates a lineage low).
 *   - records the failure as a preference pair (the losing lineage)
 *   - marks the responsible cell for shrink (apoptosis pressure)
 *   - triggers an immediate mutation cycle (not waiting for the next
 *     diag_every)
 *   - returns the verdict (accept/stasis/reject)
 */
wubu_diag_verdict_t wubu_diag_recover(wubu_diag_loop_t *loop,
                                      uint8_t cell_idx,
                                      float failed_fitness,
                                      float survived_fitness);

#endif
