/*
 * wubu_priority_store.h -- THE PRIORITY STORE (the post-AN39 wave,
 * Phase 2 completion). C11.
 *
 * The bridge between "loss went down" and "the colony should keep
 * doing that": a per-cell PRIORITY ledger the diagnose consults
 * before the next mutation (the structure / compute / precision
 * axes). A mutation that IGNORES the priority-store evidence is
 * refused (the same gate discipline as the contracts + blueprint).
 *
 * The store is the safetensors SIDECAR to a checkpoint:
 *   - BI (block importance): per-cell importance (the Fisher/EWC
 *     diagonal — how much the loss cares about that cell)
 *   - the FISHER estimate: the mean squared gradient (the EWC
 *     regularizer's diagonal — the online update)
 *   - PRECISION DELTAS: the per-family bit-ladder deltas the
 *     precision plan produced (what the profile said to change)
 *   - the MUTATION LEDGER: the accepted/rejected mutation history
 *     (the same provenance the hive keeps, mirrored here for the
 *     sidecar)
 *
 * The diagnose path: before mutating a cell, check the store — a
 * cell with HIGH importance (Fisher) and a recent REJECTED mutation
 * is protected (refuse mutations that would trash it); a cell with
 * LOW importance is a mutation candidate.
 *
 * Pure C11, opaque. The sidecar is a tiny binary format (not full
 * safetensors — the reader is already in the repo for checkpoints).
 */
#ifndef WUBU_PRIORITY_STORE_H
#define WUBU_PRIORITY_STORE_H

#include <stdint.h>
#include <stddef.h>

/* the store's per-cell entry */
typedef struct {
    uint8_t  cell_idx;       /* the colony cell */
    uint8_t  family;         /* the matrix family (wubu_family_t) */
    float    importance;     /* the BI: 0..1 block importance */
    float    fisher;         /* the Fisher/EWC diagonal (online) */
    float    precision_delta;/* the ladder delta this cell got */
    uint64_t n_mutations;
    uint64_t n_rejected;     /* how many mutations the gate rejected */
    uint8_t  protected;      /* 1 = high-importance + recent reject */
} wubu_prio_cell_t;

/* the priority store state */
typedef struct wubu_priority_store {
    wubu_prio_cell_t cells[32];  /* the per-cell ledger (bounded) */
    int  n;
    /* the gate thresholds */
    float protect_fisher;    /* fisher above this + a recent reject = protect */
    float mutate_importance; /* importance below this = a mutation candidate */
    /* telemetry */
    uint64_t n_consulted, n_refused, n_allowed;
} wubu_priority_store_t;

/* PS1: init. */
int wubu_prio_init(wubu_priority_store_t *ps, float protect_fisher,
                   float mutate_importance);

/* PS2: register a cell (the trainer/profiler feeds it). */
int wubu_prio_register(wubu_priority_store_t *ps, uint8_t cell_idx,
                       uint8_t family, float importance, uint64_t batch);

/* PS3: the online Fisher update (EWC): fisher = (1-beta)*fisher +
 * beta*grad^2 — the mean squared gradient accumulates the diagonal. */
void wubu_prio_update_fisher(wubu_priority_store_t *ps, uint8_t cell_idx,
                             float grad_norm, float beta);

/* PS4: record a mutation outcome in the ledger (accepted/rejected —
 * the same provenance as the hive, mirrored for the sidecar). */
void wubu_prio_record_mutation(wubu_priority_store_t *ps, uint8_t cell_idx,
                               int accepted);

/* PS5: the DIAGNOSE GATE — consult the store before mutating a cell:
 * returns 1 = the mutation is allowed, 0 = REFUSED (the cell is
 * protected: high Fisher + a recent rejection, so a new mutation
 * would trash what the loss already cares about). Consulting counts
 * (the telemetry is part of the gate). */
int wubu_prio_gate(wubu_priority_store_t *ps, uint8_t cell_idx);

/* PS6: the per-family precision delta (what the ladder changed —
 * the store's contribution to the precision morph). */
void wubu_prio_set_delta(wubu_priority_store_t *ps, uint8_t cell_idx,
                         float delta);

/* PS7: the sidecar save (the tiny binary format — a checkpoint's
 * priority ledger). Returns bytes written. */
long wubu_prio_save(const wubu_priority_store_t *ps, void *buf, long cap);

/* PS8: the sidecar load. Returns the cell count restored. */
int wubu_prio_load(wubu_priority_store_t *ps, const void *buf, long n);

/* PS9: the store stats. */
void wubu_prio_stats(const wubu_priority_store_t *ps, char *buf, size_t cap);

#endif
