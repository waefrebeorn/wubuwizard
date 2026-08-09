/*
 * wubu_metadiag.h -- DUAL-TIMESCALE DIAGNOSE (roadmap #3). C11.
 *
 * One diagnose pass per batch is not enough for long-horizon AGI
 * behavior:
 *   - FAST path (every batch/trajectory): local cell fitness, expert
 *     utilization, repetition/coherence flags. This is the amoeba's
 *     per-batch input.
 *   - SLOW path (on a schedule / after N fast cycles): global hive
 *     statistics, lineage health, task-harness trends, distribution
 *     shift between recent and older cells, and a compact "colony
 *     state" summary written as a META-CELL. Slow results can raise
 *     or lower the mutation rate + the fitness floor for the next
 *     window.
 *
 * Both paths are pure C11 + hive-native; no external dashboard is
 * required for the loop to run.
 */
#ifndef WUBU_METADIAG_H
#define WUBU_METADIAG_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_hive.h"

/* the fast-path per-batch signals */
typedef struct {
    float loss;             /* this batch's loss */
    float util_mean;        /* expert utilization mean */
    float util_spread;      /* utilization spread (max - min) */
    float repeat;           /* repetition/coherence flag (0..1) */
    float grad_norm;        /* the gradient health */
    float task_score;       /* Phase 3: the harness suite score
                               (the first-class fitness signal — the
                               colony cannot pass by loss alone) */
} wubu_fast_signal_t;

/* the slow-path colony-state meta-cell (the summary written into the
 * hive — the colony can read its own global state) */
typedef struct {
    uint64_t batch;
    size_t   hive_live;        /* cells alive */
    size_t   hive_capacity;    /* the tissue size */
    float    lineage_health;   /* surviving-lineage fraction */
    float    shift;            /* distribution shift recent vs old */
    float    trend;            /* the long-horizon task trend */
    float    mutation_rate;    /* the next window's rate */
    float    fitness_floor;    /* the next window's floor */
} wubu_meta_cell_t;

/* the dual-timescale state */
typedef struct {
    wubu_hive_t *tissue;
    int slow_every;            /* run the slow path every N fast */
    uint64_t fast_cycles;
    uint64_t slow_cycles;
    /* the rolling window (for shift/trend) */
    float  *window;            /* the recent losses */
    int     win_n, win_cap;
    float   mutation_rate;     /* the current rate (slow adjusts it) */
    float   fitness_floor;
    float   lr_scale;          /* how much the slow path moves things */
    float   task_ema;          /* Phase 3: the running suite-score EMA */
    uint64_t n_policy_changes; /* AN47 #8: policy-change meta-cells */
    int      stasis_patience;  /* DA fix: consecutive flat-trend slow
                                  passes before the policy HOLDS (the
                                  early-stopping patience window —
                                  a single flat snapshot is noise) */
    int      stasis_window;    /* the patience threshold (default 3) */
} wubu_metadiag_t;

/* M1: init. */
int wubu_metadiag_init(wubu_metadiag_t *md, wubu_hive_t *tissue,
                       int slow_every, int window_cap, float lr_scale);

/* M2: the FAST path — every batch. Returns 1 when a slow pass is due. */
int wubu_metadiag_fast(wubu_metadiag_t *md, const wubu_fast_signal_t *s);

/* M3: the SLOW path — the colony-state meta-cell + the rate/floor
 * adjustment for the next window. Returns 0 on success. */
int wubu_metadiag_slow(wubu_metadiag_t *md);

/* M4: read the current window's colony state (the rate/floor the
 * next window runs under). */
void wubu_metadiag_state(const wubu_metadiag_t *md, float *rate, float *floor);

/* M5: the meta-cell count in the hive (the colony's self-knowledge). */
size_t wubu_metadiag_metacells(const wubu_metadiag_t *md);

/* M6: the stats. */
void wubu_metadiag_stats(const wubu_metadiag_t *md, char *buf, size_t cap);

#endif
