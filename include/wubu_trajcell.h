/*
 * wubu_trajcell.h -- VERIFIED TOOL-TRAJECTORY CELLS (roadmap #2).
 * C11. Tool use and multi-step work join the hive's fitness model.
 *
 * Every successful (and failed) tool trajectory becomes a TYPED HIVE
 * CELL: goal, steps, outcome, cost (tokens/time), and a compact hash
 * of the trajectory. Diagnose can query "trajectories that solved
 * similar goals" and bias specialist spawning toward those patterns.
 * Failed trajectories with high partial credit become mutation seeds
 * (prompt/scaffold/expert-routing changes) instead of being discarded.
 *
 * Pure C11, opaque, no third party. The cells are hive inserts (the
 * colony IS the memory).
 */
#ifndef WUBU_TRAJCELL_H
#define WUBU_TRAJCELL_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_hive.h"

/* one trajectory cell (a hive insert) */
typedef struct {
    uint64_t traj_id;         /* the trajectory id */
    uint16_t goal_token;      /* the goal it served */
    uint8_t  n_steps;         /* how many tool steps */
    uint8_t  ok;              /* 1 = success, 0 = failure */
    float    outcome;         /* the fitness outcome (loss/task score) */
    float    partial_credit;  /* 0..1 — useful even though it failed */
    float    cost;            /* tokens+time cost (normalized) */
    uint32_t hash;            /* the compact trajectory hash (FNV) */
    uint64_t batch;           /* provenance */
} wubu_trajcell_t;

/* the trajectory registry state */
typedef struct {
    wubu_hive_t *tissue;      /* the cells live here */
    uint64_t next_traj_id;
    /* telemetry */
    uint64_t n_success, n_failed;
} wubu_traj_tracker_t;

/* T1: init. The caller owns the hive. */
int wubu_traj_init(wubu_traj_tracker_t *tt, wubu_hive_t *tissue);

/* T2: record a trajectory as a typed hive cell. hash is the compact
 * trajectory hash (or 0 to compute from the goal+steps). Returns the
 * traj id (>0) or -1. */
int64_t wubu_traj_record(wubu_traj_tracker_t *tt, uint16_t goal,
                         const uint32_t *step_hashes, int n_steps,
                         int ok, float outcome, float partial_credit,
                         float cost, uint64_t batch);

/* T3: the FNV trajectory hash (compact, stable) — the cell's
 * identity for dedup + similarity queries. */
uint32_t wubu_traj_hash(const uint32_t *step_hashes, int n_steps,
                        uint16_t goal);

/* T4: query "trajectories that solved similar goals": walk the hive,
 * return the traj ids whose goal is within tol AND ok=1, sorted by
 * outcome (best first). Returns the count. */
int wubu_traj_similar(const wubu_traj_tracker_t *tt, uint16_t goal,
                      float tol, int64_t *out, int out_cap);

/* T5: the mutation-seed query: FAILED trajectories with high partial
 * credit (>= credit_th) — these are the seeds for prompt/scaffold/
 * routing mutations, not garbage. Returns the count. */
int wubu_traj_seeds(const wubu_traj_tracker_t *tt, float credit_th,
                    int64_t *out, int out_cap);

/* T6: the tracker stats. */
void wubu_traj_stats(const wubu_traj_tracker_t *tt, char *buf, size_t cap);

#endif
