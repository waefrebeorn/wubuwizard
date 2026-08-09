/*
 * wubu_harness.h -- THE SUSTAINED AUTONOMY TASK HARNESS (the
 * post-AN39 wave, Phase 3: Pillars 11 + 20). C11.
 *
 * A fixed multi-hour computer-task suite the colony cannot "pass" by
 * short-batch loss alone:
 *   - tasks of 3 kinds: code-edit, file-transform (via userfs/codec),
 *     tool-use goals — each with a deterministic scorer
 *   - scored outcomes are written as TRAJ CELLS (the hive)
 *   - the slow metadiag treats the suite score as a FIRST-CLASS
 *     fitness signal (alongside loss)
 *   - the loopguard (deadline + step ceiling) makes the harness safe
 *     to run unattended: it stops cleanly
 *
 * Pure C11, opaque, deterministic, no external world (Phase 5 brings
 * the external tools behind the Colonel cap surface).
 */
#ifndef WUBU_HARNESS_H
#define WUBU_HARNESS_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_hive.h"
#include "wubu_loopguard.h"

/* the task kinds */
typedef enum {
    WUBU_TASK_EDIT = 0,      /* code edit: transform A -> B */
    WUBU_TASK_TRANSFORM = 1, /* file transform via the codec/userfs */
    WUBU_TASK_TOOLUSE = 2    /* tool-use goal: compose steps */
} wubu_task_kind_t;

/* one harness task (the fixed suite) */
typedef struct {
    wubu_task_kind_t kind;
    uint16_t goal_token;       /* the goal this task serves */
    /* the deterministic scorer inputs */
    int    steps_done;         /* the agent's completed steps */
    int    steps_required;     /* the minimal steps for a pass */
    float  correctness;        /* 0..1 the output's correctness */
    float  cost;               /* normalized cost (tokens+time) */
    int    deadline_ok;        /* finished before the deadline */
} wubu_task_t;

/* the harness state */
typedef struct {
    wubu_hive_t *tissue;       /* the traj cells land here */
    wubu_loopguard_t guard;    /* the deadline + step ceiling */
    wubu_task_t suite[16];     /* the fixed task suite */
    int  n_tasks;
    long step;                 /* the harness step counter */
    long now_ns;               /* the virtual clock */
    /* the running suite score (the first-class fitness signal) */
    float suite_score;         /* 0..1 weighted mean over the tasks */
    uint64_t n_passed, n_failed;
} wubu_harness_t;

/* H1: init the harness with the fixed task suite. */
int wubu_harness_init(wubu_harness_t *h, wubu_hive_t *tissue,
                      long deadline_ns, long max_steps);

/* H2: run one task (the deterministic scorer). The agent's attempt is
 * scored and written as a TRAJ CELL (goal, outcome, cost). Returns 1
 * when the task passed. */
int wubu_harness_run_task(wubu_harness_t *h, int task_idx,
                          int steps_done, float correctness, float cost);

/* H3: the loopguard check — 0 when the deadline or the step ceiling
 * says STOP (the harness stops cleanly, unattended). */
int wubu_harness_may_continue(wubu_harness_t *h);

/* H4: recompute the suite score (the first-class fitness signal the
 * slow metadiag reads). Returns 0..1. */
float wubu_harness_suite_score(wubu_harness_t *h);

/* H5: the harness stats. */
void wubu_harness_stats(const wubu_harness_t *h, char *buf, size_t cap);

#endif
