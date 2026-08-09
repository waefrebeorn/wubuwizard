/*
 * wubu_agi.h -- the AGI loop: WuBu's recursive learning machine.
 *
 * The AGI_HOME_METAGAME + AGI_OS_GOALPOSTS docs define the loop:
 *   act -> observe -> learn -> verify -> grow -> repeat
 * This module wires the WuBu model's organs into that loop:
 *
 *   - the MODEL (wubu): the seed, the trained base
 *   - the HIVE (wubu_hive): the working memory -- tokens, observations
 *   - the AGENTS (wubu_moe2): the mixed agents, few active per token
 *   - the GEOMETRY (wubu_hyper + wubu_nest): the Lean-verified math
 *   - the MIXER (wubu_deltanet): linear attention, the KV saving
 *   - the VERIFIER (wubu_prover): the Lean-style reward signal
 *   - the RECOVERY (WuBuOS wubu_recovery): mistakes are safe
 *
 * The loop (one iteration):
 *   1. observe: the environment feeds an observation into the hive
 *   2. think: the model proposes an action (route through the agents)
 *   3. verify: the prover checks the proposal (the reward signal)
 *   4. act: the accepted action is applied
 *   5. learn: the outcome is a training datum (the trainer consumes it)
 *   6. grow: the checkpoint is saved (the 5+1 rollback slots)
 *
 * Pure C11. The organs are self-contained modules; this is the
 * conductor, not the orchestra.
 */
#ifndef WUBU_AGI_H
#define WUBU_AGI_H

#include <stdint.h>
#include <stddef.h>

#include "wubu.h"
#include "wubu_hive.h"
#include "wubu_moe2.h"
#include "wubu_prover.h"

/* the AGI's state: one conductor owning the organs */
typedef struct {
    wubu_model_t *model;        /* the WuBu seed */
    wubu_buf_t   *buf;          /* the model's working buffers */
    wubu_hive_t    memory;       /* the working memory (observations) */
    wubu_moe2_t   *agents;       /* the mixed agents */
    wubu_proof_t   verifier;     /* the prover state (EE04) */
    /* telemetry */
    uint64_t       iterations;
    uint64_t       accepted_steps;   /* verified proof steps */
    uint64_t       rejected_steps;
    uint64_t       memory_writes;
    uint64_t       memory_reuses;
} wubu_agi_t;

/* A1: init the AGI (owns nothing -- the caller provides the organs). */
int wubu_agi_init(wubu_agi_t *agi, wubu_model_t *m, wubu_buf_t *b,
                  wubu_moe2_t *agents);

/* A2: observe -- push an observation (token id) into the hive. */
int wubu_agi_observe(wubu_agi_t *agi, uint16_t token);

/* A3: think -- run the model forward over the current context, route
 * through the agents, return the top-k next-token candidates. */
int wubu_agi_think(wubu_agi_t *agi, uint16_t *out_tokens, int max_out);

/* A4: verify -- check a proposed proof step; the accepted/rejected
 * counters drive the RL reward. */
int wubu_agi_verify(wubu_agi_t *agi, const wubu_pf_step_t *step);

/* A5: one full loop iteration (observe -> think -> verify -> act). */
int wubu_agi_step(wubu_agi_t *agi, uint16_t observation,
                  uint16_t *action_out);

/* A6: telemetry. */
void wubu_agi_stats(const wubu_agi_t *agi, char *buf, size_t cap);

/* ── the specialist-cell orchestrator (the user directive #3: the
 * biggest missing pillar — hierarchical planning / multi-agent
 * orchestration, pure C11, no Python) ────────────────────────── */

/* the specialist lenses (what each cell is good at) */
typedef enum {
    WUBU_CELL_CODE = 0,     /* the code lens */
    WUBU_CELL_MATH = 1,     /* the math lens */
    WUBU_CELL_TOOL = 2,     /* the tool-use lens */
    WUBU_CELL_CRIT = 3,     /* the critique lens (the adversary) */
    WUBU_CELL_JUDGE = 4     /* the judge (merges the specialists) */
} wubu_cell_lens_t;

/* one specialist cell: a short-lived hive insert with a lens */
typedef struct {
    wubu_cell_lens_t lens;
    uint16_t goal_token;       /* the sub-goal this cell attacks */
    float    confidence;       /* the cell's self-reported confidence */
    int      accepted;         /* the judge's verdict on this cell */
    uint64_t batch;            /* provenance */
} wubu_specialist_t;

/* the orchestrator state (a sub-session of the AGI loop) */
typedef struct {
    wubu_hive_t *tissue;         /* the shared hive (cells insert here) */
    uint16_t     goal;           /* the goal token */
    wubu_specialist_t cells[4];  /* code/math/tool/critique */
    int          n_cells;        /* spawned this session */
    uint16_t     decision;       /* the judge's merged decision token */
    float        judge_confidence;
    uint64_t     batch;
} wubu_orch_t;

/* O1: spawn the specialist cells for a goal. Each cell is a temporary
 * hive insert (a short-lived lens worker). Returns the count spawned. */
int wubu_orch_spawn(wubu_orch_t *orch, wubu_hive_t *tissue,
                    uint16_t goal, uint64_t batch);

/* O2: collect each specialist's verdict (the cells report back —
 * in the real loop these are the lens-forward passes; the stub wires
 * the mechanism). Returns the merged decision token (the judge cell). */
int wubu_orch_judge(wubu_orch_t *orch, const float *cell_confidences);

/* O3: the hive insert for a specialist (the cell IS a hive cell —
 * the colony is the memory). Returns 0 on success. */
int wubu_orch_insert_cell(wubu_orch_t *orch, wubu_cell_lens_t lens,
                          float confidence);

#endif
