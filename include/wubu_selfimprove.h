/*
 * wubu_selfimprove.h -- the TRACE/SPAN OPERATOR + the RSI-POWERED
 * MUTATION ENGINE (closes P20: "Trace/span operator hook (DA-3) —
 * wubu_selfimprove does NOT exist"). C11.
 *
 * Two halves:
 *
 *   A. TRACE/SPAN (P20): a bounded trace of the agent's work — each
 *      span is a unit of work (a batch, a tool call, a specialist
 *      cell's run) with its outcome. The trace writes INTO THE HIVE
 *      (the sole long-term memory substrate), not a separate log.
 *      The hive walk queries it by span kind / outcome.
 *
 *   B. THE MUTATION ENGINE (directive #4: "wire RSI primitives into
 *      the live diagnose/mutate path — they should be the mutation
 *      engine the amoeba actually calls"): one step = the RSI gate
 *      (bounded verifiable improvement) -> LADDER decompose -> prompt
 *      mutation / reflection -> bounded self-modification delta ->
 *      the amoeba's mutation. The experience loop (IV09) is the
 *      continuous diagnose input, not an offline process.
 *
 * Pure C11, opaque, no third party.
 */
#ifndef WUBU_SELFIMPROVE_H
#define WUBU_SELFIMPROVE_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_hive.h"

/* the trace span kinds (what the agent was doing) */
typedef enum {
    WUBU_SPAN_BATCH = 0,     /* a training batch */
    WUBU_SPAN_TOOL = 1,      /* a tool trajectory */
    WUBU_SPAN_CELL = 2,      /* a specialist cell's run */
    WUBU_SPAN_MUTATION = 3,  /* a mutation proposal */
    WUBU_SPAN_EVAL = 4       /* an evaluation harness run */
} wubu_span_kind_t;

/* one trace span (the P20 hook — written into the hive) */
typedef struct {
    wubu_span_kind_t kind;
    uint64_t trace_id;       /* the owning trace */
    uint64_t batch;
    float    value;          /* the span's fitness contribution */
    int      ok;             /* 1 = success, 0 = failure */
    uint16_t goal_token;     /* the goal this span served */
    uint8_t  cell_idx;       /* the responsible cell */
} wubu_span_t;

/* the mutation proposal (what the engine wants to change) */
typedef enum {
    WUBU_MUT_ARCH = 0,       /* architecture (layer count, routing) */
    WUBU_MUT_PROMPT = 1,     /* prompt / scaffold evolution */
    WUBU_MUT_ROUTE = 2,      /* expert routing weights */
    WUBU_MUT_WEIGHT = 3,     /* weight-space delta (bounded) */
    WUBU_MUT_NEST = 4        /* nesting curvature */
} wubu_mut_kind_t;

typedef struct {
    wubu_mut_kind_t kind;
    float    strength;       /* how big the change (bounded) */
    float    fitness_gate;   /* the verifier score that must pass */
    int      n_subgoals;     /* the LADDER decomposition count */
} wubu_mutation_t;

/* the self-improve engine state */
typedef struct {
    wubu_hive_t *tissue;     /* the hive (traces + spans land here) */
    /* the experience loop (IV09): streaming telemetry -> improvement */
    long evals, wins;
    float running;
    int  consecutive_fails;  /* the RSI gate's failure counter */
    /* counters */
    uint64_t spans_written;
    uint64_t mutations_proposed;
    uint64_t mutations_accepted;
    float    last_delta;
} wubu_selfimprove_t;

/* S1: init. The caller owns the hive. */
int wubu_selfimprove_init(wubu_selfimprove_t *si, wubu_hive_t *tissue);

/* S2: write a trace span INTO THE HIVE (the P20 hook — the trace is
 * hive-native, queryable by the walk). Returns 0 on success. */
int wubu_selfimprove_trace(wubu_selfimprove_t *si, wubu_span_kind_t kind,
                           uint64_t trace_id, uint64_t batch, float value,
                           int ok, uint16_t goal, uint8_t cell_idx);

/* S3: the RSI experience loop (IV09): stream one evaluation outcome
 * into the running stats — the continuous diagnose input. */
int wubu_selfimprove_experience(wubu_selfimprove_t *si, int win, float value);

/* S4: propose a mutation under the RSI gate (bounded verifiable
 * improvement): gate -> decompose -> mutate kind + strength. The
 * strength is bounded by the RSI bounded-delta (IV15). Returns 1 if
 * the proposal passes the gate (caller runs it through the amoeba). */
int wubu_selfimprove_propose(wubu_selfimprove_t *si, float verifier_score,
                             float difficulty, float budget,
                             wubu_mutation_t *out);

/* S5: the full RSI-powered mutation step (directive #4): the gate ->
 * decompose -> propose -> trace -> the amoeba's mutate. Returns the
 * proposal verdict (1 = accepted by the engine's gate). */
int wubu_selfimprove_step(wubu_selfimprove_t *si, float verifier_score,
                          float difficulty, float budget, float fitness_gate,
                          wubu_mutation_t *out);

/* S6: report a mutation's outcome (feeds the experience loop + the
 * consecutive-fail counter the gate uses). */
void wubu_selfimprove_report(wubu_selfimprove_t *si, int accepted);

/* S7: the engine stats. */
void wubu_selfimprove_stats(const wubu_selfimprove_t *si, char *buf, size_t cap);

#endif
