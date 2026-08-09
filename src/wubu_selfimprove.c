/*
 * wubu_selfimprove.c -- the TRACE/SPAN OPERATOR + the RSI mutation
 * engine (see wubu_selfimprove.h). Closes P20; wires the RSI
 * primitives into the live diagnose/mutate path (directive #4).
 */
#include "wubu_selfimprove.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_rsi.h"

int wubu_selfimprove_init(wubu_selfimprove_t *si, wubu_hive_t *tissue)
{
    if (!si || !tissue) return -1;
    memset(si, 0, sizeof(*si));
    si->tissue = tissue;
    si->running = 0.5f;
    return 0;
}

int wubu_selfimprove_trace(wubu_selfimprove_t *si, wubu_span_kind_t kind,
                           uint64_t trace_id, uint64_t batch, float value,
                           int ok, uint16_t goal, uint8_t cell_idx)
{
    if (!si || !si->tissue) return -1;
    /* the span IS a hive cell (the trace is hive-native — the sole
     * long-term memory substrate, queryable by the walk) */
    wubu_span_t *span = (wubu_span_t *)calloc(1, sizeof(wubu_span_t));
    if (!span) return -1;
    span->kind = kind;
    span->trace_id = trace_id;
    span->batch = batch;
    span->value = value;
    span->ok = ok;
    span->goal_token = goal;
    span->cell_idx = cell_idx;
    wubu_hive_insert(si->tissue, span);
    si->spans_written++;
    return 0;
}

int wubu_selfimprove_experience(wubu_selfimprove_t *si, int win, float value)
{
    if (!si) return -1;
    wubu_rsi_exp_t e = { si->evals, si->wins, si->running };
    int r = wubu_rsi_experience(&e, win, value);
    si->evals = e.evals; si->wins = e.wins; si->running = e.running;
    return r;
}

int wubu_selfimprove_propose(wubu_selfimprove_t *si, float verifier_score,
                             float difficulty, float budget,
                             wubu_mutation_t *out)
{
    if (!si || !out) return -1;
    /* the RSI gate (IV01): improve only if the verifier passes and we
     * are not in a consecutive-fail streak */
    if (!wubu_rsi_gate(verifier_score, 0.5f, &si->consecutive_fails))
        return 0;
    /* the LADDER decomposition (IV03): a hard goal splits into
     * sub-goals the colony can actually run */
    int n_sub = 0;
    wubu_rsi_decompose(difficulty, budget, 2, &n_sub);
    /* the mutation kind from the decompose depth: hard goals mutate
     * architecture, easy ones prompt/route */
    memset(out, 0, sizeof(*out));
    out->kind = (difficulty > 0.7f) ? WUBU_MUT_ARCH
               : (n_sub > 1 ? WUBU_MUT_ROUTE : WUBU_MUT_PROMPT);
    /* the bounded delta (IV15): the safe-pace strength */
    out->strength = wubu_rsi_bounded_delta(1.0f, 0.05f,
                                           budget > 0 ? budget : 1.0f);
    out->fitness_gate = verifier_score;
    out->n_subgoals = n_sub;
    si->mutations_proposed++;
    return 1;
}

int wubu_selfimprove_step(wubu_selfimprove_t *si, float verifier_score,
                          float difficulty, float budget, float fitness_gate,
                          wubu_mutation_t *out)
{
    if (!si) return -1;
    /* trace the propose (P20: the mutation proposal is a span) */
    wubu_selfimprove_trace(si, WUBU_SPAN_MUTATION, 0, (uint64_t)si->evals,
                           verifier_score, verifier_score >= fitness_gate,
                           0, 0xFF);
    if (!wubu_selfimprove_propose(si, verifier_score, difficulty, budget, out))
        return 0;
    /* the experience loop input: the proposal's verifier score counts
     * as a win when it clears the fitness gate */
    wubu_selfimprove_experience(si, verifier_score >= fitness_gate,
                                verifier_score);
    si->last_delta = out->strength;
    return 1;
}

void wubu_selfimprove_report(wubu_selfimprove_t *si, int accepted)
{
    if (!si) return;
    if (accepted) {
        si->mutations_accepted++;
        si->consecutive_fails = 0;
    } else {
        si->consecutive_fails++;
    }
}

void wubu_selfimprove_stats(const wubu_selfimprove_t *si, char *buf, size_t cap)
{
    if (!si || !buf || cap == 0) return;
    snprintf(buf, cap,
             "spans=%llu proposed=%llu accepted=%llu evals=%ld wins=%ld "
             "running=%.3f fails=%d last_delta=%.4f",
             (unsigned long long)si->spans_written,
             (unsigned long long)si->mutations_proposed,
             (unsigned long long)si->mutations_accepted,
             si->evals, si->wins, si->running,
             si->consecutive_fails, si->last_delta);
}
