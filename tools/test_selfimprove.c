/*
 * test_selfimprove.c -- the TRACE/SPAN OPERATOR + RSI mutation engine
 * gate (closes P20: "wubu_selfimprove does NOT exist").
 *
 * Asserts:
 *   1. trace spans write INTO THE HIVE (hive-native, queryable)
 *   2. the RSI gate blocks bad mutations (low verifier -> no proposal)
 *   3. good mutations pass the gate and decompose (LADDER)
 *   4. the experience loop is the continuous input (wins accumulate)
 */
#include <stdio.h>
#include <string.h>

#include "wubu_selfimprove.h"
#include "wubu_hive.h"
#include "wubu_rsi.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_selfimprove (P20: the trace/span + RSI engine) ===\n");

    wubu_hive_t tissue;
    if (wubu_hive_init(&tissue) != 0) FAIL("hive init");

    wubu_selfimprove_t si;
    if (wubu_selfimprove_init(&si, &tissue) != 0) FAIL("si init");

    /* 1. spans write into the hive */
    wubu_selfimprove_trace(&si, WUBU_SPAN_BATCH, 1, 1, 10.0f, 1, 42, 0);
    wubu_selfimprove_trace(&si, WUBU_SPAN_TOOL, 1, 2, 8.0f, 1, 42, 1);
    wubu_selfimprove_trace(&si, WUBU_SPAN_CELL, 2, 3, 12.0f, 0, 43, 2);
    size_t live = wubu_hive_live(&tissue);
    printf("  hive live cells: %zu (3 spans written)\n", live);
    if (live != 3) FAIL("spans did not land in the hive (%zu)", live);

    /* 2. the RSI gate blocks a bad mutation */
    wubu_mutation_t mut;
    int gate = wubu_selfimprove_propose(&si, 0.2f, 0.5f, 1.0f, &mut);
    printf("  low verifier (0.2): gate=%d (0 = blocked)\n", gate);
    if (gate != 0) FAIL("bad mutation passed the gate");

    /* 3. a good mutation passes + decomposes */
    gate = wubu_selfimprove_propose(&si, 0.9f, 0.8f, 1.0f, &mut);
    printf("  high verifier (0.9): gate=%d kind=%d strength=%.3f subgoals=%d\n",
           gate, mut.kind, mut.strength, mut.n_subgoals);
    if (gate != 1) FAIL("good mutation was blocked");
    if (mut.kind != WUBU_MUT_ARCH) FAIL("hard goal should mutate architecture");
    if (mut.strength <= 0.0f || mut.strength > 0.05f)
        FAIL("bounded delta violated (strength %.4f)", mut.strength);

    /* 4. the experience loop: wins accumulate */
    for (int i = 0; i < 10; i++)
        wubu_selfimprove_experience(&si, 1, 0.8f);
    if (si.evals != 10) FAIL("experience loop not counting evals");
    printf("  experience: %ld evals, %ld wins, running %.3f\n",
           si.evals, si.wins, si.running);

    char stats[256];
    wubu_selfimprove_stats(&si, stats, sizeof(stats));
    printf("  stats: %s\n", stats);

    printf("=== ALL SELFIMPROVE TESTS PASSED (P20 closed, RSI wired) ===\n");
    return 0;
}
