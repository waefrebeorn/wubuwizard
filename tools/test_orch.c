/*
 * test_orch.c -- the specialist-cell orchestrator gate (the user
 * directive #3: hierarchical planning — the goal decomposer spawns
 * short-lived specialist cells, the judge merges their outputs).
 *
 * Asserts:
 *   1. spawning creates the 4 lens cells (code/math/tool/critique)
 *   2. the judge merges by confidence-weighted vote
 *   3. the critique's confident veto blocks the decision (the DA
 *      pattern: the adversary argues)
 *   4. every cell + the judge are hive inserts (the colony IS the
 *      memory — the hive live count grows)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_agi.h"
#include "wubu_hive.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_orch (the specialist-cell orchestrator) ===\n");

    wubu_hive_t tissue;
    if (wubu_hive_init(&tissue) != 0) FAIL("hive init");

    wubu_orch_t orch;
    uint16_t goal = 42;
    int n = wubu_orch_spawn(&orch, &tissue, goal, 1);
    if (n != 4) FAIL("spawned %d cells (expected 4 lenses)", n);
    printf("  spawned %d specialist cells (code/math/tool/critique)\n", n);

    /* 1. the four lenses exist with the right defaults */
    if (orch.cells[WUBU_CELL_CODE].lens != WUBU_CELL_CODE) FAIL("code lens");
    if (orch.cells[WUBU_CELL_CRIT].lens != WUBU_CELL_CRIT) FAIL("crit lens");
    if (orch.cells[WUBU_CELL_CRIT].confidence != 0.1f) FAIL("crit should start low");

    /* 2. the judge merges: code+math confident, critique low */
    float conf[4] = { 0.9f, 0.8f, 0.3f, 0.1f };  /* code, math, tool, crit */
    int best = wubu_orch_judge(&orch, conf);
    printf("  judge picked lens %d (0=code 1=math 2=tool 3=crit), conf %.2f\n",
           best, orch.judge_confidence);
    if (best < 0 || best > 3) FAIL("judge returned a bad lens");
    if (orch.cells[best].accepted != 1) FAIL("judge did not mark the winner");
    if (best == WUBU_CELL_CRIT) FAIL("the critique should not win with low conf");

    /* 3. the critique veto: a confident rejection blocks the decision */
    wubu_orch_t orch2;
    wubu_orch_spawn(&orch2, &tissue, goal, 2);
    float conf2[4] = { 0.9f, 0.9f, 0.9f, 0.9f };  /* the adversary is confident */
    int veto = wubu_orch_judge(&orch2, conf2);
    printf("  critique veto: decision=%u (goal=%u) conf %.2f\n",
           orch2.decision, goal, orch2.judge_confidence);
    if (veto != 0) FAIL("the critique veto did not block");
    if (orch2.decision != goal) FAIL("the veto must keep the goal");

    /* 4. the colony IS the memory: 4 spawn + 2 judge inserts */
    size_t live = wubu_hive_live(&tissue);
    printf("  hive live cells: %zu (4 spawns + 2 judges = 6 expected)\n", live);
    if (live < 6) FAIL("the cells/judge are not in the hive");

    printf("=== ALL ORCH TESTS PASSED (the colony delegates + argues) ===\n");
    return 0;
}
