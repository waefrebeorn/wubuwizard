/*
 * test_events.c — the COLONY RUN RECORDER gate (post-AN54 A1).
 *
 * Asserts:
 *   1. events append as JSONL lines (one per round)
 *   2. the file survives an append/read round-trip (the post-mortem
 *      tools can read exactly what the run wrote)
 *   3. the attribution fields survive (cell/fisher/skill/traj — A4)
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "wubu_events.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_events (the colony run recorder) ===\n");

    const char *path = "/tmp/colony_test.events.jsonl";
    remove(path);

    wubu_events_t ev;
    if (wubu_events_open(&ev, path) != 0) FAIL("open");

    /* 2 rounds: one accept (with attribution), one reject */
    wubu_event_t e1;
    memset(&e1, 0, sizeof(e1));
    e1.round = 1;
    e1.loss = 9.8f;
    e1.suite_score = 0.62f;
    e1.verdict = 1;             /* accept */
    e1.policy_reason = 1;       /* loss rising */
    e1.mutation_rate = 0.47f;
    e1.n_contract_checks = 10;
    e1.cell_idx = 3;
    e1.prio_fisher = 0.56f;
    e1.skill_version = 7;
    e1.traj_id = 99;
    wubu_event_t e2 = e1;
    e2.round = 2;
    e2.loss = 9.9f;
    e2.verdict = 0;             /* reject */
    e2.policy_reason = 0;

    if (wubu_events_append(&ev, &e1) != 0) FAIL("append 1");
    if (wubu_events_append(&ev, &e2) != 0) FAIL("append 2");
    printf("  2 events appended (fsynced per event)\n");
    wubu_events_close(&ev);

    /* 2. the round-trip: the post-mortem tools read them back */
    wubu_event_t out[8];
    int n = wubu_events_read(path, out, 8);
    printf("  read back %d events\n", n);
    if (n != 2) FAIL("round-trip lost events (%d)", n);

    /* 3. the attribution survived */
    if (out[0].round != 1 || out[0].verdict != 1) FAIL("event 1 fields wrong");
    if (out[0].cell_idx != 3) FAIL("cell attribution lost");
    if (out[0].prio_fisher != 0.56f) FAIL("fisher attribution lost");
    if (out[0].skill_version != 7) FAIL("skill attribution lost");
    if (out[0].traj_id != 99) FAIL("traj attribution lost");
    if (out[1].verdict != 0) FAIL("event 2 verdict wrong");
    printf("  attribution intact: cell %u, fisher %.2f, skill %u, traj %llu\n",
           out[0].cell_idx, out[0].prio_fisher, out[0].skill_version,
           (unsigned long long)out[0].traj_id);

    remove(path);
    printf("=== ALL EVENTS TESTS PASSED (the run speaks JSONL) ===\n");
    return 0;
}
