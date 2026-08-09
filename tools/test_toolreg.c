/*
 * test_toolreg.c -- the DENY-BY-DEFAULT TOOL REGISTRY gate (Phase 5:
 * Pillars 13/15 — "push real agency without opening the host").
 *
 * Asserts:
 *   1. an UNREGISTERED tool is denied (deny-by-default, no ambient
 *      host authority)
 *   2. a REGISTERED tool is allowed
 *   3. every action is a traj cell in the hive (cost + outcome)
 *   4. a thrashing tool (high fail rate) is auto-barred
 */
#include <stdio.h>
#include <string.h>

#include "wubu_toolreg.h"
#include "wubu_hive.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_toolreg (deny-by-default agency) ===\n");

    wubu_hive_t tissue;
    if (wubu_hive_init(&tissue) != 0) FAIL("hive init");

    wubu_toolreg_t tr;
    if (wubu_toolreg_init(&tr, &tissue, 0.5f) != 0) FAIL("toolreg init");

    /* 1. the unregistered tool is DENIED (deny-by-default) */
    int r = wubu_toolreg_run(&tr, "fetch_url", 42, 0, 1.0f);
    printf("  unregistered 'fetch_url': allowed=%d (0 = denied)\n", r);
    if (r != 0) FAIL("the unregistered tool ran (ambient authority leak)");

    /* 2. register + run: the ONLY path to an effect */
    int idx = wubu_toolreg_register(&tr, "fetch_url", 0, 1);
    if (idx < 0) FAIL("registration failed");
    r = wubu_toolreg_run(&tr, "fetch_url", 42, 0, 1.0f);
    printf("  registered 'fetch_url': allowed=%d (1 = the gate is open)\n", r);
    if (r != 1) FAIL("the registered tool was denied");

    /* 3. every action is a traj cell */
    size_t live = wubu_hive_live(&tissue);
    printf("  hive live cells: %zu (2 actions = 2 traj cells)\n", live);
    if (live != 2) FAIL("actions not recorded as traj cells");

    /* 4. the throttle: fail the tool repeatedly -> auto-barred */
    for (int i = 0; i < 6; i++) {
        wubu_toolreg_run(&tr, "fetch_url", 43, 1, 0.5f);
        wubu_toolreg_report(&tr, "fetch_url", 0);   /* every run fails */
    }
    int barred = wubu_toolreg_run(&tr, "fetch_url", 44, 1, 0.5f);
    printf("  thrashing tool (6 fails): allowed=%d (0 = auto-barred)\n", barred);
    if (barred != 0) FAIL("the thrashing tool was not barred");

    char stats[256];
    wubu_toolreg_stats(&tr, stats, sizeof(stats));
    printf("  stats: %s\n", stats);

    printf("=== ALL TOOLREG TESTS PASSED (agency without ambient authority) ===\n");
    return 0;
}
