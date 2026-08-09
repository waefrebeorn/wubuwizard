/*
 * wubu_toolreg.h -- THE DENY-BY-DEFAULT TOOL REGISTRY (the post-AN39
 * wave, Phase 5: Pillars 13/15 — "push real agency without opening
 * the host"). C11.
 *
 * The ONLY path from a specialist cell to an external effect:
 *   - a tool is REGISTERED (explicitly enabled) before it can run —
 *     deny-by-default, no ambient host authority
 *   - every external action is a TRAJ CELL with cost + outcome (the
 *     hive records what the colony actually did)
 *   - the metadiag can raise the bar if external actions fail or
 *     thrash (the registry tracks the failure rate)
 *
 * The registry is the Colonel's capability surface: the Brain speaks
 * only through registered tools; the host is never ambiently open.
 */
#ifndef WUBU_TOOLREG_H
#define WUBU_TOOLREG_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_hive.h"

/* one registered tool */
typedef struct {
    char     name[64];       /* the tool name (the capability path) */
    uint8_t  kind;           /* 0=read 1=write 2=exec 3=delegate */
    int      enabled;        /* 1 = the deny-by-default gate is OPEN */
    uint64_t uses;
    uint64_t fails;
    float    cost_sum;       /* the accumulated cost */
    uint64_t batch;          /* when it was registered */
} wubu_tool_t;

/* the registry state */
typedef struct {
    wubu_hive_t *tissue;     /* the traj cells land here */
    wubu_tool_t tools[32];   /* the registered set (bounded) */
    int  n;
    /* the throttle: a tool that fails/thrashes gets barred */
    float max_fail_rate;
    /* telemetry */
    uint64_t n_denied, n_allowed;
} wubu_toolreg_t;

/* R1: init (deny-by-default: nothing is enabled until registered). */
int wubu_toolreg_init(wubu_toolreg_t *tr, wubu_hive_t *tissue,
                      float max_fail_rate);

/* R2: register a tool (the ONLY way it becomes runnable). Returns the
 * tool index or -1. */
int wubu_toolreg_register(wubu_toolreg_t *tr, const char *name,
                          uint8_t kind, uint64_t batch);

/* R3: the gate — a specialist cell's action is ALLOWED only if the
 * tool is registered + enabled + not thrashing. The action (whether
 * allowed or denied) is a TRAJ CELL with cost. Returns 1 = allowed. */
int wubu_toolreg_run(wubu_toolreg_t *tr, const char *name,
                     uint16_t goal, uint8_t cell_idx, float cost);

/* R4: report an outcome (0 = fail). Failures count toward the
 * throttle; a thrashing tool is auto-disabled (the metadiag raises
 * the bar). */
void wubu_toolreg_report(wubu_toolreg_t *tr, const char *name, int ok);

/* R5: the registry stats. */
void wubu_toolreg_stats(const wubu_toolreg_t *tr, char *buf, size_t cap);

#endif
