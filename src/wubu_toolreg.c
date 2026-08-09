/*
 * wubu_toolreg.c -- the DENY-BY-DEFAULT TOOL REGISTRY (see the
 * header). Real agency without ambient host authority.
 */
#include "wubu_toolreg.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_trajcell.h"

int wubu_toolreg_init(wubu_toolreg_t *tr, wubu_hive_t *tissue,
                      float max_fail_rate)
{
    if (!tr || !tissue) return -1;
    memset(tr, 0, sizeof(*tr));
    tr->tissue = tissue;
    tr->max_fail_rate = max_fail_rate > 0 ? max_fail_rate : 0.5f;
    return 0;
}

static wubu_tool_t *find_tool(wubu_toolreg_t *tr, const char *name)
{
    for (int i = 0; i < tr->n; i++)
        if (strcmp(tr->tools[i].name, name) == 0) return &tr->tools[i];
    return NULL;
}

int wubu_toolreg_register(wubu_toolreg_t *tr, const char *name,
                          uint8_t kind, uint64_t batch)
{
    if (!tr || !name || tr->n >= 32) return -1;
    wubu_tool_t *t = &tr->tools[tr->n++];
    memset(t, 0, sizeof(*t));
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->kind = kind;
    t->enabled = 1;      /* registered = explicitly enabled (deny-default) */
    t->batch = batch;
    return tr->n - 1;
}

int wubu_toolreg_run(wubu_toolreg_t *tr, const char *name,
                     uint16_t goal, uint8_t cell_idx, float cost)
{
    if (!tr || !name || !tr->tissue) return 0;
    wubu_tool_t *t = find_tool(tr, name);
    int allowed = (t && t->enabled) ? 1 : 0;
    /* the fail-rate throttle: a thrashing tool is barred even if it
     * is enabled (the metadiag raises the bar) */
    if (allowed && t->uses >= 4 && t->fails > 0) {
        float rate = (float)t->fails / (float)t->uses;
        if (rate > tr->max_fail_rate) { allowed = 0; t->enabled = 0; }
    }
    if (allowed) { tr->n_allowed++; t->uses++; t->cost_sum += cost; }
    else         { tr->n_denied++; }
    /* EVERY action is a traj cell with cost + outcome (the hive
     * records what the colony actually tried) */
    wubu_traj_tracker_t tt;
    wubu_traj_init(&tt, tr->tissue);
    uint32_t steps[1] = { (uint32_t)goal };
    wubu_traj_record(&tt, goal, steps, 1, allowed, allowed ? 1.0f : 0.0f,
                     allowed ? 1.0f : 0.0f, cost, t ? t->batch : 0);
    return allowed;
}

void wubu_toolreg_report(wubu_toolreg_t *tr, const char *name, int ok)
{
    if (!tr || !name) return;
    wubu_tool_t *t = find_tool(tr, name);
    if (!t) return;
    if (!ok) t->fails++;
}

void wubu_toolreg_stats(const wubu_toolreg_t *tr, char *buf, size_t cap)
{
    if (!tr || !buf || cap == 0) return;
    snprintf(buf, cap,
             "tools=%d allowed=%llu denied=%llu fail_rate_max=%.2f",
             tr->n,
             (unsigned long long)tr->n_allowed,
             (unsigned long long)tr->n_denied,
             tr->max_fail_rate);
}
