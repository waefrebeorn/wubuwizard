/*
 * wubu_trajcell.c -- VERIFIED TOOL-TRAJECTORY CELLS (see the header).
 * Tool use joins the hive's fitness model; failures become seeds.
 */
#include "wubu_trajcell.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int wubu_traj_init(wubu_traj_tracker_t *tt, wubu_hive_t *tissue)
{
    if (!tt || !tissue) return -1;
    memset(tt, 0, sizeof(*tt));
    tt->tissue = tissue;
    tt->next_traj_id = 1;
    return 0;
}

uint32_t wubu_traj_hash(const uint32_t *step_hashes, int n_steps,
                        uint16_t goal)
{
    /* FNV-1a over the goal + the step hashes — the trajectory's
     * compact identity (dedup + similarity) */
    unsigned h = 2166136261u;
    unsigned v = goal;
    for (int b = 0; b < 4; b++) { h ^= (v >> (8 * b)) & 0xFF; h *= 16777619u; }
    for (int i = 0; i < n_steps; i++) {
        v = step_hashes[i];
        for (int b = 0; b < 4; b++) { h ^= (v >> (8 * b)) & 0xFF; h *= 16777619u; }
    }
    return (uint32_t)h;
}

int64_t wubu_traj_record(wubu_traj_tracker_t *tt, uint16_t goal,
                         const uint32_t *step_hashes, int n_steps,
                         int ok, float outcome, float partial_credit,
                         float cost, uint64_t batch)
{
    if (!tt || !tt->tissue) return -1;
    wubu_trajcell_t *c = (wubu_trajcell_t *)calloc(1, sizeof(wubu_trajcell_t));
    if (!c) return -1;
    c->traj_id = tt->next_traj_id++;
    c->goal_token = goal;
    c->n_steps = (uint8_t)(n_steps > 255 ? 255 : (n_steps < 0 ? 0 : n_steps));
    c->ok = ok ? 1 : 0;
    c->outcome = outcome;
    c->partial_credit = partial_credit < 0 ? 0 : (partial_credit > 1 ? 1 : partial_credit);
    c->cost = cost;
    c->hash = wubu_traj_hash(step_hashes, n_steps, goal);
    c->batch = batch;
    wubu_hive_insert(tt->tissue, c);
    if (ok) tt->n_success++; else tt->n_failed++;
    return (int64_t)c->traj_id;
}

/* the walk helper context for similar/seeds queries */
typedef struct {
    uint16_t goal;
    float    tol;
    float    credit_th;
    int      seeds_mode;
    int64_t *out;
    int      cap;
    int      k;
} traj_query_ctx_t;

static int traj_query_cb(void *ptr, void *user)
{
    traj_query_ctx_t *ctx = (traj_query_ctx_t *)user;
    wubu_trajcell_t *c = (wubu_trajcell_t *)ptr;
    if (ctx->seeds_mode) {
        /* mutation seeds: failed + high partial credit */
        if (!c->ok && c->partial_credit >= ctx->credit_th && ctx->k < ctx->cap)
            ctx->out[ctx->k++] = (int64_t)c->traj_id;
    } else {
        /* similar: successful + goal within tol */
        int d = (int)c->goal_token - (int)ctx->goal;
        if (d < 0) d = -d;
        if (c->ok && d <= (int)ctx->tol && ctx->k < ctx->cap)
            ctx->out[ctx->k++] = (int64_t)c->traj_id;
    }
    return 0;
}

int wubu_traj_similar(const wubu_traj_tracker_t *tt, uint16_t goal,
                      float tol, int64_t *out, int out_cap)
{
    if (!tt || !tt->tissue || !out || out_cap <= 0) return 0;
    traj_query_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.goal = goal;
    ctx.tol = tol;
    ctx.seeds_mode = 0;
    ctx.out = out; ctx.cap = out_cap;
    wubu_hive_foreach(tt->tissue, traj_query_cb, &ctx);
    return ctx.k;
}

int wubu_traj_seeds(const wubu_traj_tracker_t *tt, float credit_th,
                    int64_t *out, int out_cap)
{
    if (!tt || !tt->tissue || !out || out_cap <= 0) return 0;
    traj_query_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.credit_th = credit_th;
    ctx.seeds_mode = 1;
    ctx.out = out; ctx.cap = out_cap;
    wubu_hive_foreach(tt->tissue, traj_query_cb, &ctx);
    return ctx.k;
}

void wubu_traj_stats(const wubu_traj_tracker_t *tt, char *buf, size_t cap)
{
    if (!tt || !buf || cap == 0) return;
    snprintf(buf, cap, "trajs=%llu success=%llu failed=%llu",
             (unsigned long long)tt->next_traj_id - 1,
             (unsigned long long)tt->n_success,
             (unsigned long long)tt->n_failed);
}
