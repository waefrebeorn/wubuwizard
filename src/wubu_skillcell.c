/*
 * wubu_skillcell.c -- the SKILL CURRICULUM (see the header). The
 * colony learns skills from experience, reuses them, prunes them.
 */
#include "wubu_skillcell.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int wubu_skill_init(wubu_skill_tracker_t *sk, wubu_hive_t *tissue)
{
    if (!sk || !tissue) return -1;
    memset(sk, 0, sizeof(*sk));
    sk->tissue = tissue;
    sk->next_version = 1;
    return 0;
}

/* find a cell by index (the hive has no index lookup — the foreach
 * visits in slot order; we track via the pointer identity instead).
 * The cell_idx returned by propose/accept is the CELL's own pointer
 * offset cast — the caller keeps it opaque. */
static wubu_skill_t *find_cell(wubu_hive_t *h, int64_t idx)
{
    /* idx encodes the pointer (the hive keeps pointers stable) */
    if (idx <= 0) return NULL;
    return (wubu_skill_t *)(intptr_t)idx;
}

int64_t wubu_skill_propose(wubu_skill_tracker_t *sk, uint16_t goal,
                           uint8_t lens, uint32_t pattern_hash,
                           float fitness, uint64_t batch)
{
    if (!sk || !sk->tissue) return -1;
    wubu_skill_t *cell = (wubu_skill_t *)calloc(1, sizeof(wubu_skill_t));
    if (!cell) return -1;
    cell->version = sk->next_version;   /* the draft's provisional version */
    cell->goal_token = goal;
    cell->lens = lens;
    cell->pattern_hash = pattern_hash;
    cell->fitness = fitness;
    cell->draft = 1;
    cell->uses = 0;
    cell->batch = batch;
    wubu_hive_insert(sk->tissue, cell);
    sk->n_drafts++;
    /* the opaque cell index = the pointer (the hive keeps it stable) */
    return (int64_t)(intptr_t)cell;
}

uint32_t wubu_skill_accept(wubu_skill_tracker_t *sk, int64_t cell_idx)
{
    wubu_skill_t *cell = find_cell(sk ? sk->tissue : NULL, cell_idx);
    if (!cell || !cell->draft) return 0;
    cell->draft = 0;
    cell->version = sk->next_version++;
    sk->n_accepted++;
    return cell->version;
}

/* the match walk context */
typedef struct {
    uint16_t goal;
    uint8_t  lens;
    float    tol;
    int64_t  best;
    float    best_fit;
} skill_match_ctx_t;

static int skill_match_cb(void *ptr, void *user)
{
    skill_match_ctx_t *ctx = (skill_match_ctx_t *)user;
    wubu_skill_t *cell = (wubu_skill_t *)ptr;
    if (cell->draft) return 0;   /* only accepted skills match */
    int dg = (int)cell->goal_token - (int)ctx->goal;
    if (dg < 0) dg = -dg;
    if (cell->lens != ctx->lens) return 0;
    if (dg <= (int)ctx->tol && cell->fitness > ctx->best_fit) {
        ctx->best_fit = cell->fitness;
        ctx->best = (int64_t)(intptr_t)cell;
    }
    return 0;
}

int64_t wubu_skill_match(wubu_skill_tracker_t *sk, uint16_t goal,
                         uint8_t lens, float tol)
{
    if (!sk || !sk->tissue) return -1;
    skill_match_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.goal = goal;
    ctx.lens = lens;
    ctx.tol = tol;
    ctx.best = -1;
    ctx.best_fit = -1.0f;
    wubu_hive_foreach(sk->tissue, skill_match_cb, &ctx);
    return ctx.best;
}

void wubu_skill_use(wubu_skill_tracker_t *sk, int64_t cell_idx)
{
    wubu_skill_t *cell = find_cell(sk ? sk->tissue : NULL, cell_idx);
    if (!cell || cell->draft) return;
    cell->uses++;
}

/* the prune: TWO PASSES — collect the doomed pointers, then erase
 * (the hive walk cannot erase-while-iterating: the block being walked
 * would be mutated). */
typedef struct {
    uint64_t min_uses;
    float    min_fitness;
    void    *doomed[64];
    int      n;
} skill_prune_ctx_t;

static int skill_prune_cb(void *ptr, void *user)
{
    skill_prune_ctx_t *ctx = (skill_prune_ctx_t *)user;
    wubu_skill_t *cell = (wubu_skill_t *)ptr;
    if (!cell->draft && cell->uses < ctx->min_uses &&
        cell->fitness < ctx->min_fitness && ctx->n < 64) {
        ctx->doomed[ctx->n++] = ptr;
    }
    return 0;
}

int wubu_skill_prune(wubu_skill_tracker_t *sk, uint64_t min_uses,
                     float min_fitness)
{
    if (!sk || !sk->tissue) return 0;
    skill_prune_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.min_uses = min_uses;
    ctx.min_fitness = min_fitness;
    wubu_hive_foreach(sk->tissue, skill_prune_cb, &ctx);
    for (int i = 0; i < ctx.n; i++) {
        wubu_hive_erase(sk->tissue, ctx.doomed[i]);
        free(ctx.doomed[i]);
    }
    sk->n_pruned += (uint64_t)ctx.n;
    return ctx.n;
}

void wubu_skill_stats(const wubu_skill_tracker_t *sk, char *buf, size_t cap)
{
    if (!sk || !buf || cap == 0) return;
    snprintf(buf, cap,
             "drafts=%llu accepted=%llu pruned=%llu next_version=%u",
             (unsigned long long)sk->n_drafts,
             (unsigned long long)sk->n_accepted,
             (unsigned long long)sk->n_pruned,
             sk->next_version);
}
