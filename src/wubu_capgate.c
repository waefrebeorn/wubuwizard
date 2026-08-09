/*
 * wubu_capgate.c -- CAPABILITY-GATED SPECIALIST SPAWNING (see the
 * header). Prevents unbounded agent spawning; makes missing skills
 * visible to the standing loop.
 */
#include "wubu_capgate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int wubu_capgate_init(wubu_capgate_t *cg, wubu_hive_t *tissue,
                      float min_headroom, float similar_tol,
                      float similar_recent_max)
{
    if (!cg || !tissue) return -1;
    memset(cg, 0, sizeof(*cg));
    cg->tissue = tissue;
    cg->min_headroom = min_headroom > 0 ? min_headroom : 0.3f;
    cg->similar_tol = similar_tol > 0 ? similar_tol : 0.15f;
    cg->similar_recent_max = similar_recent_max > 0 ? similar_recent_max : 50.0f;
    return 0;
}

wubu_cap_verdict_t wubu_capgate_check(wubu_capgate_t *cg,
                                      uint8_t lens, uint16_t goal,
                                      const wubu_cap_input_t *in)
{
    if (!cg || !in) return WUBU_CAP_DENY;
    (void)lens; (void)goal;
    /* 1. the hard safety floor: the backend must have headroom */
    if (in->headroom < cg->min_headroom) return WUBU_CAP_DECOMPOSE;
    /* 2. the required tool must be present */
    if (!in->tool_present) return WUBU_CAP_FALLBACK;
    /* 3. the domain verifier must be ready */
    if (!in->verifier_ready) return WUBU_CAP_FALLBACK;
    /* 4. no similar specialist lineage with recent positive fitness:
     * a redundant spawn is a waste (the colony already has that lens
     * working on the goal) */
    if (in->similar_fitness >= 0.0f &&
        in->similar_recent <= cg->similar_recent_max &&
        in->similar_fitness > 0.5f)
        return WUBU_CAP_DENY;
    return WUBU_CAP_PASS;
}

int wubu_capgate_gap(wubu_capgate_t *cg, uint8_t lens, uint16_t goal,
                     uint8_t reason, uint64_t batch)
{
    if (!cg || !cg->tissue) return -1;
    wubu_cap_gap_t *gap = (wubu_cap_gap_t *)calloc(1, sizeof(wubu_cap_gap_t));
    if (!gap) return -1;
    gap->goal_token = goal;
    gap->lens = lens;
    gap->reason = reason;
    gap->batch = batch;
    wubu_hive_insert(cg->tissue, gap);
    cg->n_gaps++;
    return 0;
}

/* the walk helper for gap counting */
typedef struct { int n; } gap_count_ctx_t;

static int gap_count_cb(void *ptr, void *user)
{
    (void)ptr;
    ((gap_count_ctx_t *)user)->n++;
    return 0;
}

size_t wubu_capgate_gap_count(const wubu_capgate_t *cg)
{
    if (!cg || !cg->tissue) return 0;
    return cg->n_gaps;
}

void wubu_capgate_stats(const wubu_capgate_t *cg, char *buf, size_t cap)
{
    if (!cg || !buf || cap == 0) return;
    snprintf(buf, cap, "gaps=%llu headroom>=%.2f similar_tol=%.2f recent<=%.0f",
             (unsigned long long)cg->n_gaps, cg->min_headroom,
             cg->similar_tol, cg->similar_recent_max);
}
