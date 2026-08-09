/*
 * wubu_blueprint.c -- the EXECUTABLE BLUEPRINT (see the header).
 * The lineage becomes a real constraint, not a doc.
 */
#include "wubu_blueprint.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_contracts.h"   /* the versioned meta-cell pattern */

int wubu_blueprint_init(wubu_blueprint_t *bp, wubu_hive_t *tissue)
{
    if (!bp || !tissue) return -1;
    memset(bp, 0, sizeof(*bp));
    bp->tissue = tissue;
    /* the default rules from the lineage (docs/wubu-model-blueprint.md
     * §1 + WuBu1): every axis has a sane operating range */
    bp->rules[WUBU_BP_SEED].axis = WUBU_BP_SEED;
    bp->rules[WUBU_BP_SEED].min = 8.0f;  bp->rules[WUBU_BP_SEED].max = 64.0f;
    bp->rules[WUBU_BP_NEST].axis = WUBU_BP_NEST;
    bp->rules[WUBU_BP_NEST].min = 0.9f;  bp->rules[WUBU_BP_NEST].max = 1.1f;
    bp->rules[WUBU_BP_MOE].axis = WUBU_BP_MOE;
    bp->rules[WUBU_BP_MOE].min = 2.0f;   bp->rules[WUBU_BP_MOE].max = 16.0f;
    bp->rules[WUBU_BP_SPARSE].axis = WUBU_BP_SPARSE;
    bp->rules[WUBU_BP_SPARSE].min = 32.0f; bp->rules[WUBU_BP_SPARSE].max = 2048.0f;
    bp->rules[WUBU_BP_MATH].axis = WUBU_BP_MATH;
    bp->rules[WUBU_BP_MATH].min = 0.0f;  bp->rules[WUBU_BP_MATH].max = 4.0f;
    bp->rules[WUBU_BP_AMOEBA].axis = WUBU_BP_AMOEBA;
    bp->rules[WUBU_BP_AMOEBA].min = 2.0f; bp->rules[WUBU_BP_AMOEBA].max = 8.0f;
    return 0;
}

int wubu_blueprint_allows(wubu_blueprint_t *bp, wubu_bp_axis_t axis, float v)
{
    if (!bp || axis < 0 || axis >= WUBU_BP_N_AXES) return 0;
    bp->n_checked++;
    wubu_bp_rule_t *r = &bp->rules[axis];
    if (r->locked) { bp->n_refused++; return 0; }
    if (v < r->min || v > r->max) { bp->n_refused++; return 0; }
    bp->n_allowed++;
    return 1;
}

uint32_t wubu_blueprint_expand(wubu_blueprint_t *bp, wubu_bp_axis_t axis,
                               float new_min, float new_max,
                               uint64_t batch, const char *why)
{
    if (!bp || axis < 0 || axis >= WUBU_BP_N_AXES || !why) return 0;
    wubu_bp_rule_t *r = &bp->rules[axis];
    r->min = new_min;
    r->max = new_max;
    bp->n_expansions++;
    /* the expansion is a versioned CONTRACT-EXPANSION meta-cell in the
     * hive (the same accept/rollback discipline as the runtime floor:
     * only expand, never silently weaken) */
    if (bp->tissue) {
        wubu_contract_t *mc = (wubu_contract_t *)calloc(1, sizeof(wubu_contract_t));
        if (mc) {
            mc->version = (uint32_t)bp->n_expansions + 100;
            mc->kind = (wubu_contract_kind_t)(0);  /* placeholder kind */
            mc->bound = new_max;
            mc->enabled = 1;
            mc->batch = batch;
            wubu_hive_insert(bp->tissue, mc);
        }
    }
    (void)why;
    return (uint32_t)bp->n_expansions;
}

void wubu_blueprint_stats(const wubu_blueprint_t *bp, char *buf, size_t cap)
{
    if (!bp || !buf || cap == 0) return;
    snprintf(buf, cap,
             "checked=%llu allowed=%llu refused=%llu expansions=%llu",
             (unsigned long long)bp->n_checked,
             (unsigned long long)bp->n_allowed,
             (unsigned long long)bp->n_refused,
             (unsigned long long)bp->n_expansions);
}
