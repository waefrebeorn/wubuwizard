/*
 * wubu_blueprint.h -- THE EXECUTABLE BLUEPRINT (the post-AN39 wave,
 * Phase 6: WB01 — "the design doc is still open. Turn the lineage
 * into an executable blueprint that the training + diagnose path
 * follows"). C11.
 *
 * The WuBu lineage (from docs/wubu-model-blueprint.md + WuBu1):
 *   seed + nesting + MoE + sparse attention + math-RL + amoeba.
 * This module is the machine-readable plan:
 *   - the ALLOWED mutation axes (the blueprint's degrees of freedom)
 *   - a mutation is ON-BLUEPRINT only when its axis is allowed AND
 *     within the axis bound (e.g. nesting curvature, routing counts,
 *     precision per the ladder)
 *   - an OFF-BLUEPRINT mutation is REFUSED by default — it requires
 *     an explicit contract-expansion meta-cell (the runtime contracts
 *     module already versioned the floor; the blueprint extends it to
 *     the ARCHITECTURE axes)
 *
 * WB01 is wired only when a train+diagnose run refuses off-blueprint
 * mutations by default.
 */
#ifndef WUBU_BLUEPRINT_H
#define WUBU_BLUEPRINT_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_hive.h"

/* the blueprint axes (the lineage's degrees of freedom) */
typedef enum {
    WUBU_BP_SEED = 0,        /* the seed: base dims, layers */
    WUBU_BP_NEST = 1,        /* the hyperbolic nesting curvature */
    WUBU_BP_MOE = 2,         /* the mixed-agents routing (expert count) */
    WUBU_BP_SPARSE = 3,      /* the sparse attention window */
    WUBU_BP_MATH = 4,        /* the math-RL reward / prover depth */
    WUBU_BP_AMOEBA = 5,      /* the colony: cell count, survival floor */
    WUBU_BP_N_AXES = 6
} wubu_bp_axis_t;

/* the per-axis bound (what the blueprint allows) */
typedef struct {
    wubu_bp_axis_t axis;
    float min, max;          /* the allowed range */
    int   locked;            /* 1 = no mutations on this axis at all */
} wubu_bp_rule_t;

/* the blueprint state */
typedef struct {
    wubu_hive_t *tissue;
    wubu_bp_rule_t rules[WUBU_BP_N_AXES];
    uint64_t n_checked, n_refused, n_allowed;
    uint64_t n_expansions;   /* contract-expansion meta-cells written */
} wubu_blueprint_t;

/* B1: init with the blueprint's default rules (the lineage axes). */
int wubu_blueprint_init(wubu_blueprint_t *bp, wubu_hive_t *tissue);

/* B2: the gate — is a mutation on `axis` with value `v` ON the
 * blueprint? Locked axes always refuse. Returns 1 = allowed. */
int wubu_blueprint_allows(wubu_blueprint_t *bp, wubu_bp_axis_t axis, float v);

/* B3: an OFF-BLUEPRINT mutation requires an explicit expansion: write
 * a contract-expansion meta-cell (versioned) that records WHY the
 * axis bound moved. Returns the new bound's version (0 = refused —
 * the blueprint does NOT expand unless the caller really overrides). */
uint32_t wubu_blueprint_expand(wubu_blueprint_t *bp, wubu_bp_axis_t axis,
                               float new_min, float new_max,
                               uint64_t batch, const char *why);

/* B4: the blueprint stats. */
void wubu_blueprint_stats(const wubu_blueprint_t *bp, char *buf, size_t cap);

#endif
