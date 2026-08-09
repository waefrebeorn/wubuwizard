/*
 * wubu_skillcell.h -- THE SKILL CURRICULUM (the post-AN39 wave,
 * Phase 4: Pillar 17). C11. The colony adds skills from experience,
 * reuses them, and shrinks them when they stop helping.
 *
 *   - recurring successful traj patterns become VERSIONED SKILL CELLS
 *     in the hive (NOT a separate skill DB — the hive is the memory)
 *   - failed high-partial traj + capgate gaps propose skill DRAFTS;
 *     the contracts + fitness gate accept them
 *   - the orchestrator prefers spawning specialists that match an
 *     existing skill cell before inventing new ones
 *   - the selfimprove traces record skill USE so extinction can
 *     prune unused skills
 *
 * Pure C11, opaque, no third party.
 */
#ifndef WUBU_SKILLCELL_H
#define WUBU_SKILLCELL_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_hive.h"

/* one skill cell (a versioned hive insert) */
typedef struct {
    uint32_t   version;      /* the skill version (only grows) */
    uint16_t   goal_token;   /* the goal domain this skill serves */
    uint8_t    lens;         /* the specialist lens it matches */
    uint32_t   pattern_hash; /* the recurring traj pattern hash */
    float      fitness;      /* the skill's measured fitness */
    uint8_t    draft;        /* 1 = a proposed draft (not yet accepted) */
    uint64_t   uses;         /* how many times the colony used it */
    uint64_t   batch;        /* provenance */
} wubu_skill_t;

/* the skill registry state */
typedef struct {
    wubu_hive_t *tissue;
    uint32_t next_version;
    /* telemetry */
    uint64_t n_drafts, n_accepted, n_pruned;
} wubu_skill_tracker_t;

/* K1: init. */
int wubu_skill_init(wubu_skill_tracker_t *sk, wubu_hive_t *tissue);

/* K2: propose a skill DRAFT from a successful traj pattern (or a
 * failed high-partial one — the curriculum's input). The draft lands
 * in the hive; the fitness gate accepts it later. Returns the draft's
 * hive cell index (or -1). */
int64_t wubu_skill_propose(wubu_skill_tracker_t *sk, uint16_t goal,
                           uint8_t lens, uint32_t pattern_hash,
                           float fitness, uint64_t batch);

/* K3: accept a draft (it passed the contracts + fitness gate): the
 * draft becomes a versioned skill (draft=0). Returns the version. */
uint32_t wubu_skill_accept(wubu_skill_tracker_t *sk, int64_t cell_idx);

/* K4: find the best matching skill for a goal+lens (the orchestrator
 * prefers this BEFORE spawning a new specialist). Returns the cell
 * index or -1. */
int64_t wubu_skill_match(wubu_skill_tracker_t *sk, uint16_t goal,
                         uint8_t lens, float tol);

/* K5: record a skill USE (the selfimprove traces feed this so
 * extinction can prune unused skills). */
void wubu_skill_use(wubu_skill_tracker_t *sk, int64_t cell_idx);

/* K6: the extinction pass — skills with uses==0 and low fitness get
 * pruned (the hive erases them). Returns the pruned count. */
int wubu_skill_prune(wubu_skill_tracker_t *sk, uint64_t min_uses,
                     float min_fitness);

/* K7: the tracker stats. */
void wubu_skill_stats(const wubu_skill_tracker_t *sk, char *buf, size_t cap);

#endif
