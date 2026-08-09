/*
 * wubu_capgate.h -- CAPABILITY-GATED SPECIALIST SPAWNING (roadmap
 * #4). C11. Hierarchical planning needs a hard gate so the colony
 * does not spawn specialists it cannot actually run or verify.
 *
 * Before inserting a specialist cell, the planner checks:
 *   - backend bandwidth / thermal headroom (the Body reports it)
 *   - required tools present (the capability registry)
 *   - a verifier/oracle available for that domain
 *   - no similar specialist lineage with recent positive fitness
 *     already exists (avoid redundant spawns)
 *
 * If the gate FAILS, the planner either decomposes further into
 * cheaper sub-goals or falls back to a single generalist cell with an
 * explicit CAPABILITY-GAP record in the hive. The gaps themselves are
 * first-class cells the slow diagnose path can turn into training or
 * tool-acquisition goals.
 */
#ifndef WUBU_CAPGATE_H
#define WUBU_CAPGATE_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_hive.h"

/* the capability gate inputs (the Body + the hive report) */
typedef struct {
    float  headroom;         /* 0..1 the backend thermal/bandwidth headroom */
    int    tool_present;     /* the required tool is registered */
    int    verifier_ready;   /* the domain verifier/oracle is available */
    float  similar_fitness;  /* -1 = none; else the recent fitness of a
                                similar specialist lineage */
    float  similar_recent;   /* how recent (batches ago) */
} wubu_cap_input_t;

/* the gate verdict */
typedef enum {
    WUBU_CAP_PASS = 0,       /* spawn the specialist */
    WUBU_CAP_DECOMPOSE = 1,  /* too heavy: split into cheaper sub-goals */
    WUBU_CAP_FALLBACK = 2,   /* use a single generalist + record the gap */
    WUBU_CAP_DENY = 3        /* unsafe: no spawn at all */
} wubu_cap_verdict_t;

/* the capability-gap record (a hive cell) */
typedef struct {
    uint16_t goal_token;     /* what could not be spawned for */
    uint8_t  lens;           /* the specialist lens that was gated */
    uint8_t  reason;         /* 0=headroom 1=tool 2=verifier 3=redundant */
    uint64_t batch;          /* provenance */
} wubu_cap_gap_t;

/* the gate state */
typedef struct {
    wubu_hive_t *tissue;
    float min_headroom;      /* the spawn threshold */
    float similar_tol;       /* how close a similar lineage counts */
    float similar_recent_max;/* how recent it must be to count */
    uint64_t n_gaps;
} wubu_capgate_t;

/* C1: init. */
int wubu_capgate_init(wubu_capgate_t *cg, wubu_hive_t *tissue,
                      float min_headroom, float similar_tol,
                      float similar_recent_max);

/* C2: the gate. Returns the verdict for spawning a specialist cell of
 * lens `lens` for goal `goal`. */
wubu_cap_verdict_t wubu_capgate_check(wubu_capgate_t *cg,
                                      uint8_t lens, uint16_t goal,
                                      const wubu_cap_input_t *in);

/* C3: record a capability gap as a hive cell (first-class — the slow
 * diagnose path reads them as training/tool-acquisition goals). */
int wubu_capgate_gap(wubu_capgate_t *cg, uint8_t lens, uint16_t goal,
                     uint8_t reason, uint64_t batch);

/* C4: count the gaps in the hive (the visible missing skills). */
size_t wubu_capgate_gap_count(const wubu_capgate_t *cg);

/* C5: the gate stats. */
void wubu_capgate_stats(const wubu_capgate_t *cg, char *buf, size_t cap);

#endif
