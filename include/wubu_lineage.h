/*
 * wubu_lineage.h -- LINEAGE-AWARE FITNESS + EXTINCTION PRESSURE
 * (the extended roadmap #1). C11.
 *
 * The amoeba becomes actual evolutionary pressure with memory:
 *   - every accepted mutation records its parent cell IDs + a compact
 *     lineage hash (the ancestry chain)
 *   - fitness = immediate loss/task success AND the multi-generation
 *     survival rate (a lineage that keeps winning survives)
 *   - SOFT EXTINCTION: lineages that stop improving for N diagnose
 *     cycles get progressive shrink probability even when individual
 *     cells still pass the floor
 *   - the archive keeps the winning lineage AND a small sample of
 *     near-miss cousins (so later diagnose avoids rediscovering dead
 *     ends)
 *
 * Pure C11, opaque, no third party.
 */
#ifndef WUBU_LINEAGE_H
#define WUBU_LINEAGE_H

#include <stdint.h>
#include <stddef.h>

/* the lineage entry (one cell lineage's life) */
typedef struct {
    uint64_t lineage_id;     /* the lineage hash (compact) */
    uint64_t parent_id;      /* the parent lineage (0 = root) */
    float    fitness;        /* the current best fitness */
    float    best_fitness;   /* the all-time best (never falls) */
    uint64_t generations;    /* accepted mutations in this lineage */
    int      stagnant_for;   /* diagnose cycles without improvement */
    float    survival;       /* the multi-gen survival rate (0..1) */
    uint8_t  extinct;        /* 1 = soft-extinct (shrink pressure) */
} wubu_lineage_t;

/* the lineage registry state */
typedef struct wubu_lineage_tracker {
    wubu_lineage_t *lines;    /* [cap] */
    int  n, cap;
    int  extinction_window;   /* stagnant-for this many -> pressure */
    float extinction_rate;    /* the shrink probability per cycle */
    uint64_t next_id;
    /* telemetry */
    uint64_t n_extinctions;
    uint64_t n_improvements;
} wubu_lineage_tracker_t;

/* L1: init. */
int wubu_lineage_init(wubu_lineage_tracker_t *lt, int cap,
                      int extinction_window, float extinction_rate);

/* L2: register a mutation outcome. The parent lineage (or 0 for a
 * fresh lineage) + the new fitness. Returns the lineage id. A lineage
 * that improves is a win (survival up); stagnation is counted. */
uint64_t wubu_lineage_record(wubu_lineage_tracker_t *lt, uint64_t parent_id,
                             float fitness, float prev_fitness);

/* L3: the extinction pass (call on the slow diagnose schedule): any
 * lineage stagnant for >= the window gets progressive shrink
 * probability. Returns the count soft-extinct this pass. */
int wubu_lineage_extinction_pass(wubu_lineage_tracker_t *lt);

/* L4: the survival rate of a lineage (the fitness component: a
 * lineage that keeps winning survives longer). */
float wubu_lineage_survival(const wubu_lineage_tracker_t *lt,
                            uint64_t lineage_id);

/* L5: the near-miss sample: lineages that were close (best fitness
 * within tol of the winner) but didn't win — the archive keeps them
 * so diagnose avoids rediscovering dead ends. Returns the count. */
int wubu_lineage_nearmiss(const wubu_lineage_tracker_t *lt,
                          float winner_fitness, float tol,
                          uint64_t *out, int out_cap);

/* L6: the tracker stats. */
void wubu_lineage_stats(const wubu_lineage_tracker_t *lt, char *buf, size_t cap);

/* L7: free. */
void wubu_lineage_free(wubu_lineage_tracker_t *lt);

#endif
