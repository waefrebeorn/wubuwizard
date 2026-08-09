/*
 * wubu_pref.h -- the RLHF ORACLE (the user directive #4: "give the
 * oracle teeth"). The loop ends with 'RLHF oracle -> repeat'; this
 * module makes the oracle real:
 *
 *   - preference pairs generated FROM the hive cells that just
 *     mutated (accepted vs rejected = preferred vs dispreferred)
 *   - a small ONLINE preference model: a per-cell survival score
 *     updated by the pairwise comparisons (the Bradley-Terry update)
 *   - credit assignment flows back into cell fitness: successful
 *     mutation lineages get higher survival probability
 *
 * Pure C11, opaque, no third party. The preference model is another
 * specialized cell type in the colony (a thin cell that learns which
 * mutation lineages survive).
 */
#ifndef WUBU_PREF_H
#define WUBU_PREF_H

#include <stdint.h>
#include <stddef.h>

/* one preference pair: the accepted (preferred) vs the rejected
 * (dispreferred) hive outcome. The pairs come from the diagnosis
 * archive (the ledger vs the graveyard). */
typedef struct {
    uint64_t batch;          /* provenance */
    float    win_fitness;    /* the preferred outcome's fitness */
    float    lose_fitness;   /* the dispreferred outcome's fitness */
    uint8_t  cell_idx;       /* which cell the pair concerns */
    uint8_t  lens;           /* the specialist lens (0xFF = whole colony) */
    uint8_t  won;            /* 1 = this cell's lineage WON (fitness
                                improved), 0 = it LOST */
} wubu_pref_pair_t;

/* the online preference model state */
typedef struct {
    /* the per-cell survival scores (the Bradley-Terry logits) */
    float *survival;          /* [n_cells] */
    int    n_cells;
    /* the pairing history (the recent pairs) */
    wubu_pref_pair_t *pairs;
    int    pairs_n, pairs_cap;
    /* the learning rate (the oracle's teeth) */
    float  lr;
    /* telemetry */
    uint64_t updates;         /* preference updates applied */
    uint64_t flips;           /* times the oracle changed a survival */
} wubu_pref_t;

/* P1: init the preference model. n_cells = the colony size. */
int wubu_pref_init(wubu_pref_t *pf, int n_cells, int pairs_cap, float lr);

/* P2: build a preference pair from a mutation outcome (the accepted
 * mutation is preferred over the rejected one — from the SAME hive
 * lineage). won = 1 if this cell's lineage improved (the accepted
 * side), 0 if it lost (the rejected side). Returns 0 on success. */
int wubu_pref_pair_from_outcome(wubu_pref_t *pf, uint64_t batch,
                                float win_fitness, float lose_fitness,
                                uint8_t cell_idx, uint8_t lens,
                                uint8_t won);

/* P3: apply the preference update (the Bradley-Terry logit update):
 * the winner's survival += lr*(1-p), the loser's survival -= lr*p
 * with p = sigmoid(win - lose). This is the credit assignment: the
 * successful lineage's cells get higher survival probability. */
int wubu_pref_update(wubu_pref_t *pf);

/* P4: read a cell's survival score (the mutation gate uses this:
 * mutations with higher lineage survival are more likely to pass). */
float wubu_pref_survival(const wubu_pref_t *pf, int cell_idx);

/* P5: the oracle telemetry. */
void wubu_pref_stats(const wubu_pref_t *pf, char *buf, size_t cap);

/* P6: free. */
void wubu_pref_free(wubu_pref_t *pf);

#endif
