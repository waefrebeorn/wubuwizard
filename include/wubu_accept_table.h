#ifndef WUBU_ACCEPT_TABLE_H
#define WUBU_ACCEPT_TABLE_H

/*
 * wubu_accept_table.h -- EG-94: empirical acceptance table for tree ordering.
 * SpecPilot d3m: log P(candidate is the true next token) keyed by
 * (slot 1..10, sibling-rank bucket 0..6, drafter-probability bin 0..17),
 * fitted with prior weight 5 and blended 0.5 with log p for best-first order.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_ACCEPT_SLOTS 10
#define WUBU_ACCEPT_RANKS 7
#define WUBU_ACCEPT_BINS  18
#define WUBU_ACCEPT_BLEND 0.5

typedef struct wubu_accept_table wubu_accept_table_t;

wubu_accept_table_t *wubu_accept_table_create(void);
void wubu_accept_table_free(wubu_accept_table_t *p);

/* Add one verify-step observation. slot >= 1, rank >= 0, p in [0,1],
 * accepted in {0,1}. Returns 0 on success, -1 on bad input. */
int wubu_accept_table_observe(wubu_accept_table_t *p,
                              int slot, int rank, double prob, int accepted);

/* log empirical acceptance for the cell; 0.0 for an unobserved cell. */
double wubu_accept_table_logq(const wubu_accept_table_t *p,
                              int slot, int rank, double p_draft);

/* Best-first candidate index: score = BLEND*log(p) + (1-BLEND)*logq(cell).
 * Ties choose the lower index. Returns -1 on bad input. */
int wubu_accept_table_best(const wubu_accept_table_t *p,
                           const int *slots, const int *ranks,
                           const double *probs, int n);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_ACCEPT_TABLE_H */
