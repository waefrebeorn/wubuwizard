#ifndef WUBU_SPEC_BUDGET_H
#define WUBU_SPEC_BUDGET_H

/* wubu_spec_budget.h -- LP-07: speculative verification-budget planner.
 * C11, opaque planner, status-as-value, no third-party deps.
 *
 * A planner ranks draft sequences by the cumulative product of their
 * per-step acceptance confidences and distributes a fixed token budget
 * longest-first: the highest-confidence sequences receive the longest
 * verification windows, each clamped to the draft depth. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_spec_budget wubu_spec_budget_t;

/* Heap planner. Holds reusable scratch so plan() need not allocate in the
 * steady state. Returns NULL on allocation failure. */
wubu_spec_budget_t *wubu_spec_budget_create(void);
void wubu_spec_budget_free(wubu_spec_budget_t *p);

/* Distribute exactly query_budget verification tokens.
 *   confidences : n_seq*nextn floats, row-major [seq*nextn + step], each
 *                 in [0,1]; the cumulative product is P(whole draft accepted).
 *   out_lengths : n_seq ints, receives lengths in [0,nextn].
 * Returns 0 on success, -1 on invalid input. If query_budget exceeds the
 * total capacity n_seq*nextn it is clamped to that capacity (best effort).
 * Sequences are ranked by cumulative product descending, ties by ascending
 * index; longer windows go to better ranks. */
int wubu_spec_budget_plan(wubu_spec_budget_t *p,
                          const float *confidences, int n_seq, int nextn,
                          int query_budget, int *out_lengths);

/* Copy the ranking from the last successful plan() into out_order (best
 * first). cap must be >= n_seq. Returns 0 on success, -1 otherwise. */
int wubu_spec_budget_last_order(const wubu_spec_budget_t *p,
                                int *out_order, int cap);

/* Cumulative product for sequence s from the last successful plan().
 * Returns -1.0f if s is out of range or p is NULL. */
float wubu_spec_budget_cum(const wubu_spec_budget_t *p, int s);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_SPEC_BUDGET_H */
