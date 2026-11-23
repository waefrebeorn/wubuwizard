#ifndef WUBU_DSPARK_H
#define WUBU_DSPARK_H

/*
 * wubu_dspark.h -- EG-62: rank-limited Markov correction head, confidence head
 * predicting 1-TV, and per-position verification budget (DSpark lineage).
 * One gate: accepted length at equal verification budget.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* draft[v] = normalize(base[v] + sum_{k<rank} A[v,k] * B[k]). */
int wubu_dspark_correct(const float *base, const float *A, const float *B,
                        int vocab, int rank, float *draft);

/* Confidence head: 1 - total-variation distance between draft and target. */
int wubu_dspark_confidence(const float *draft, const float *target,
                           int vocab, float *conf);

/* Allocate a fixed total verification budget across positions, one unit
 * minimum each, favoring higher-confidence positions. Returns 0 or -1. */
int wubu_dspark_budget(const float *conf, int n, int total, int *budgets);

/* Expected accepted length under a per-position budget. */
double wubu_dspark_accepted(const float *conf, const int *budgets, int n);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_DSPARK_H */
