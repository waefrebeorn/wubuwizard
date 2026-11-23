#ifndef WUBU_EXPERT_SUB_H
#define WUBU_EXPERT_SUB_H

/*
 * wubu_expert_sub.h -- EG-78: constant-vector novice expert substitution.
 * Redundant experts keep one [hidden] approx vector instead of their MLP,
 * freeing MLP bytes while preserving logical expert ids and router weights.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_EXPERT_MAX 64
#define WUBU_HIDDEN_MAX 16

typedef struct wubu_expert_sub wubu_expert_sub_t;

/* experts in [1,64], hidden in [1,16], mlp_bytes >= 1. NULL on bad input. */
wubu_expert_sub_t *wubu_expert_sub_create(int experts, int hidden,
                                          size_t mlp_bytes);
void wubu_expert_sub_free(wubu_expert_sub_t *p);

/* Replace one expert with a constant vector; returns bytes freed. */
size_t wubu_expert_sub_mark(wubu_expert_sub_t *p, int expert_id,
                            const float *const_vector);

int wubu_expert_sub_is_substituted(const wubu_expert_sub_t *p, int expert_id);
size_t wubu_expert_sub_bytes_saved(const wubu_expert_sub_t *p);

/* Accumulate router_weight*const_vector for substituted experts in selected[];
 * normal experts contribute only their unchanged router weight. 0/-1. */
int wubu_expert_sub_contrib(const wubu_expert_sub_t *p,
                            const double *router_weights,
                            const int *selected, int k, float *out_hidden);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_EXPERT_SUB_H */
