#ifndef WUBU_POD_PLAN_H
#define WUBU_POD_PLAN_H

/*
 * wubu_pod_plan.h -- EG-83: POD split-K decode plan with reusable aux.
 * A float partial workspace holds split*heads*dim accumulators; a small per-
 * plan aux buffer (counters) is allocated once and rebound across layers
 * rather than per layer. Portability gate: no CUDA API is claimed.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_pod_plan wubu_pod_plan_t;

wubu_pod_plan_t *wubu_pod_plan_create(int heads, int splits, int dim);
void wubu_pod_plan_free(wubu_pod_plan_t *p);

size_t wubu_pod_plan_workspace_bytes(const wubu_pod_plan_t *p);
size_t wubu_pod_plan_aux_bytes(const wubu_pod_plan_t *p);

/* Fill partials[s*(heads*dim) + j] from kv[i*(heads*dim)+j].
 * Returns number of splits, -1 on bad input. */
int wubu_pod_plan_split(const wubu_pod_plan_t *p, const float *kv, int n,
                        float *partials);

/* Sum the split partials into out[heads*dim]. 0/-1. */
int wubu_pod_plan_reduce(const wubu_pod_plan_t *p,
                         const float *partials, float *out);

/* Rebind the same aux to another layer; returns 0/-1. */
int wubu_pod_plan_bind_layer(wubu_pod_plan_t *p, int layer);

int wubu_pod_plan_aux_allocations(const wubu_pod_plan_t *p);
int wubu_pod_plan_layers_bound(const wubu_pod_plan_t *p);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_POD_PLAN_H */
