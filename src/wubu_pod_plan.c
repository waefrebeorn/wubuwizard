/* wubu_pod_plan.c -- EG-83 POD split-K plan. Strict C11. */
#include "wubu_pod_plan.h"

#include <stdlib.h>

struct wubu_pod_plan {
    int heads, splits, dim, aux_allocations, layers_bound;
};

wubu_pod_plan_t *wubu_pod_plan_create(int heads, int splits, int dim)
{
    wubu_pod_plan_t *p;
    if (heads < 1 || splits < 1 || dim < 1) return NULL;
    p = (wubu_pod_plan_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->heads = heads;
    p->splits = splits;
    p->dim = dim;
    p->aux_allocations = 1;         /* one aux buffer per plan, not per layer */
    return p;
}

void wubu_pod_plan_free(wubu_pod_plan_t *p) { free(p); }

size_t wubu_pod_plan_workspace_bytes(const wubu_pod_plan_t *p)
{
    if (!p) return 0;
    return (size_t)p->splits * (size_t)p->heads * (size_t)p->dim * sizeof(float);
}

size_t wubu_pod_plan_aux_bytes(const wubu_pod_plan_t *p)
{
    return p ? (size_t)p->splits * sizeof(unsigned int) : 0;
}

int wubu_pod_plan_split(const wubu_pod_plan_t *p, const float *kv, int n,
                        float *partials)
{
    int width;
    if (!p || !kv || !partials || n < p->splits) return -1;
    width = p->heads * p->dim;
    for (int s = 0; s < p->splits; ++s)
        for (int j = 0; j < width; ++j) partials[(size_t)s * width + j] = 0.0f;
    for (int s = 0; s < p->splits; ++s) {
        int lo = s * n / p->splits, hi = (s + 1) * n / p->splits;
        for (int i = lo; i < hi; ++i)
            for (int j = 0; j < width; ++j)
                partials[(size_t)s * width + j] += kv[(size_t)i * width + j];
    }
    return p->splits;
}

int wubu_pod_plan_reduce(const wubu_pod_plan_t *p,
                         const float *partials, float *out)
{
    int width;
    if (!p || !partials || !out) return -1;
    width = p->heads * p->dim;
    for (int j = 0; j < width; ++j) {
        float v = 0.0f;
        for (int s = 0; s < p->splits; ++s) v += partials[(size_t)s * width + j];
        out[j] = v;
    }
    return 0;
}

int wubu_pod_plan_aux_allocations(const wubu_pod_plan_t *p)
{
    return p ? p->aux_allocations : -1;
}

int wubu_pod_plan_bind_layer(wubu_pod_plan_t *p, int layer)
{
    if (!p || layer < 0) return -1;
    ++p->layers_bound;              /* rebind only; no aux allocation */
    return 0;
}

int wubu_pod_plan_layers_bound(const wubu_pod_plan_t *p)
{
    return p ? p->layers_bound : -1;
}
