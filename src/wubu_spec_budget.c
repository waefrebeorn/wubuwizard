/* wubu_spec_budget.c -- LP-07: speculative verification-budget planner.
 * Strict C11, opaque struct, status-as-value, no third-party deps.
 * See include/wubu_spec_budget.h. */
#include "wubu_spec_budget.h"
#include <stdlib.h>
#include <limits.h>

struct wubu_spec_budget {
    float *cum;    /* scratch: capacity entries */
    int   *order;  /* scratch: capacity entries */
    int    cap;    /* scratch capacity, in sequences */
    int    n;      /* sequences in the last successful plan */
};

static int ensure_scratch(wubu_spec_budget_t *p, int n)
{
    if (n <= p->cap) return 0;
    float *c = (float *)realloc(p->cum, (size_t)n * sizeof(*c));
    if (!c) return -1;
    p->cum = c;
    int *o = (int *)realloc(p->order, (size_t)n * sizeof(*o));
    if (!o) return -1; /* p->cum is valid and owned; cap stays unchanged */
    p->order = o;
    p->cap = n;
    return 0;
}

wubu_spec_budget_t *wubu_spec_budget_create(void)
{
    return (wubu_spec_budget_t *)calloc(1, sizeof(wubu_spec_budget_t));
}

void wubu_spec_budget_free(wubu_spec_budget_t *p)
{
    if (!p) return;
    free(p->cum);
    free(p->order);
    free(p);
}

int wubu_spec_budget_plan(wubu_spec_budget_t *p, const float *confidences,
                          int n_seq, int nextn, int query_budget,
                          int *out_lengths)
{
    if (!p || !confidences || !out_lengths || n_seq <= 0 || nextn <= 0 ||
        query_budget < 0) return -1;
    if (n_seq > INT_MAX / nextn) return -1; /* capacity overflow guard */

    int capacity = n_seq * nextn;
    if (query_budget > capacity) query_budget = capacity; /* best effort */
    if (ensure_scratch(p, n_seq) != 0) return -1;

    /* Cumulative product per sequence defines the acceptance of the chain. */
    for (int s = 0; s < n_seq; ++s) {
        double prod = 1.0;
        const float *row = confidences + (size_t)s * (size_t)nextn;
        for (int j = 0; j < nextn; ++j) {
            double c = (double)row[j];
            if (c < 0.0) c = 0.0;
            if (c > 1.0) c = 1.0;
            prod *= c;
        }
        p->cum[s] = (float)prod;
        p->order[s] = s;
    }

    /* Rank: cumulative confidence descending, tie-break index ascending. */
    for (int i = 1; i < n_seq; ++i) {
        int key = p->order[i];
        float kc = p->cum[key];
        int j = i - 1;
        while (j >= 0 && (p->cum[p->order[j]] < kc ||
                          (p->cum[p->order[j]] == kc && p->order[j] > key))) {
            p->order[j + 1] = p->order[j];
            --j;
        }
        p->order[j + 1] = key;
    }

    /* Allocate longest-first: each rank takes up to nextn, last takes the
     * remainder. Total is exactly query_budget (when within capacity). */
    int remaining = query_budget;
    for (int r = 0; r < n_seq; ++r) {
        int seq = p->order[r];
        int len = remaining > nextn ? nextn : remaining;
        out_lengths[seq] = len;
        remaining -= len;
    }
    p->n = n_seq;
    return 0;
}

int wubu_spec_budget_last_order(const wubu_spec_budget_t *p,
                                int *out_order, int cap)
{
    if (!p || !out_order || p->n <= 0 || cap < p->n) return -1;
    for (int i = 0; i < p->n; ++i) out_order[i] = p->order[i];
    return 0;
}

float wubu_spec_budget_cum(const wubu_spec_budget_t *p, int s)
{
    if (!p || s < 0 || s >= p->n) return -1.0f;
    return p->cum[s];
}
