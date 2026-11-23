/* wubu_prefetch_feedback.c -- EG-95 closed-loop prefetch depth gate. C11. */
#include "wubu_prefetch_feedback.h"

#include <stdlib.h>

#define LOW_USEFUL     0.60
#define HIGH_USEFUL    0.90
#define REGRESSION     0.05
#define PROBE_TOL      0.005
#define FREE_MIN       0.25
#define STABLE_WINDOWS 2
#define COOLDOWN       2

struct wubu_prefetch_feedback {
    int depth;
    int stable;
    int cooldown;
    int probing;
    int probe_from;
    int rejections;
    int rejected[WUBU_PREFETCH_MAX_DEPTH + 1];
    double ref_latency_us;
    double phase_latency_us;
};

wubu_prefetch_feedback_t *wubu_prefetch_feedback_create(int initial_depth)
{
    wubu_prefetch_feedback_t *p;
    if (initial_depth < WUBU_PREFETCH_MIN_DEPTH ||
        initial_depth > WUBU_PREFETCH_MAX_DEPTH) return NULL;
    p = (wubu_prefetch_feedback_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->depth = initial_depth;
    return p;
}

void wubu_prefetch_feedback_free(wubu_prefetch_feedback_t *p) { free(p); }

int wubu_prefetch_feedback_observe(wubu_prefetch_feedback_t *p,
                                   int consumed, int useful, int late,
                                   int evictions, int budget_skips,
                                   double latency_us,
                                   double peak_inflight_bytes,
                                   double inflight_budget_bytes)
{
    double useful_ratio;
    if (!p || consumed < 0 || useful < 0 || useful > consumed || late < 0 ||
        evictions < 0 || budget_skips < 0 || !(latency_us > 0.0) ||
        peak_inflight_bytes < 0.0 || !(inflight_budget_bytes > 0.0)) return -1;
    useful_ratio = consumed ? (double)useful / (double)consumed : 1.0;

    /* Resolve a pending bounded probe against its own reference latency. */
    if (p->probing) {
        if (latency_us > p->ref_latency_us * (1.0 + PROBE_TOL)) {
            p->depth = p->probe_from;
            p->rejected[p->probe_from] = 1;
            p->rejections++;
            p->cooldown = COOLDOWN;
            p->phase_latency_us = latency_us; /* new baseline, not a phase jump */
        }
        p->probing = 0;
        p->stable = 0;
        p->ref_latency_us = latency_us;
        return 0;
    }

    /* >5% baseline shift reopens remembered transitions for this window. */
    if (p->phase_latency_us > 0.0 &&
        (latency_us > p->phase_latency_us * (1.0 + REGRESSION) ||
         latency_us < p->phase_latency_us * (1.0 - REGRESSION))) {
        for (int i = 0; i <= WUBU_PREFETCH_MAX_DEPTH; ++i) p->rejected[i] = 0;
        p->phase_latency_us = latency_us;
        p->ref_latency_us = latency_us;
        p->stable = 0;
        p->cooldown = 0;
        return 0;
    }
    if (p->phase_latency_us == 0.0) p->phase_latency_us = latency_us;

    /* Drop on regression, budget skips, low usefulness or unexplained eviction. */
    if (p->ref_latency_us > 0.0 && latency_us > p->ref_latency_us * (1.0 + REGRESSION))
        p->depth = p->depth > WUBU_PREFETCH_MIN_DEPTH ? p->depth - 1 : p->depth;
    if (budget_skips > 0 || useful_ratio < LOW_USEFUL ||
        (evictions > 0 && useful_ratio < HIGH_USEFUL))
        p->depth = p->depth > WUBU_PREFETCH_MIN_DEPTH ? p->depth - 1 : p->depth;

    /* Rise only after late arrivals, two stable useful windows, free budget. */
    if (p->cooldown > 0) --p->cooldown;
    if (useful_ratio >= HIGH_USEFUL && budget_skips == 0 && evictions == 0 &&
        (p->ref_latency_us <= 0.0 ||
         latency_us <= p->ref_latency_us * (1.0 + REGRESSION)))
        ++p->stable;
    else
        p->stable = 0;
    if (late > 0 && p->stable >= STABLE_WINDOWS && p->cooldown == 0 &&
        p->depth < WUBU_PREFETCH_MAX_DEPTH && !p->rejected[p->depth] &&
        (inflight_budget_bytes - peak_inflight_bytes) / inflight_budget_bytes >= FREE_MIN) {
        p->probe_from = p->depth;
        ++p->depth;
        p->probing = 1;
        p->stable = 0;
    }
    p->ref_latency_us = latency_us;
    return 0;
}

int wubu_prefetch_feedback_depth(const wubu_prefetch_feedback_t *p)
{
    return p ? p->depth : -1;
}

int wubu_prefetch_feedback_probe_active(const wubu_prefetch_feedback_t *p)
{
    return p ? p->probing : -1;
}

int wubu_prefetch_feedback_rejections(const wubu_prefetch_feedback_t *p)
{
    return p ? p->rejections : -1;
}

int wubu_prefetch_feedback_rejected(const wubu_prefetch_feedback_t *p, int from_depth)
{
    if (!p || from_depth < WUBU_PREFETCH_MIN_DEPTH ||
        from_depth > WUBU_PREFETCH_MAX_DEPTH) return -1;
    return p->rejected[from_depth];
}
