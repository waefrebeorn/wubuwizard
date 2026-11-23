/* wubu_ev_depth.c -- EG-87 expected-value speculative depth gate.
 * Strict C11, opaque struct, status-as-value, no third-party deps. */
#include "wubu_ev_depth.h"

#include <stdlib.h>

#define EV_ALPHA          0.12f
#define EV_SAFETY         0.10f
#define EV_FLOOR          0.18f
#define EV_WARMUP_CYCLES  4
#define EV_PROBE_PERIOD   32

struct wubu_ev_depth {
    int max_depth;
    float baseline_tok_s;
    float extra_cost_s;
    float ewma[WUBU_EV_DEPTH_MAX];
    float confidence;
    float observed_extra_s;
    int cycles;
    int cost_samples;
};

wubu_ev_depth_t *wubu_ev_depth_create(int max_depth,
                                      float baseline_tok_s,
                                      float extra_cost_s)
{
    static const float priors[WUBU_EV_DEPTH_MAX] = {0.92f, 0.64f, 0.32f, 0.16f};
    wubu_ev_depth_t *p;
    if (max_depth < 1 || max_depth > WUBU_EV_DEPTH_MAX ||
        !(baseline_tok_s > 0.0f) || extra_cost_s < 0.0f) return NULL;
    p = (wubu_ev_depth_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->max_depth = max_depth;
    p->baseline_tok_s = baseline_tok_s;
    p->extra_cost_s = extra_cost_s;
    p->confidence = 1.0f;
    for (int d = 0; d < WUBU_EV_DEPTH_MAX; ++d) p->ewma[d] = priors[d];
    return p;
}

void wubu_ev_depth_free(wubu_ev_depth_t *p) { free(p); }

int wubu_ev_depth_observe(wubu_ev_depth_t *p, int accepted,
                          float baseline_s, float draft_verify_s)
{
    if (!p || accepted < 0 || accepted > p->max_depth ||
        baseline_s < 0.0f || draft_verify_s < 0.0f) return -1;
    /* accepted=k means depths 1..k succeeded, depth k+1 rejected. */
    for (int d = 0; d < p->max_depth; ++d) {
        float observed = (accepted > d) ? 1.0f : 0.0f;
        p->ewma[d] += EV_ALPHA * (observed - p->ewma[d]);
    }
    if (baseline_s > 0.0f && draft_verify_s > 0.0f) {
        float extra = draft_verify_s - baseline_s;
        if (extra < 0.0f) extra = 0.0f;
        p->observed_extra_s = p->cost_samples ?
            p->observed_extra_s + EV_ALPHA * (extra - p->observed_extra_s) : extra;
        p->cost_samples++;
    }
    p->cycles++;
    return 0;
}

float wubu_ev_depth_extra_prob(const wubu_ev_depth_t *p, int d)
{
    float prefix = 1.0f;
    if (!p || d < 1 || d > p->max_depth) return -1.0f;
    for (int i = 0; i < d; ++i) prefix *= p->ewma[i];
    return prefix;
}

int wubu_ev_depth_decide(const wubu_ev_depth_t *p)
{
    float threshold, prefix = 1.0f;
    int best_depth = 0;
    if (!p) return -1;
    if (p->cycles < EV_WARMUP_CYCLES) return p->max_depth;
    if (p->cycles > 0 && p->cycles % EV_PROBE_PERIOD == 0) return p->max_depth;
    threshold = (p->cost_samples >= 4 ? p->observed_extra_s : p->extra_cost_s)
                * p->baseline_tok_s * (1.0f + EV_SAFETY);
    for (int d = 1; d <= p->max_depth; ++d) {
        float ev;
        prefix *= p->ewma[d - 1];
        ev = prefix * p->confidence;
        if (ev > threshold && ev >= EV_FLOOR && d > best_depth) best_depth = d;
    }
    return best_depth;
}

void wubu_ev_depth_set_confidence(wubu_ev_depth_t *p, float factor)
{
    if (!p) return;
    p->confidence = factor < 0.0f ? 0.0f : (factor > 1.0f ? 1.0f : factor);
}

int wubu_ev_depth_cost_source(const wubu_ev_depth_t *p)
{
    if (!p) return -1;
    return p->cost_samples >= 4 ? 1 : 0;
}
