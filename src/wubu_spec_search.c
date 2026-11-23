/*
 * wubu_spec_search.c -- EG-41 speculative config search. Pure C11.
 */
#include "wubu_spec_search.h"

#include <math.h>

int wubu_spec_search(const wubu_spec_candidate_t *cands, int n,
                     double draft_cost, double verify_cost,
                     size_t vram_free, wubu_spec_choice_t *out) {
    int feasible = 0, best = -1;
    double best_rate = -1.0;
    if (!cands || !out || n <= 0 || draft_cost < 0.0 ||
        verify_cost <= 0.0) return -1;

    out->best_id = -1;
    out->n_feasible = 0;
    out->throughput = 0.0;

    for (int i = 0; i < n; i++) {
        const wubu_spec_candidate_t *c = &cands[i];
        double expected, time, rate;
        if (c->k <= 0 || c->heads <= 0 || c->draft_bytes > vram_free ||
            !(c->accept_p > 0.0f && c->accept_p <= 1.0f)) continue;
        feasible++;
        if (c->accept_p >= 1.0f) expected = (double)c->k;
        else expected = (1.0 - pow((double)c->accept_p, c->k)) /
                        (1.0 - (double)c->accept_p);
        time = c->k * draft_cost + verify_cost;
        rate = expected / time;
        if (rate > best_rate) {
            best_rate = rate;
            best = c->id;
        }
    }
    out->n_feasible = feasible;
    if (best >= 0) {
        out->best_id = best;
        out->throughput = best_rate;
    }
    return 0;
}
