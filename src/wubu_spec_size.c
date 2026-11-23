/*
 * wubu_spec_size.c -- EG-25a Sequoia-style sizing. Pure C11.
 */
#include "wubu_spec_size.h"

#include <math.h>

int wubu_spec_size(int kmax, double accept_p,
                   double draft_cost, double verify_cost,
                   wubu_spec_size_t *out) {
    int k, best = 1;
    double best_rate = -1.0, best_expected = 0.0;
    if (kmax < 1 || !out || !(accept_p > 0.0 && accept_p <= 1.0) ||
        draft_cost < 0.0 || verify_cost <= 0.0) return -1;

    for (k = 1; k <= kmax; k++) {
        /* Truncated geometric: 1 + p + ... + p^(k-1). */
        double expected = accept_p >= 1.0
            ? (double)k
            : (1.0 - pow(accept_p, k)) / (1.0 - accept_p);
        double time = k * draft_cost + verify_cost;
        double rate = expected / time;
        if (rate > best_rate) {
            best_rate = rate;
            best = k;
            best_expected = expected;
        }
    }
    out->k = best;
    out->expected = best_expected;
    out->throughput = best_rate;
    return 0;
}
