/* wubu_spectr_select.c -- EG-25e candidate mass selection. Strict C11. */
#include "wubu_spectr_select.h"

int wubu_spectr_select(const double *p_target, int n, int k, int *out)
{
    unsigned char taken[WUBU_SPECTR_MAX_V] = {0};
    if (!p_target || !out || n < 1 || n > WUBU_SPECTR_MAX_V ||
        k < 1 || k > n) return -1;
    for (int i = 0; i < k; ++i) {
        int best = -1;
        for (int j = 0; j < n; ++j) {
            if (taken[j]) continue;
            if (best < 0 || p_target[j] > p_target[best]) best = j;
        }
        taken[best] = 1;
        out[i] = best;
    }
    return k;
}

double wubu_spectr_transport_cost(const double *p_target,
                                  const int *selected, int k)
{
    double covered = 0.0;
    if (!p_target || !selected || k < 1) return -1.0;
    for (int i = 0; i < k; ++i) {
        int id = selected[i];
        if (id < 0 || id >= WUBU_SPECTR_MAX_V || p_target[id] < 0.0) return -1.0;
        covered += p_target[id];
    }
    return 1.0 - covered > 0.0 ? 1.0 - covered : 0.0;
}
