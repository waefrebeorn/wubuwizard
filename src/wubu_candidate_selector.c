/* wubu_candidate_selector.c -- EG-74 p-min selector gate. Strict C11. */
#include "wubu_candidate_selector.h"

#include <stdlib.h>

struct wubu_candidate_selector {
    int max;
};

wubu_candidate_selector_t *wubu_candidate_selector_create(int max_candidates)
{
    wubu_candidate_selector_t *p;
    if (max_candidates < 1 || max_candidates > WUBU_SELECT_MAX) return NULL;
    p = (wubu_candidate_selector_t *)calloc(1, sizeof(*p));
    if (p) p->max = max_candidates;
    return p;
}

void wubu_candidate_selector_free(wubu_candidate_selector_t *p) { free(p); }

int wubu_candidate_selector_select(wubu_candidate_selector_t *p,
                                   const float *probs, const double *scores,
                                   int n, float p_min, int *out_idx)
{
    double key[WUBU_SELECT_MAX];
    int k = 0;
    if (!p || !probs || !out_idx || n < 1 || n > p->max ||
        !(p_min >= 0.0f && p_min <= 1.0f)) return -1;
    for (int i = 0; i < n; ++i) {
        if (!(probs[i] >= 0.0f && probs[i] <= 1.0f)) return -1;
        if (probs[i] < p_min) continue;
        out_idx[k] = i;
        key[k] = (double)probs[i] + (scores ? scores[i] : 0.0);
        ++k;
    }
    if (k == 0) {                 /* p-min never empties the candidate set */
        int best = 0;
        for (int i = 1; i < n; ++i)
            if (probs[i] > probs[best]) best = i;
        out_idx[0] = best;
        k = 1;
    }
    for (int i = 1; i < k; ++i) {
        int idx = out_idx[i];
        double v = key[i];
        int j = i - 1;
        while (j >= 0 && (key[j] < v || (key[j] == v && out_idx[j] > idx))) {
            key[j + 1] = key[j];
            out_idx[j + 1] = out_idx[j];
            --j;
        }
        key[j + 1] = v;
        out_idx[j + 1] = idx;
    }
    return k;
}
