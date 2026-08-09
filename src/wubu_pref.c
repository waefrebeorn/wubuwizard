/*
 * wubu_pref.c -- the RLHF ORACLE (see wubu_pref.h). The Bradley-Terry
 * online preference model: pairs from the hive mutation outcomes,
 * per-cell survival scores, credit assignment back into fitness.
 */
#include "wubu_pref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int wubu_pref_init(wubu_pref_t *pf, int n_cells, int pairs_cap, float lr)
{
    if (!pf || n_cells <= 0) return -1;
    memset(pf, 0, sizeof(*pf));
    pf->n_cells = n_cells;
    pf->pairs_cap = pairs_cap > 0 ? pairs_cap : 64;
    pf->lr = lr > 0 ? lr : 0.05f;
    pf->survival = (float *)calloc((size_t)n_cells, sizeof(float));
    pf->pairs = (wubu_pref_pair_t *)calloc((size_t)pf->pairs_cap, sizeof(wubu_pref_pair_t));
    if (!pf->survival || !pf->pairs) {
        free(pf->survival); free(pf->pairs);
        memset(pf, 0, sizeof(*pf));
        return -1;
    }
    /* the survival scores start at 0.5 (indifferent) */
    for (int i = 0; i < n_cells; i++) pf->survival[i] = 0.5f;
    return 0;
}

int wubu_pref_pair_from_outcome(wubu_pref_t *pf, uint64_t batch,
                                float win_fitness, float lose_fitness,
                                uint8_t cell_idx, uint8_t lens,
                                uint8_t won)
{
    if (!pf) return -1;
    int slot = pf->pairs_n < pf->pairs_cap ? pf->pairs_n++
                                           : (int)(batch % (uint64_t)pf->pairs_cap);
    pf->pairs[slot].batch = batch;
    pf->pairs[slot].win_fitness = win_fitness;
    pf->pairs[slot].lose_fitness = lose_fitness;
    pf->pairs[slot].cell_idx = cell_idx;
    pf->pairs[slot].lens = lens;
    pf->pairs[slot].won = won;
    return 0;
}

int wubu_pref_update(wubu_pref_t *pf)
{
    if (!pf) return -1;
    /* the Bradley-Terry update over the recent pairs: the winner's
     * survival rises, the loser's falls, by the margin-scaled lr. */
    for (int i = 0; i < pf->pairs_n; i++) {
        wubu_pref_pair_t *p = &pf->pairs[i];
        int c = p->cell_idx;
        if (c >= pf->n_cells) c = pf->n_cells - 1;
        /* p(win) = sigmoid(win - lose) in the logit space (fitness is
         * the LOSS: lower is better, so the win has the LOWER value —
         * the margin is the better-vs-worse difference) */
        float margin = p->win_fitness - p->lose_fitness;
        float pwin = 1.0f / (1.0f + expf(-margin));
        float before = pf->survival[c];
        if (p->won) {
            /* the winner: survival rises by lr*(1-pwin) */
            pf->survival[c] += pf->lr * (1.0f - pwin);
        } else {
            /* the loser: survival falls by lr*pwin */
            pf->survival[c] -= pf->lr * pwin;
        }
        if (pf->survival[c] > 1.0f) pf->survival[c] = 1.0f;
        if (pf->survival[c] < 0.0f) pf->survival[c] = 0.0f;
        if (fabsf(pf->survival[c] - before) > 1e-6f) pf->flips++;
        pf->updates++;
    }
    return 0;
}

float wubu_pref_survival(const wubu_pref_t *pf, int cell_idx)
{
    if (!pf || cell_idx < 0 || cell_idx >= pf->n_cells) return 0.5f;
    return pf->survival[cell_idx];
}

void wubu_pref_stats(const wubu_pref_t *pf, char *buf, size_t cap)
{
    if (!pf || !buf || cap == 0) return;
    snprintf(buf, cap,
             "updates=%llu flips=%llu pairs=%d survival[0]=%.3f survival[last]=%.3f",
             (unsigned long long)pf->updates, (unsigned long long)pf->flips,
             pf->pairs_n,
             pf->n_cells > 0 ? pf->survival[0] : 0.5f,
             pf->n_cells > 0 ? pf->survival[pf->n_cells - 1] : 0.5f);
}

void wubu_pref_free(wubu_pref_t *pf)
{
    if (!pf) return;
    free(pf->survival);
    free(pf->pairs);
    memset(pf, 0, sizeof(*pf));
}
