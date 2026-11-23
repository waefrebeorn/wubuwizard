/* wubu_accept_table.c -- EG-94 empirical acceptance table. Strict C11. */
#include "wubu_accept_table.h"

#include <stdlib.h>
#include <math.h>

struct wubu_accept_table {
    unsigned int acc[WUBU_ACCEPT_SLOTS][WUBU_ACCEPT_RANKS][WUBU_ACCEPT_BINS];
    unsigned int tot[WUBU_ACCEPT_SLOTS][WUBU_ACCEPT_RANKS][WUBU_ACCEPT_BINS];
};

static int cell(double p, int *slot, int *rank, int *bin)
{
    if (*slot < 1 || *slot > WUBU_ACCEPT_SLOTS || *rank < 0 ||
        !(p >= 0.0 && p <= 1.0)) return -1;
    if (*rank >= WUBU_ACCEPT_RANKS) *rank = WUBU_ACCEPT_RANKS - 1;
    *bin = (int)(p * WUBU_ACCEPT_BINS);
    if (*bin >= WUBU_ACCEPT_BINS) *bin = WUBU_ACCEPT_BINS - 1;
    return 0;
}

wubu_accept_table_t *wubu_accept_table_create(void)
{
    return (wubu_accept_table_t *)calloc(1, sizeof(wubu_accept_table_t));
}

void wubu_accept_table_free(wubu_accept_table_t *p) { free(p); }

int wubu_accept_table_observe(wubu_accept_table_t *p,
                              int slot, int rank, double prob, int accepted)
{
    int bin;
    if (!p || (accepted != 0 && accepted != 1)) return -1;
    if (cell(prob, &slot, &rank, &bin) != 0) return -1;
    --slot;
    ++p->tot[slot][rank][bin];
    if (accepted) ++p->acc[slot][rank][bin];
    return 0;
}

double wubu_accept_table_logq(const wubu_accept_table_t *p,
                              int slot, int rank, double p_draft)
{
    int bin;
    unsigned int n, a;
    double prob;
    if (!p || cell(p_draft, &slot, &rank, &bin) != 0) return 0.0;
    --slot;
    n = p->tot[slot][rank][bin];
    if (n == 0) return 0.0;                 /* neutral prior for unseen cells */
    a = p->acc[slot][rank][bin];
    prob = ((double)a + 5.0 * 0.5) / ((double)n + 5.0);
    return log(prob);
}

int wubu_accept_table_best(const wubu_accept_table_t *p,
                           const int *slots, const int *ranks,
                           const double *probs, int n)
{
    int best = -1;
    double best_score = 0.0;
    if (!p || !slots || !ranks || !probs || n <= 0) return -1;
    for (int i = 0; i < n; ++i) {
        double pv = probs[i];
        double score;
        if (!(pv > 0.0) || pv > 1.0) continue;
        score = WUBU_ACCEPT_BLEND * log(pv) +
                (1.0 - WUBU_ACCEPT_BLEND) *
                wubu_accept_table_logq(p, slots[i], ranks[i], pv);
        if (best < 0 || score > best_score) {
            best = i;
            best_score = score;
        }
    }
    return best;
}
