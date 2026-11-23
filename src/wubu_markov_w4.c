/*
 * wubu_markov_w4.c -- EG-104 W=4 Markov beam-tree branch selection. Pure C11.
 */
#include "wubu_markov_w4.h"

#include <stdlib.h>

static size_t ctx_count(int vocab) {
    return (size_t)vocab * (size_t)vocab * (size_t)vocab;
}

static int ctx_index(const int *hist, int hlen, int vocab) {
    int a = hist[hlen - 3], b = hist[hlen - 2], c = hist[hlen - 1];
    return (a * vocab + b) * vocab + c;
}

int wubu_markov_w4_init(wubu_markov_w4_t *m, int vocab) {
    size_t n;
    if (!m || vocab <= 1 || vocab > 64) return -1;
    n = ctx_count(vocab) * (size_t)vocab;
    m->counts = (unsigned long long *)calloc(n, sizeof(unsigned long long));
    if (!m->counts) return -1;
    m->vocab = vocab;
    return 0;
}

int wubu_markov_w4_free(wubu_markov_w4_t *m) {
    if (!m) return -1;
    free(m->counts);
    m->counts = NULL;
    m->vocab = 0;
    return 0;
}

int wubu_markov_w4_observe(wubu_markov_w4_t *m, const int *seq, int n) {
    if (!m || !m->counts || !seq || n < WUBU_MARKOV_W) return -1;
    for (int i = WUBU_MARKOV_W - 1; i < n; i++) {
        int ctx, tok;
        if (seq[i - 3] < 0 || seq[i - 3] >= m->vocab ||
            seq[i - 2] < 0 || seq[i - 2] >= m->vocab ||
            seq[i - 1] < 0 || seq[i - 1] >= m->vocab ||
            seq[i] < 0 || seq[i] >= m->vocab) return -1;
        ctx = ctx_index(seq + i - 3, 3, m->vocab);
        tok = seq[i];
        m->counts[(size_t)ctx * (size_t)m->vocab + (size_t)tok]++;
    }
    return 0;
}

double wubu_markov_w4_prob(const wubu_markov_w4_t *m,
                           const int *hist, int hlen, int token) {
    int ctx;
    unsigned long long total = 0, hit;
    if (!m || !m->counts || !hist || hlen < WUBU_MARKOV_W - 1 ||
        token < 0 || token >= m->vocab) return -1.0;
    ctx = ctx_index(hist, hlen, m->vocab);
    for (int v = 0; v < m->vocab; v++)
        total += m->counts[(size_t)ctx * (size_t)m->vocab + (size_t)v];
    if (total == 0) return 1.0 / (double)m->vocab;
    hit = m->counts[(size_t)ctx * (size_t)m->vocab + (size_t)token];
    return (double)hit / (double)total;
}

int wubu_markov_w4_select(const wubu_markov_w4_t *m,
                          const int *hist, int hlen,
                          const int *cand, int ncand,
                          int width, int *out, int cap) {
    int pick[WUBU_MARKOV_W], npick = 0;
    double score[WUBU_MARKOV_W];
    if (!m || !m->counts || !hist || !cand || !out ||
        hlen < WUBU_MARKOV_W - 1 || ncand <= 0 ||
        width <= 0 || width > WUBU_MARKOV_W || cap < width) return -1;
    for (int i = 0; i < ncand; i++) {
        double p = wubu_markov_w4_prob(m, hist, hlen, cand[i]);
        int j;
        if (p < 0.0) return -1;
        if (npick == width && p <= score[npick - 1]) continue;
        if (npick < width) npick++;
        j = npick - 1;
        while (j > 0 && p > score[j - 1]) {
            score[j] = score[j - 1];
            pick[j] = pick[j - 1];
            j--;
        }
        score[j] = p;
        pick[j] = i;
    }
    for (int i = 0; i < npick; i++) out[i] = pick[i];
    return npick;
}
