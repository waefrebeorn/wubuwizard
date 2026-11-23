#ifndef WUBU_MARKOV_W4_H
#define WUBU_MARKOV_W4_H

/*
 * wubu_markov_w4.h -- EG-104: W=4 Markov beam-tree branch selection
 * (Strata v0.2.19 DSpark lineage), reimplemented in C11.
 *
 * Learns order-4 (3-token context) next-token transition counts from a token
 * stream and selects the highest-probability W=4 branches from a candidate
 * set. The selected branches feed an exact EG-21 tree verification pass.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_MARKOV_W 4

typedef struct {
    int vocab;                 /* 1 < vocab <= 64 */
    unsigned long long *counts;/* vocab^3 contexts * vocab next-token counts */
} wubu_markov_w4_t;

int wubu_markov_w4_init(wubu_markov_w4_t *m, int vocab);
int wubu_markov_w4_free(wubu_markov_w4_t *m);

/* Train on seq[0..n-1]; needs n >= W tokens. Returns 0 or -1. */
int wubu_markov_w4_observe(wubu_markov_w4_t *m, const int *seq, int n);

/* P(token | last W-1 history tokens), uniform backoff on unseen context. */
double wubu_markov_w4_prob(const wubu_markov_w4_t *m,
                           const int *hist, int hlen, int token);

/* Pick up to 'width' (<= W) highest-probability candidates, descending.
 * Writes candidate indices into out (capacity cap). Returns count or -1. */
int wubu_markov_w4_select(const wubu_markov_w4_t *m,
                          const int *hist, int hlen,
                          const int *cand, int ncand,
                          int width, int *out, int cap);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MARKOV_W4_H */
