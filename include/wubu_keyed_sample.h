#ifndef WUBU_KEYED_SAMPLE_H
#define WUBU_KEYED_SAMPLE_H

/*
 * wubu_keyed_sample.h -- EG-89: position-keyed exponential-race sampling.
 * SpecPilot k001: x[pos] = argmin_v E(seed, pos, v) / p[v], with a counter-based
 * Exp(1) field keyed by (seed, absolute position, vocab id). Tree shape cannot
 * change a committed token: the noise key is the position, not the tree slot.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_keyed_sample wubu_keyed_sample_t;

/* NULL only on allocation failure. */
wubu_keyed_sample_t *wubu_keyed_sample_create(uint64_t seed);
void wubu_keyed_sample_free(wubu_keyed_sample_t *p);

/* Argmin of E(seed,pos,v)/p[v] for v in [0,vocab); pos >= 0.
 * Returns -1 on bad input (NULL, negative pos, vocab <= 0, all-zero p). */
int wubu_keyed_sample(const wubu_keyed_sample_t *p, int pos,
                      const float *probs, int vocab);

/* Tree node helper: absolute position = prefix_len + depth. -1 on bad input. */
int wubu_keyed_sample_at(const wubu_keyed_sample_t *p,
                         int prefix_len, int depth,
                         const float *probs, int vocab);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KEYED_SAMPLE_H */
