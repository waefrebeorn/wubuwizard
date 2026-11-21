#ifndef WUBU_SPEC_SEED_H
#define WUBU_SPEC_SEED_H

/* wubu_spec_seed.h -- LP-08: deterministic seeded rejection sampling.
 * C11, status-as-value, no third-party deps. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Draw up to n_out accepted token ids from a categorical distribution using
 * xorshift64* plus rejection sampling.
 *   probs   : n floats in [0,1], acceptance probability for candidate id.
 *   seed    : PRNG seed; 0 is replaced by a fixed nonzero constant.
 *   out_ids : receives accepted ids, each in [0,n).
 * Returns the number of accepted ids (0..n_out), or -1 on invalid input.
 * Fully deterministic: identical (probs,n,seed,n_out) yields identical output;
 * a shorter draw is an exact prefix of a longer one with the same seed. */
int wubu_spec_seed_accept(const float *probs, int n, uint64_t seed,
                          int *out_ids, int n_out);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_SPEC_SEED_H */
