#ifndef WUBU_RELAXED_VERIFY_H
#define WUBU_RELAXED_VERIFY_H

/*
 * wubu_relaxed_verify.h -- EG-25d: relaxed speculative verification policy.
 * Exact verification accepts a drafted token when the target probability is
 * at least the draft probability. Relaxed verification admits a ratio
 * shortfall of delta: q >= p * (1 - delta). The gate measures the extra
 * accepted prefix and flags it as a policy trade, not a distribution claim.
 */

#ifdef __cplusplus
extern "C" {
#endif

int wubu_relaxed_accept(double p_draft, double p_target, double delta);

/* Fill flags[0..n) with 1 while the chain is accepted, then 0. 0/-1. */
int wubu_relaxed_verify_chain(const double *p_draft, const double *p_target,
                              int n, double delta, int *flags);

int wubu_relaxed_accepted(const int *flags, int n);
int wubu_relaxed_extra(const int *exact, const int *relaxed, int n);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_RELAXED_VERIFY_H */
