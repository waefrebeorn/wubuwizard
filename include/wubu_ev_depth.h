#ifndef WUBU_EV_DEPTH_H
#define WUBU_EV_DEPTH_H

/*
 * wubu_ev_depth.h -- EG-87: expected-value speculative depth gate.
 * MTPLX ExpectedValueDepthPolicy (adaptive.py:69-236), reimplemented in C11.
 *
 * Per-depth acceptance EWMAs feed expected_extra_accept = product(EWMA[0..d-1])
 * * EWMA[d-1] * confidence. Continue to depth d only while that probability
 * exceeds extra_cost_s * baseline_tok_s * (1 + safety_margin), with a floor
 * of 0.18. Four full-depth warmup cycles and a 32-cycle max-depth probe keep
 * the controller from starving itself of evidence. No weights, no CUDA.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_EV_DEPTH_MAX 4

typedef struct wubu_ev_depth wubu_ev_depth_t;

/* max_depth in [1,4]; baseline_tok_s > 0; extra_cost_s >= 0.
 * NULL on invalid input. Priors: .92/.64/.32/.16 (depth 4 extrapolated). */
wubu_ev_depth_t *wubu_ev_depth_create(int max_depth,
                                      float baseline_tok_s,
                                      float extra_cost_s);
void wubu_ev_depth_free(wubu_ev_depth_t *p);

/* Record one cycle. accepted in [0,max_depth] is the number of consecutive
 * draft tokens accepted; times are seconds (0 disables observed-cost update).
 * Returns 0 on success, -1 on invalid input. */
int wubu_ev_depth_observe(wubu_ev_depth_t *p, int accepted,
                          float baseline_s, float draft_verify_s);

/* Chosen depth in [0,max_depth]; 0 disables speculation. -1 on bad input. */
int wubu_ev_depth_decide(const wubu_ev_depth_t *p);

/* Probability of accepting d consecutive draft tokens at current EWMAs.
 * d in [1,max_depth]; -1.0f on bad input. */
float wubu_ev_depth_extra_prob(const wubu_ev_depth_t *p, int d);

/* Scale on the expected probability (e.g. draft confidence), clamped [0,1]. */
void wubu_ev_depth_set_confidence(wubu_ev_depth_t *p, float factor);

/* 0 = configured priors/cost, 1 = observed draft+verify cost samples. */
int wubu_ev_depth_cost_source(const wubu_ev_depth_t *p);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_EV_DEPTH_H */
