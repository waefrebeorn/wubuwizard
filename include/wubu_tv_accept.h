#ifndef WUBU_TV_ACCEPT_H
#define WUBU_TV_ACCEPT_H

/*
 * wubu_tv_accept.h -- EG-42: TV-objective drafter identity,
 * accept(p,q) = 1 - TV(p,q) = sum_v min(p_v,q_v), with a deterministic
 * calibration sampler (TVP lineage). Pure C11.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* sum_v min(p[v],q[v]); both rows must sum to ~1. Returns -1 on bad input. */
float wubu_tv_accept_prob(const float *p, const float *q, int n);

/* 0.5 * sum_v |p[v]-q[v]|. */
float wubu_tv_distance(const float *p, const float *q, int n);

/* Empirical acceptance by speculative-sampling accept rule min(1,p/q). */
int wubu_tv_calibrate(const float *p, const float *q, int n,
                      unsigned seed, int trials, float *empirical);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_TV_ACCEPT_H */
