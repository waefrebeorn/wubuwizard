#ifndef WUBU_PREFETCH_FEEDBACK_H
#define WUBU_PREFETCH_FEEDBACK_H

/*
 * wubu_prefetch_feedback.h -- EG-95: closed-loop prefetch depth, MemVanta H11.
 *
 * Per-window telemetry drives depth drops (>5% latency regression, budget
 * skips, usefulness < 0.60, unexplained evictions). Rises are bounded probes
 * gated on late arrivals, two stable highly-useful windows and >=25% free byte
 * budget; a probe that regresses by >0.5% is rejected with a cooldown and the
 * exact transition is remembered until a >5% baseline phase change.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_PREFETCH_MIN_DEPTH 1
#define WUBU_PREFETCH_MAX_DEPTH 4

typedef struct wubu_prefetch_feedback wubu_prefetch_feedback_t;

/* initial_depth in [1,4]. NULL on invalid input. */
wubu_prefetch_feedback_t *wubu_prefetch_feedback_create(int initial_depth);
void wubu_prefetch_feedback_free(wubu_prefetch_feedback_t *p);

/* Record one telemetry window. Returns 0 on success, -1 on bad input. */
int wubu_prefetch_feedback_observe(wubu_prefetch_feedback_t *p,
                                   int consumed, int useful, int late,
                                   int evictions, int budget_skips,
                                   double latency_us,
                                   double peak_inflight_bytes,
                                   double inflight_budget_bytes);

int wubu_prefetch_feedback_depth(const wubu_prefetch_feedback_t *p);
int wubu_prefetch_feedback_probe_active(const wubu_prefetch_feedback_t *p);
int wubu_prefetch_feedback_rejections(const wubu_prefetch_feedback_t *p);
int wubu_prefetch_feedback_rejected(const wubu_prefetch_feedback_t *p, int from_depth);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_PREFETCH_FEEDBACK_H */
