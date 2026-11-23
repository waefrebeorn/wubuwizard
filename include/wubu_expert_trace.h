#ifndef WUBU_EXPERT_TRACE_H
#define WUBU_EXPERT_TRACE_H

/*
 * wubu_expert_trace.h -- EG-20: compute-centric MoE expert activation trace
 * cache (MoE-Infinity iEAM/pEAM lineage), reimplemented in C11.
 *
 * Per-layer decayed activation scores predict which experts a coming token
 * will need. A small slot cache holds prefetched experts; on a miss the
 * lowest-score resident expert in that layer is evicted. Trace decay lets the
 * prediction follow a changed routing distribution without retraining.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_expert_trace wubu_expert_trace_t;

typedef struct {
    int hits, misses, loads, evictions;
    int decay_steps;
    size_t bytes_moved;
} wubu_expert_trace_stats_t;

wubu_expert_trace_t *wubu_expert_trace_create(int layers, int experts,
                                               int slots, int decay_period,
                                               size_t expert_bytes);
void wubu_expert_trace_free(wubu_expert_trace_t *t);

/* Record an observed routing decision. Returns 0 or -1 on bad ids. */
int wubu_expert_trace_observe(wubu_expert_trace_t *t, int layer, int expert);

/* Return the highest-score expert for a layer. Returns expert id or -1. */
int wubu_expert_trace_predict(const wubu_expert_trace_t *t, int layer);

/* Request an expert: 1 if cached, 0 if loaded on demand, -1 on bad id. */
int wubu_expert_trace_prefetch(wubu_expert_trace_t *t, int layer, int expert);

/* Advance the decay clock. Returns the number of decay steps applied. */
int wubu_expert_trace_tick(wubu_expert_trace_t *t);

void wubu_expert_trace_stats(const wubu_expert_trace_t *t,
                             wubu_expert_trace_stats_t *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_EXPERT_TRACE_H */
