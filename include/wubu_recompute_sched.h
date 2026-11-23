#ifndef WUBU_RECOMPUTE_SCHED_H
#define WUBU_RECOMPUTE_SCHED_H

/*
 * wubu_recompute_sched.h -- EG-19: KV restore-vs-recompute scheduler
 * (ALISA / KVPR / mzCache lineage), reimplemented in portable C11.
 *
 * For a prefix that must be reconstructed, compare the time to restore its
 * KV from the cold tier against the time to recompute it through the model.
 * Restoration is forced off when the KV will not fit in the device budget;
 * otherwise a margin lets the caller trade a slightly slower restore for
 * fewer recomputed tokens.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_recompute_sched wubu_recompute_sched_t;

typedef struct {
    double cold_bandwidth_bps;    /* cold-tier bandwidth, bytes/second */
    double token_layers_per_s;    /* recompute throughput, token-layers/second */
    size_t device_budget;         /* max KV bytes that can be resident */
    double restore_margin;        /* restore if restore_ms <= recompute_ms*margin */
} wubu_recompute_config_t;

typedef struct {
    int restore;                  /* 1 = restore KV, 0 = recompute */
    double restore_ms;
    double recompute_ms;
    size_t bytes_moved;
    int tokens_recomputed;
    int forced_by_budget;
} wubu_recompute_decision_t;

wubu_recompute_sched_t *wubu_recompute_sched_create(wubu_recompute_config_t cfg);
void wubu_recompute_sched_free(wubu_recompute_sched_t *s);

/* Plan one prefix reconstruction. Returns 0 or -1 on bad arguments. */
int wubu_recompute_plan(wubu_recompute_sched_t *s, int tokens, int layers,
                        size_t kv_bytes, wubu_recompute_decision_t *out);

/* Cumulative counters. Any argument may be NULL. */
void wubu_recompute_stats(const wubu_recompute_sched_t *s,
                          size_t *bytes_moved, int *tokens_recomputed,
                          int *restores, int *recomputes);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_RECOMPUTE_SCHED_H */
