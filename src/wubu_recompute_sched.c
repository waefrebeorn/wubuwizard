/*
 * wubu_recompute_sched.c -- EG-19 restore/recompute cost scheduler. Pure C11.
 */
#include "wubu_recompute_sched.h"

#include <stdlib.h>
#include <string.h>

struct wubu_recompute_sched {
    wubu_recompute_config_t cfg;
    size_t bytes_moved;
    int tokens_recomputed;
    int restores, recomputes;
};

wubu_recompute_sched_t *wubu_recompute_sched_create(wubu_recompute_config_t cfg) {
    wubu_recompute_sched_t *s;
    if (cfg.cold_bandwidth_bps <= 0.0 || cfg.token_layers_per_s <= 0.0 ||
        cfg.device_budget == 0 || cfg.restore_margin < 0.0) return NULL;
    s = (wubu_recompute_sched_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->cfg = cfg;
    return s;
}

void wubu_recompute_sched_free(wubu_recompute_sched_t *s) { free(s); }

int wubu_recompute_plan(wubu_recompute_sched_t *s, int tokens, int layers,
                        size_t kv_bytes, wubu_recompute_decision_t *out) {
    wubu_recompute_decision_t d;
    double restore_ms, recompute_ms;
    int forced = 0;
    if (!s || !out || tokens <= 0 || layers <= 0) return -1;

    restore_ms = (double)kv_bytes / s->cfg.cold_bandwidth_bps * 1000.0;
    recompute_ms = (double)tokens * (double)layers /
                   s->cfg.token_layers_per_s * 1000.0;

    if (kv_bytes > s->cfg.device_budget) {
        d.restore = 0;
        forced = 1;
    } else {
        d.restore = restore_ms <= recompute_ms * s->cfg.restore_margin;
    }
    d.restore_ms = restore_ms;
    d.recompute_ms = recompute_ms;
    d.bytes_moved = d.restore ? kv_bytes : 0;
    d.tokens_recomputed = d.restore ? 0 : tokens;
    d.forced_by_budget = forced;

    if (d.restore) {
        s->bytes_moved += kv_bytes;
        s->restores++;
    } else {
        s->tokens_recomputed += tokens;
        s->recomputes++;
    }
    *out = d;
    return 0;
}

void wubu_recompute_stats(const wubu_recompute_sched_t *s,
                          size_t *bytes_moved, int *tokens_recomputed,
                          int *restores, int *recomputes) {
    if (!s) return;
    if (bytes_moved) *bytes_moved = s->bytes_moved;
    if (tokens_recomputed) *tokens_recomputed = s->tokens_recomputed;
    if (restores) *restores = s->restores;
    if (recomputes) *recomputes = s->recomputes;
}
