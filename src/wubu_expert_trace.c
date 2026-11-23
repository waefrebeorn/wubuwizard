/*
 * wubu_expert_trace.c -- EG-20 expert trace cache. Pure C11.
 */
#include "wubu_expert_trace.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    int layer, expert, used;
    unsigned score;   /* copied decayed score at load time */
} slot_t;

struct wubu_expert_trace {
    int layers, experts, slots, decay_period;
    int clock;
    size_t expert_bytes;
    unsigned *score;  /* [layers][experts], fixed-point x16 */
    slot_t *cache;
    wubu_expert_trace_stats_t st;
};

wubu_expert_trace_t *wubu_expert_trace_create(int layers, int experts,
                                               int slots, int decay_period,
                                               size_t expert_bytes) {
    wubu_expert_trace_t *t;
    if (layers <= 0 || experts <= 0 || slots <= 0 || decay_period <= 0) return NULL;
    t = (wubu_expert_trace_t *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->layers = layers; t->experts = experts;
    t->slots = slots; t->decay_period = decay_period;
    t->expert_bytes = expert_bytes;
    t->score = (unsigned *)calloc((size_t)layers * experts, sizeof(unsigned));
    t->cache = (slot_t *)calloc((size_t)slots, sizeof(slot_t));
    if (!t->score || !t->cache) {
        wubu_expert_trace_free(t);
        return NULL;
    }
    return t;
}

void wubu_expert_trace_free(wubu_expert_trace_t *t) {
    if (!t) return;
    free(t->score);
    free(t->cache);
    free(t);
}

static int valid(const wubu_expert_trace_t *t, int layer, int expert) {
    return t && layer >= 0 && layer < t->layers &&
           expert >= 0 && expert < t->experts;
}

int wubu_expert_trace_observe(wubu_expert_trace_t *t, int layer, int expert) {
    if (!valid(t, layer, expert)) return -1;
    t->score[(size_t)layer * t->experts + expert] += 16;
    return 0;
}

int wubu_expert_trace_predict(const wubu_expert_trace_t *t, int layer) {
    int e, best = 0;
    unsigned top;
    if (!t || layer < 0 || layer >= t->layers) return -1;
    top = t->score[(size_t)layer * t->experts];
    for (e = 1; e < t->experts; e++) {
        unsigned v = t->score[(size_t)layer * t->experts + e];
        if (v > top) { top = v; best = e; }
    }
    return best;
}

int wubu_expert_trace_prefetch(wubu_expert_trace_t *t, int layer, int expert) {
    int i, victim = 0;
    unsigned low;
    if (!valid(t, layer, expert)) return -1;
    for (i = 0; i < t->slots; i++)
        if (t->cache[i].used && t->cache[i].layer == layer &&
            t->cache[i].expert == expert) {
            t->st.hits++;
            return 1;
        }
    t->st.misses++;
    for (i = 0; i < t->slots; i++) if (!t->cache[i].used) break;
    if (i == t->slots) {
        low = t->cache[0].score;
        for (i = 1; i < t->slots; i++)
            if (t->cache[i].score < low) {
                low = t->cache[i].score;
                victim = i;
            }
        i = victim;
        t->st.evictions++;
    }
    t->cache[i].used = 1;
    t->cache[i].layer = layer;
    t->cache[i].expert = expert;
    t->cache[i].score = t->score[(size_t)layer * t->experts + expert];
    t->st.loads++;
    t->st.bytes_moved += t->expert_bytes;
    return 0;
}

int wubu_expert_trace_tick(wubu_expert_trace_t *t) {
    int i, steps = 0;
    if (!t) return -1;
    t->clock++;
    if (t->clock % t->decay_period) return 0;
    for (i = 0; i < t->layers * t->experts; i++) t->score[i] /= 2;
    t->st.decay_steps++;
    steps = 1;
    return steps;
}

void wubu_expert_trace_stats(const wubu_expert_trace_t *t,
                             wubu_expert_trace_stats_t *out) {
    if (!t || !out) return;
    *out = t->st;
}
