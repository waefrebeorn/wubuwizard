/*
 * wubu_lineage.c -- LINEAGE-AWARE FITNESS + EXTINCTION PRESSURE
 * (see wubu_lineage.h). Evolutionary pressure with memory.
 */
#include "wubu_lineage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int wubu_lineage_init(wubu_lineage_tracker_t *lt, int cap,
                      int extinction_window, float extinction_rate)
{
    if (!lt || cap <= 0) return -1;
    memset(lt, 0, sizeof(*lt));
    lt->cap = cap;
    lt->extinction_window = extinction_window > 0 ? extinction_window : 10;
    lt->extinction_rate = extinction_rate > 0 ? extinction_rate : 0.1f;
    lt->lines = (wubu_lineage_t *)calloc((size_t)cap, sizeof(wubu_lineage_t));
    if (!lt->lines) return -1;
    lt->next_id = 1;
    return 0;
}

static wubu_lineage_t *find_line(wubu_lineage_tracker_t *lt, uint64_t id)
{
    for (int i = 0; i < lt->n; i++)
        if (lt->lines[i].lineage_id == id) return &lt->lines[i];
    return NULL;
}

static wubu_lineage_t *new_line(wubu_lineage_tracker_t *lt, uint64_t parent_id)
{
    if (lt->n >= lt->cap) return NULL;
    wubu_lineage_t *l = &lt->lines[lt->n++];
    memset(l, 0, sizeof(*l));
    l->lineage_id = lt->next_id++;
    l->parent_id = parent_id;
    return l;
}

uint64_t wubu_lineage_record(wubu_lineage_tracker_t *lt, uint64_t parent_id,
                             float fitness, float prev_fitness)
{
    if (!lt) return 0;
    wubu_lineage_t *l;
    if (parent_id == 0) {
        l = new_line(lt, 0);
        if (!l) return 0;
    } else {
        l = find_line(lt, parent_id);
        if (!l) { l = new_line(lt, parent_id); if (!l) return 0; }
    }
    l->generations++;
    l->fitness = fitness;
    if (fitness < l->best_fitness || l->best_fitness == 0.0f) {
        l->best_fitness = fitness;       /* lower loss = better */
        l->stagnant_for = 0;
        l->survival = l->survival * 0.7f + 0.3f;   /* the win bumps it */
        if (lt) lt->n_improvements++;
    } else {
        l->stagnant_for++;
        l->survival = l->survival * 0.9f;          /* stagnation decays */
    }
    if (l->survival > 1.0f) l->survival = 1.0f;
    if (l->survival < 0.0f) l->survival = 0.0f;
    return l->lineage_id;
}

int wubu_lineage_extinction_pass(wubu_lineage_tracker_t *lt)
{
    if (!lt) return 0;
    int extinct = 0;
    for (int i = 0; i < lt->n; i++) {
        wubu_lineage_t *l = &lt->lines[i];
        if (l->extinct) continue;
        if (l->stagnant_for >= lt->extinction_window) {
            /* soft extinction: progressive shrink probability —
             * the lineage does NOT die instantly, it fades */
            if ((float)l->stagnant_for * lt->extinction_rate
                    > 1.0f - l->survival) {
                l->extinct = 1;
                l->survival = 0.0f;
                extinct++;
                lt->n_extinctions++;
            }
        }
    }
    return extinct;
}

float wubu_lineage_survival(const wubu_lineage_tracker_t *lt, uint64_t lineage_id)
{
    if (!lt) return 0.0f;
    wubu_lineage_t *l = find_line((wubu_lineage_tracker_t *)lt, lineage_id);
    return l ? l->survival : 0.0f;
}

int wubu_lineage_nearmiss(const wubu_lineage_tracker_t *lt,
                          float winner_fitness, float tol,
                          uint64_t *out, int out_cap)
{
    if (!lt || !out || out_cap <= 0) return 0;
    int k = 0;
    for (int i = 0; i < lt->n && k < out_cap; i++) {
        wubu_lineage_t *l = &lt->lines[i];
        /* near-miss: close to the winner but did not beat it */
        if (l->best_fitness > winner_fitness &&
            l->best_fitness - winner_fitness <= tol) {
            out[k++] = l->lineage_id;
        }
    }
    return k;
}

void wubu_lineage_stats(const wubu_lineage_tracker_t *lt, char *buf, size_t cap)
{
    if (!lt || !buf || cap == 0) return;
    snprintf(buf, cap,
             "lineages=%d extinct=%llu improved=%llu window=%d rate=%.2f",
             lt->n,
             (unsigned long long)lt->n_extinctions,
             (unsigned long long)lt->n_improvements,
             lt->extinction_window, lt->extinction_rate);
}

void wubu_lineage_free(wubu_lineage_tracker_t *lt)
{
    if (!lt) return;
    free(lt->lines);
    memset(lt, 0, sizeof(*lt));
}
