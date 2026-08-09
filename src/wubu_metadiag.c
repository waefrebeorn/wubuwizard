/*
 * wubu_metadiag.c -- DUAL-TIMESCALE DIAGNOSE (see the header).
 */
#include "wubu_metadiag.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int wubu_metadiag_init(wubu_metadiag_t *md, wubu_hive_t *tissue,
                       int slow_every, int window_cap, float lr_scale)
{
    if (!md || !tissue) return -1;
    memset(md, 0, sizeof(*md));
    md->tissue = tissue;
    md->slow_every = slow_every > 0 ? slow_every : 20;
    md->win_cap = window_cap > 0 ? window_cap : 64;
    md->lr_scale = lr_scale > 0 ? lr_scale : 0.1f;
    md->mutation_rate = 0.5f;
    md->fitness_floor = 0.0f;
    md->window = (float *)calloc((size_t)md->win_cap, sizeof(float));
    if (!md->window) return -1;
    return 0;
}

int wubu_metadiag_fast(wubu_metadiag_t *md, const wubu_fast_signal_t *s)
{
    if (!md || !s) return -1;
    md->fast_cycles++;
    /* push the loss into the rolling window */
    if (md->win_n >= md->win_cap) {
        memmove(md->window, md->window + 1,
                (size_t)(md->win_cap - 1) * sizeof(float));
        md->window[md->win_cap - 1] = s->loss;
    } else {
        md->window[md->win_n++] = s->loss;
    }
    /* the fast path is due for a slow pass on the schedule */
    return (md->fast_cycles % (uint64_t)md->slow_every == 0) ? 1 : 0;
}

int wubu_metadiag_slow(wubu_metadiag_t *md)
{
    if (!md || !md->tissue) return -1;
    md->slow_cycles++;
    /* the loss trend over the window: the last-8 linear fit */
    float trend = 0.0f;
    if (md->win_n >= 2) {
        int win = md->win_n < 8 ? md->win_n : 8;
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < win; i++) {
            double x = (double)i, y = (double)md->window[md->win_n - win + i];
            sx += x; sy += y; sxx += x * x; sxy += x * y;
        }
        double den = (double)win * sxx - sx * sx;
        trend = (float)(den != 0.0 ? ((double)win * sxy - sx * sy) / den : 0.0);
    }
    /* the distribution shift: the first-half vs second-half window mean */
    float shift = 0.0f;
    if (md->win_n >= 4) {
        int half = md->win_n / 2;
        double m1 = 0, m2 = 0;
        for (int i = 0; i < half; i++) m1 += md->window[i];
        for (int i = half; i < md->win_n; i++) m2 += md->window[i];
        m1 /= half; m2 /= (double)(md->win_n - half);
        shift = (float)fabs(m2 - m1);
    }
    /* the lineage health: hive live vs capacity (a proxy for the
     * surviving fraction — the real lineage tracker feeds this) */
    size_t live = wubu_hive_live(md->tissue);
    size_t cap = wubu_hive_capacity(md->tissue);
    float health = (cap > 0) ? (float)live / (float)cap : 0.5f;

    /* the colony-state meta-cell: the summary the colony can read */
    wubu_meta_cell_t *mc = (wubu_meta_cell_t *)calloc(1, sizeof(wubu_meta_cell_t));
    if (!mc) return -1;
    mc->batch = md->fast_cycles;
    mc->hive_live = live;
    mc->hive_capacity = cap;
    mc->lineage_health = health;
    mc->shift = shift;
    mc->trend = trend;
    /* the adjustment: a positive trend (loss rising) raises the
     * mutation rate + lowers the floor (the colony gets more
     * aggressive); a negative trend (improving) does the opposite */
    float prev_rate = md->mutation_rate;
    if (trend > 0) {
        md->mutation_rate += md->lr_scale * 0.5f;
        md->fitness_floor -= md->lr_scale * 0.2f;
    } else {
        md->mutation_rate -= md->lr_scale * 0.3f;
        md->fitness_floor += md->lr_scale * 0.1f;
    }
    if (md->mutation_rate < 0.1f) md->mutation_rate = 0.1f;
    if (md->mutation_rate > 1.0f) md->mutation_rate = 1.0f;
    if (md->fitness_floor < 0.0f) md->fitness_floor = 0.0f;
    mc->mutation_rate = md->mutation_rate;
    mc->fitness_floor = md->fitness_floor;
    /* the meta-cell goes into the hive (the colony's self-knowledge) */
    wubu_hive_insert(md->tissue, mc);
    (void)prev_rate;
    return 0;
}

void wubu_metadiag_state(const wubu_metadiag_t *md, float *rate, float *floor)
{
    if (!md) return;
    if (rate) *rate = md->mutation_rate;
    if (floor) *floor = md->fitness_floor;
}

size_t wubu_metadiag_metacells(const wubu_metadiag_t *md)
{
    return md ? md->slow_cycles : 0;
}

void wubu_metadiag_stats(const wubu_metadiag_t *md, char *buf, size_t cap)
{
    if (!md || !buf || cap == 0) return;
    snprintf(buf, cap,
             "fast=%llu slow=%llu rate=%.2f floor=%.2f window=%d meta=%llu",
             (unsigned long long)md->fast_cycles,
             (unsigned long long)md->slow_cycles,
             md->mutation_rate, md->fitness_floor, md->win_n,
             (unsigned long long)md->slow_cycles);
}

void wubu_metadiag_free(wubu_metadiag_t *md)
{
    if (!md) return;
    free(md->window);
    memset(md, 0, sizeof(*md));
}
