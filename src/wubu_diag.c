/*
 * wubu_diag.c -- THE HIVE DIAGNOSTIC SYSTEM (research/056, INDEX AN08).
 *
 * The hive is not just where the colony lives -- the hive IS the
 * diagnostic system. Every measurement is a wubu_diag_cell stored in the
 * hive tissue; diagnosis = walking that tissue; mutation = growing/
 * shrinking that tissue; the walker + 5+1 recovery = replaying it.
 * Memory-bounded by construction: the ring capacity recycles the oldest
 * cell on insert (the 103-checkpoint / 15 GiB lesson).
 *
 * C11, self-contained, wraps wubu_hive only.
 */
#include "wubu_diag.h"
#include "wubu.h"          /* WUBU_* dims for the real-grad bridge */
#include "wubu_train.h"    /* wubu_train_t gradient accumulators */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

struct wubu_diag {
    wubu_hive_t *hive;
    size_t capacity;
    unsigned kinds;                     /* enabled bitmask (0 = all) */
    int64_t next_step;                  /* the diag's own clock */
    wubu_diag_agg agg[WUBU_DIAG_NKINDS];/* running sums per kind */
};

static const char *DIAG_NAME[WUBU_DIAG_NKINDS] = {
    "LOSS", "GRAD", "ENTROPY", "ROUTE", "UTIL", "BI",
    "ORACLE", "DATA", "SYS", "MUT"
};

static int kind_enabled(const wubu_diag_t *d, wubu_diag_kind k)
{
    if (k < 0 || k >= WUBU_DIAG_NKINDS) return 0;
    return d->kinds == 0 || (d->kinds & (1u << (unsigned)k));
}

wubu_diag_t *wubu_diag_init(wubu_hive_t *hive, unsigned kinds)
{
    if (!hive) return NULL;
    wubu_diag_t *d = (wubu_diag_t *)calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->hive = hive;
    d->capacity = WUBU_DIAG_DEFAULT_CAPACITY;
    d->kinds = kinds;
    return d;
}

void wubu_diag_set_capacity(wubu_diag_t *d, size_t capacity)
{
    if (d && capacity > 0) d->capacity = capacity;
}

/* ---- ring discipline: recycle the OLDEST cell (min step) ---- */
struct oldest_ctx { wubu_diag_cell *oldest; int64_t min_step; };
static int find_oldest(void *ptr, void *user)
{
    struct oldest_ctx *o = (struct oldest_ctx *)user;
    wubu_diag_cell *c = (wubu_diag_cell *)ptr;
    if (!o->oldest || c->step < o->min_step) {
        o->oldest = c;
        o->min_step = c->step;
    }
    return 0;
}

int wubu_diag_record(wubu_diag_t *d, wubu_diag_kind kind, int cell,
                     float value, float meta)
{
    if (!d || !kind_enabled(d, kind)) return -1;

    /* at capacity: recycle the oldest cell BEFORE inserting (the ring).
     * Erase = skip-mark + freelist push, O(1); free the cell struct. */
    if (wubu_hive_live(d->hive) >= d->capacity) {
        struct oldest_ctx o = { NULL, 0 };
        wubu_hive_foreach(d->hive, find_oldest, &o);
        if (o.oldest) {
            wubu_diag_agg *a = &d->agg[(unsigned)o.oldest->kind];
            a->n--;
            a->sum -= o.oldest->value;
            a->sumsq -= (double)o.oldest->value * o.oldest->value;
            wubu_hive_erase(d->hive, o.oldest);
            free(o.oldest);
        }
    }

    wubu_diag_cell *c = (wubu_diag_cell *)malloc(sizeof(*c));
    if (!c) return -1;
    c->kind = kind;
    c->step = d->next_step++;
    c->cell = cell;
    c->value = value;
    c->meta = meta;

    if (wubu_hive_insert(d->hive, c) != 0) { free(c); return -1; }
    wubu_diag_agg *a = &d->agg[(unsigned)kind];
    a->n++;
    a->sum += value;
    a->sumsq += (double)value * value;
    return 0;
}

/* ---- per-kind live-window stats from the running sums ---- */
static double kind_mean(const wubu_diag_t *d, wubu_diag_kind k)
{
    const wubu_diag_agg *a = &d->agg[(unsigned)k];
    return a->n > 0 ? a->sum / (double)a->n : 0.0;
}
static double kind_std(const wubu_diag_t *d, wubu_diag_kind k)
{
    const wubu_diag_agg *a = &d->agg[(unsigned)k];
    if (a->n < 2) return 0.0;
    double mean = a->sum / (double)a->n;
    double var = a->sumsq / (double)a->n - mean * mean;
    return var > 0 ? sqrt(var) : 0.0;
}

float wubu_diag_zscore(const wubu_diag_t *d, wubu_diag_kind kind, float value)
{
    if (!d || kind < 0 || kind >= WUBU_DIAG_NKINDS) return 0.0f;
    double sd = kind_std(d, kind);
    if (sd == 0.0) return 0.0f;
    return (float)((value - kind_mean(d, kind)) / sd);
}

/* ---- classify: grow/shrink/stasis over the live GRAD cells ----
 * grow   = latest grad is out of family (z > +2.5) -> overworked -> mitosis
 * shrink = the cell stayed below the ABSOLUTE floor (1e-4) for its whole
 *          live window -> dead -> apoptosis (the DA bug: relative-only
 *          misses the all-dead colony).
 */
#define MAX_CELLS 256
struct cellstat { int cell; int n; int all_below_floor; float latest; };
struct classify_ctx { struct cellstat stats[MAX_CELLS]; int n_cells; };

static int classify_visit(void *ptr, void *user)
{
    struct classify_ctx *cx = (struct classify_ctx *)user;
    wubu_diag_cell *c = (wubu_diag_cell *)ptr;
    if (c->kind != WUBU_DIAG_GRAD) return 0;
    int i;
    for (i = 0; i < cx->n_cells; i++)
        if (cx->stats[i].cell == c->cell) break;
    if (i == cx->n_cells) {
        if (cx->n_cells >= MAX_CELLS) return 0;
        memset(&cx->stats[i], 0, sizeof(cx->stats[i]));
        cx->stats[i].cell = c->cell;
        cx->stats[i].all_below_floor = 1;
        cx->n_cells++;
    }
    cx->stats[i].n++;
    cx->stats[i].latest = c->value;   /* hive order = insert order */
    if (c->value >= WUBU_DIAG_GRAD_FLOOR) cx->stats[i].all_below_floor = 0;
    return 0;
}

int wubu_diag_classify(wubu_diag_t *d, float *grow, float *shrink)
{
    if (!d) return -1;
    float g = 0.0f, s = 0.0f;
    struct classify_ctx cx;
    memset(&cx, 0, sizeof(cx));
    wubu_hive_foreach(d->hive, classify_visit, &cx);
    for (int i = 0; i < cx.n_cells; i++) {
        if (cx.stats[i].all_below_floor && cx.stats[i].n > 0) {
            s++;                               /* dead -> shrink */
            continue;
        }
        float z = wubu_diag_zscore(d, WUBU_DIAG_GRAD, cx.stats[i].latest);
        if (z > WUBU_DIAG_Z_THRESH) g++;       /* overworked -> grow */
    }
    if (grow)   *grow = g;
    if (shrink) *shrink = s;
    return 0;
}

/* ---- THE REAL-GRAD BRIDGE (milestone 2) ----
 * Record the trainer's ACTUAL per-layer gradient norms as GRAD cells.
 * cell = layer index; value = Frobenius norm of the layer's accumulated
 * matrix grads (q/k/v/o/g + gate_up + down). Cell -2 = the embedding
 * grad norm (a global). This is what makes the diagnostic measure the
 * REAL training signal, not toy tasks.
 */
static double fro_norm(const float *g, size_t n)
{
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += (double)g[i] * g[i];
    return s;
}

int wubu_diag_record_grads(wubu_diag_t *d, const struct wubu_train *tr)
{
    if (!d || !tr) return -1;
    int n_rec = 0;
    const int L = WUBU_LAYERS;
    const int D = WUBU_DIM, KV = WUBU_KV_HEADS * 64, FF = WUBU_FFN_DIM;
    for (int l = 0; l < L; l++) {
        double s = 0.0;
        s += fro_norm(tr->q_proj_g[l], (size_t)D * D);
        s += fro_norm(tr->k_proj_g[l], (size_t)D * KV);
        s += fro_norm(tr->v_proj_g[l], (size_t)D * KV);
        s += fro_norm(tr->o_proj_g[l], (size_t)D * D);
        s += fro_norm(tr->g_proj_g[l], (size_t)D * D);
        s += fro_norm(tr->gate_up_g[l], (size_t)D * (2 * FF));
        s += fro_norm(tr->down_g[l], (size_t)FF * D);
        float norm = (float)sqrt(s);
        if (wubu_diag_record(d, WUBU_DIAG_GRAD, l, norm, 0.0f) != 0) return -1;
        n_rec++;
    }
    if (tr->emb_g) {
        float en = (float)sqrt(fro_norm(tr->emb_g, (size_t)16384 * D));
        if (wubu_diag_record(d, WUBU_DIAG_GRAD, -2, en, 0.0f) != 0) return -1;
        n_rec++;
    }
    return n_rec;
}

/* ---- the causal walker ----
 * On a fitness drop, find the EARLIEST out-of-family measurement that
 * strictly precedes it -- the root cause candidate. Honest fallback:
 * "unexplained" when nothing anomalous precedes the drop.
 */
struct walker_ctx { wubu_diag_t *d; int64_t drop_step;
                    wubu_diag_cell *best; float best_z; int found; };
static int walker_visit(void *ptr, void *user)
{
    struct walker_ctx *w = (struct walker_ctx *)user;
    wubu_diag_cell *c = (wubu_diag_cell *)ptr;
    if (c->step >= w->drop_step) return 0;     /* must precede the drop */
    float z = wubu_diag_zscore(w->d, c->kind, c->value);
    if (fabsf(z) > WUBU_DIAG_Z_THRESH) {
        if (!w->found || c->step < w->best->step) {
            w->best = c;
            w->best_z = z;
            w->found = 1;
        }
    }
    return 0;
}

int wubu_diag_walk(wubu_diag_t *d, int64_t drop_step, char *report,
                   size_t cap)
{
    if (!d || !report || cap == 0) return -1;
    struct walker_ctx w;
    memset(&w, 0, sizeof(w));
    w.d = d;
    w.drop_step = drop_step;
    wubu_hive_foreach(d->hive, walker_visit, &w);
    if (!w.found) {
        snprintf(report, cap,
                 "no out-of-family measurement found; fitness drop "
                 "unexplained");
        return 0;
    }
    const char *kn = (w.best->kind >= 0 && w.best->kind < WUBU_DIAG_NKINDS)
                     ? DIAG_NAME[(unsigned)w.best->kind] : "?";
    snprintf(report, cap,
             "cause at step %lld: kind=%s cell=%d z=%+.2f preceded the "
             "fitness drop at step %lld",
             (long long)w.best->step, kn, w.best->cell,
             (double)w.best_z, (long long)drop_step);
    return 1;
}

/* ---- snapshot: JSON dump (aggregates + live cells) ---- */
int wubu_diag_snapshot(wubu_diag_t *d, const char *json_path)
{
    if (!d || !json_path) return -1;
    FILE *f = fopen(json_path, "w");
    if (!f) return -1;
    fprintf(f, "{\n  \"capacity\": %zu,\n  \"live\": %zu,\n  \"kinds\": [\n",
            d->capacity, wubu_hive_live(d->hive));
    int first_kind = 1;
    for (int k = 0; k < WUBU_DIAG_NKINDS; k++) {
        if (d->agg[k].n == 0) continue;
        fprintf(f, "%s    {\"kind\": \"%s\", \"n\": %lld, \"mean\": %.6g, "
                   "\"std\": %.6g}",
                first_kind ? "" : ",\n", DIAG_NAME[k],
                (long long)d->agg[k].n, kind_mean(d, (wubu_diag_kind)k),
                kind_std(d, (wubu_diag_kind)k));
        first_kind = 0;
    }
    fprintf(f, "\n  ],\n  \"cells\": [\n");
    int first_cell = 1;
    for (wubu_hive_block_t *blk = d->hive->head; blk; blk = blk->next) {
        for (size_t i = 0; i < blk->cap; i++) {
            if (hive_skip_get(blk, i)) continue;
            wubu_diag_cell *c = (wubu_diag_cell *)blk->slots[i];
            fprintf(f, "%s    {\"kind\": \"%s\", \"step\": %lld, \"cell\": %d, "
                       "\"value\": %.6g, \"meta\": %.6g}",
                    first_cell ? "" : ",\n",
                    DIAG_NAME[(unsigned)c->kind], (long long)c->step,
                    c->cell, (double)c->value, (double)c->meta);
            first_cell = 0;
        }
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    return 0;
}

size_t wubu_diag_live(const wubu_diag_t *d)
{
    return d ? wubu_hive_live(d->hive) : 0;
}

void wubu_diag_free(wubu_diag_t *d)
{
    if (!d) return;
    /* free every live cell, then clear the tissue (the hive itself is
     * caller-owned) */
    struct free_ctx { wubu_hive_t *hive; } fc = { d->hive };
    struct free_ctx *fcp = &fc;
    (void)fcp;
    for (wubu_hive_block_t *blk = d->hive->head; blk; blk = blk->next)
        for (size_t i = 0; i < blk->cap; i++)
            if (!hive_skip_get(blk, i)) free(blk->slots[i]);
    wubu_hive_clear(d->hive);
    free(d);
}
