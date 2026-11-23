/*
 * wubu_skip_softmax.c -- EG-23 calibrated block pruning. Pure C11.
 */
#include "wubu_skip_softmax.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    int samples;
    double lo, hi;
    double threshold;
} bucket_t;

struct wubu_skip_calib {
    int n_buckets;
    double epsilon;
    bucket_t *b;
    int skipped, kept;
    double worst_err;
};

wubu_skip_calib_t *wubu_skip_calib_create(int buckets, double epsilon) {
    wubu_skip_calib_t *c;
    if (buckets <= 0 || epsilon <= 0.0) return NULL;
    c = (wubu_skip_calib_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->b = (bucket_t *)calloc((size_t)buckets, sizeof(bucket_t));
    if (!c->b) { free(c); return NULL; }
    c->n_buckets = buckets;
    c->epsilon = epsilon;
    for (int i = 0; i < buckets; i++) c->b[i].lo = HUGE_VAL;
    return c;
}

void wubu_skip_calib_free(wubu_skip_calib_t *c) {
    if (!c) return;
    free(c->b);
    free(c);
}

int wubu_skip_calib_observe(wubu_skip_calib_t *c, int bucket,
                            const double *maxima, int n) {
    bucket_t *b;
    if (!c || bucket < 0 || bucket >= c->n_buckets || !maxima || n <= 0) return -1;
    b = &c->b[bucket];
    for (int i = 0; i < n; i++) {
        if (maxima[i] < b->lo) b->lo = maxima[i];
        if (maxima[i] > b->hi) b->hi = maxima[i];
    }
    b->samples += n;
    b->threshold = b->lo + 0.25 * (b->hi - b->lo);
    return 0;
}

double wubu_skip_calib_threshold(const wubu_skip_calib_t *c, int bucket) {
    if (!c || bucket < 0 || bucket >= c->n_buckets) return 0.0;
    return c->b[bucket].threshold;
}

static double lost_mass(const double *maxima, int n, const int *keep) {
    double mx = maxima[0], num = 0.0, den = 0.0;
    for (int i = 0; i < n; i++) if (maxima[i] > mx) mx = maxima[i];
    for (int i = 0; i < n; i++) {
        double w = exp(maxima[i] - mx);
        if (keep[i]) den += w;
        else num += w;
    }
    return den > 0.0 ? num / (den + num) : 1.0;
}

int wubu_skip_calib_select(wubu_skip_calib_t *c, int bucket,
                           const double *maxima, int n,
                           int *keep, int cap) {
    bucket_t *b;
    double thr, err;
    int kept = 0;
    if (!c || bucket < 0 || bucket >= c->n_buckets || !maxima || !keep ||
        n <= 0 || cap < n) return -1;
    b = &c->b[bucket];
    if (b->samples == 0) return -1;
    thr = b->threshold;
    for (int pass = 0; pass <= n; pass++) {
        for (int i = 0; i < n; i++) keep[i] = maxima[i] >= thr;
        err = lost_mass(maxima, n, keep);
        if (err <= c->epsilon) break;
        /* Too much mass would be lost: admit the weakest skipped block. */
        int weakest = -1;
        double worst = HUGE_VAL;
        for (int i = 0; i < n; i++)
            if (!keep[i] && maxima[i] < worst) { worst = maxima[i]; weakest = i; }
        if (weakest < 0) break;
        thr = worst;
    }
    for (int i = 0; i < n; i++) kept += keep[i];
    c->skipped += n - kept;
    c->kept += kept;
    c->worst_err = fmax(c->worst_err, err);
    return kept;
}

void wubu_skip_calib_stats(const wubu_skip_calib_t *c,
                           int *skipped, int *kept, double *worst_err) {
    if (!c) return;
    if (skipped) *skipped = c->skipped;
    if (kept) *kept = c->kept;
    if (worst_err) *worst_err = c->worst_err;
}
