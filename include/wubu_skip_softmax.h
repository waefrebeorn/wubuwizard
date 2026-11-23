#ifndef WUBU_SKIP_SOFTMAX_H
#define WUBU_SKIP_SOFTMAX_H

/*
 * wubu_skip_softmax.h -- EG-23: Skip-Softmax / BLASST style calibrated
 * attention-block pruning, reimplemented in portable C11.
 *
 * A length bucket collects observed per-block maximum QK scores. Its skip
 * threshold starts at a low percentile of those maxima; selection drops
 * blocks below the threshold only while the estimated lost softmax mass stays
 * inside an accuracy budget. When blocks carry comparable mass, the guard
 * raises the effective threshold and keeps everything.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_skip_calib wubu_skip_calib_t;

wubu_skip_calib_t *wubu_skip_calib_create(int buckets, double epsilon);
void wubu_skip_calib_free(wubu_skip_calib_t *c);

/* Feed observed per-block maxima for one length bucket. Returns 0 or -1. */
int wubu_skip_calib_observe(wubu_skip_calib_t *c, int bucket,
                            const double *maxima, int n);

/* Nominal (pre-guard) threshold for a bucket. */
double wubu_skip_calib_threshold(const wubu_skip_calib_t *c, int bucket);

/* Choose which blocks to keep. keep[i]=1/0; returns kept count or -1. */
int wubu_skip_calib_select(wubu_skip_calib_t *c, int bucket,
                           const double *maxima, int n,
                           int *keep, int cap);

/* Cumulative counters and worst estimated lost mass. Any arg may be NULL. */
void wubu_skip_calib_stats(const wubu_skip_calib_t *c,
                           int *skipped, int *kept, double *worst_err);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SKIP_SOFTMAX_H */
