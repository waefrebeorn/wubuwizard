#ifndef WUBU_GOAL_RESAMPLE_H
#define WUBU_GOAL_RESAMPLE_H

/* wubu_goal_resample.h -- bounded candidate resampling pad (LP-03).
 * C11, opaque, status-as-value, no third-party deps.
 *
 * begin() opens a round; submit(score, verifier_ok) returns ACCEPT / REJECT /
 * STOP. Stops are bounded by max_attempts or max_candidates, or forced early
 * via stop(). A numeric snapshot can be saved and restored (rollback). */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_goal_resample wubu_goal_resample_t;
typedef struct wubu_goal_resample_snapshot wubu_goal_resample_snapshot_t;

typedef enum {
    WUBU_RESAMPLE_REJECT = 0,
    WUBU_RESAMPLE_ACCEPT = 1,
    WUBU_RESAMPLE_STOP = 2,
    WUBU_RESAMPLE_INVALID = -1
} wubu_resample_verdict_t;

wubu_goal_resample_t *wubu_goal_resample_create(int max_candidates, int max_attempts);
void wubu_goal_resample_free(wubu_goal_resample_t *r);

/* Reset counters and best-score state for a new round. */
int wubu_goal_resample_begin(wubu_goal_resample_t *r);

wubu_resample_verdict_t wubu_goal_resample_submit(wubu_goal_resample_t *r,
                                                  double score, int verifier_ok);

/* Force an early stop; the next submit returns STOP without consuming an
 * attempt. */
int wubu_goal_resample_stop(wubu_goal_resample_t *r);

/* Opaque numeric snapshot of the run state. */
wubu_goal_resample_snapshot_t *wubu_goal_resample_snapshot(const wubu_goal_resample_t *r);
int wubu_goal_resample_restore(wubu_goal_resample_t *r,
                               const wubu_goal_resample_snapshot_t *snap);
void wubu_goal_resample_snapshot_free(wubu_goal_resample_snapshot_t *snap);

int wubu_goal_resample_attempts(const wubu_goal_resample_t *r);
int wubu_goal_resample_accepted(const wubu_goal_resample_t *r);
double wubu_goal_resample_best_score(const wubu_goal_resample_t *r);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_GOAL_RESAMPLE_H */
