#ifndef WUBU_GOAL_STEER_H
#define WUBU_GOAL_STEER_H

/* wubu_goal_steer.h -- goal-drift steering pad (LP-02). C11, opaque,
 * status-as-value, no third-party deps.
 *
 * Observes a drift magnitude against a threshold tau. When drift exceeds tau
 * the steer biases an existing judge decision back toward CONTINUE, unless a
 * veto flag is set -- in which case the judge's decision is preserved. */

#include <stdint.h>
#include "wubu_goal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_goal_steer wubu_goal_steer_t;

typedef enum {
    WUBU_GOAL_STEER_OK = 0,
    WUBU_GOAL_STEER_APPLIED = 1,
    WUBU_GOAL_STEER_VETOED = 2,
    WUBU_GOAL_STEER_INVALID = -1
} wubu_goal_steer_status_t;

wubu_goal_steer_t *wubu_goal_steer_create(double tau);
void wubu_goal_steer_free(wubu_goal_steer_t *s);

double wubu_goal_steer_tau(const wubu_goal_steer_t *s);
int wubu_goal_steer_drift_observed(const wubu_goal_steer_t *s);

/* Returns 1 only when |drift| is strictly above tau (and records it). */
int wubu_goal_steer_observe(wubu_goal_steer_t *s, double drift);

int wubu_goal_steer_set_veto(wubu_goal_steer_t *s, int veto);
int wubu_goal_steer_veto(const wubu_goal_steer_t *s);

/* Apply steering to a judge decision (status-as-value).
 *   veto set              -> decision preserved, VETOED
 *   drift above tau       -> decision forced to CONTINUE, APPLIED
 *   otherwise             -> decision unchanged, OK
 *   NULL arguments        -> INVALID */
wubu_goal_steer_status_t wubu_goal_steer_apply(wubu_goal_steer_t *s,
                                               wubu_goal_verdict_t *decision);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_GOAL_STEER_H */
