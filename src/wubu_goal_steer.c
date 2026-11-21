/* wubu_goal_steer.c -- goal-drift steering pad (LP-02).
 * C11, opaque struct, status-as-value, no third-party deps. */
#include <stdlib.h>
#include <math.h>
#include "wubu_goal_steer.h"

struct wubu_goal_steer {
    double tau;
    double last_drift;
    int observed;   /* last observe() magnitude was above tau */
    int veto;
};

wubu_goal_steer_t *wubu_goal_steer_create(double tau)
{
    if (!(tau > 0.0)) return NULL;
    wubu_goal_steer_t *s = (wubu_goal_steer_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->tau = tau;
    return s;
}

void wubu_goal_steer_free(wubu_goal_steer_t *s) { free(s); }

double wubu_goal_steer_tau(const wubu_goal_steer_t *s)
{
    return s ? s->tau : -1.0;
}

int wubu_goal_steer_drift_observed(const wubu_goal_steer_t *s)
{
    return s ? s->observed : 0;
}

int wubu_goal_steer_observe(wubu_goal_steer_t *s, double drift)
{
    if (!s) return 0;
    s->last_drift = drift;
    s->observed = (fabs(drift) > s->tau) ? 1 : 0;
    return s->observed;
}

int wubu_goal_steer_set_veto(wubu_goal_steer_t *s, int veto)
{
    if (!s) return -1;
    s->veto = veto ? 1 : 0;
    return 0;
}

int wubu_goal_steer_veto(const wubu_goal_steer_t *s)
{
    return s ? s->veto : 0;
}

wubu_goal_steer_status_t wubu_goal_steer_apply(wubu_goal_steer_t *s,
                                               wubu_goal_verdict_t *decision)
{
    if (!s || !decision) return WUBU_GOAL_STEER_INVALID;
    if (s->veto) return WUBU_GOAL_STEER_VETOED;   /* preserve judge decision */
    if (s->observed) {
        *decision = WUBU_GOAL_CONTINUE;
        return WUBU_GOAL_STEER_APPLIED;
    }
    return WUBU_GOAL_STEER_OK;
}
