/* wubu_goal_resample.c -- bounded candidate resampling pad (LP-03).
 * C11, opaque structs, status-as-value, no third-party deps. */
#include <stdlib.h>
#include "wubu_goal_resample.h"

struct wubu_goal_resample {
    int max_candidates;
    int max_attempts;
    int attempts;
    int accepted;
    int stopped;
    double best_score;
    int best_verifier_ok;
};

struct wubu_goal_resample_snapshot {
    int attempts;
    int accepted;
    int stopped;
    double best_score;
    int best_verifier_ok;
};

wubu_goal_resample_t *wubu_goal_resample_create(int max_candidates, int max_attempts)
{
    if (max_candidates < 1 || max_attempts < 1) return NULL;
    wubu_goal_resample_t *r = (wubu_goal_resample_t *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->max_candidates = max_candidates;
    r->max_attempts = max_attempts;
    r->best_score = -1e300;
    return r;
}

void wubu_goal_resample_free(wubu_goal_resample_t *r) { free(r); }

int wubu_goal_resample_begin(wubu_goal_resample_t *r)
{
    if (!r) return -1;
    r->attempts = 0;
    r->accepted = 0;
    r->stopped = 0;
    r->best_score = -1e300;
    r->best_verifier_ok = 0;
    return 0;
}

wubu_resample_verdict_t wubu_goal_resample_submit(wubu_goal_resample_t *r,
                                                  double score, int verifier_ok)
{
    if (!r) return WUBU_RESAMPLE_INVALID;
    if (r->stopped) return WUBU_RESAMPLE_STOP;

    r->attempts++;
    wubu_resample_verdict_t v = WUBU_RESAMPLE_REJECT;
    if (verifier_ok > 0 && score > r->best_score) {
        r->best_score = score;
        r->best_verifier_ok = 1;
        r->accepted++;
        v = WUBU_RESAMPLE_ACCEPT;
    }
    if (r->attempts >= r->max_attempts || r->accepted >= r->max_candidates) {
        r->stopped = 1;
        v = WUBU_RESAMPLE_STOP;
    }
    return v;
}

int wubu_goal_resample_stop(wubu_goal_resample_t *r)
{
    if (!r) return -1;
    r->stopped = 1;
    return 0;
}

wubu_goal_resample_snapshot_t *wubu_goal_resample_snapshot(const wubu_goal_resample_t *r)
{
    if (!r) return NULL;
    wubu_goal_resample_snapshot_t *snap =
        (wubu_goal_resample_snapshot_t *)calloc(1, sizeof(*snap));
    if (!snap) return NULL;
    snap->attempts = r->attempts;
    snap->accepted = r->accepted;
    snap->stopped = r->stopped;
    snap->best_score = r->best_score;
    snap->best_verifier_ok = r->best_verifier_ok;
    return snap;
}

int wubu_goal_resample_restore(wubu_goal_resample_t *r,
                               const wubu_goal_resample_snapshot_t *snap)
{
    if (!r || !snap) return -1;
    r->attempts = snap->attempts;
    r->accepted = snap->accepted;
    r->stopped = snap->stopped;
    r->best_score = snap->best_score;
    r->best_verifier_ok = snap->best_verifier_ok;
    return 0;
}

void wubu_goal_resample_snapshot_free(wubu_goal_resample_snapshot_t *snap)
{
    free(snap);
}

int wubu_goal_resample_attempts(const wubu_goal_resample_t *r)
{
    return r ? r->attempts : -1;
}
int wubu_goal_resample_accepted(const wubu_goal_resample_t *r)
{
    return r ? r->accepted : -1;
}
double wubu_goal_resample_best_score(const wubu_goal_resample_t *r)
{
    return r ? r->best_score : -1e300;
}
