/* wubu_ttc_budget.c -- test-time-compute budget pad (LP-14).
 * C11, opaque struct, status-as-value, no third-party deps.
 *
 * Wall time is measured with clock() from begin(); add_wall_ms injects an
 * additional deterministic offset so external timers / tests can contribute. */
#include <stdlib.h>
#include <time.h>
#include "wubu_ttc_budget.h"

struct wubu_ttc_budget {
    long long max_wall_ms;
    size_t max_kv_bytes;
    int max_candidates;
    int plateau_window;
    double min_gain;

    clock_t start;
    long long injected_wall_ms;
    size_t kv_bytes;
    int candidates;
    int stopped;
    int plateaued;
    double best_score;
    int non_improve;
};

wubu_ttc_budget_t *wubu_ttc_budget_create(long long max_wall_ms,
                                          size_t max_kv_bytes,
                                          int max_candidates,
                                          int plateau_window,
                                          double min_gain)
{
    if (max_wall_ms < 0 || max_candidates < 1 || plateau_window < 1 ||
        min_gain < 0.0) return NULL;
    wubu_ttc_budget_t *b =
        (wubu_ttc_budget_t *)calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->max_wall_ms = max_wall_ms;
    b->max_kv_bytes = max_kv_bytes;
    b->max_candidates = max_candidates;
    b->plateau_window = plateau_window;
    b->min_gain = min_gain;
    b->best_score = -1e300;
    return b;
}

void wubu_ttc_budget_free(wubu_ttc_budget_t *b) { free(b); }

int wubu_ttc_budget_begin(wubu_ttc_budget_t *b)
{
    if (!b) return -1;
    b->start = clock();
    b->injected_wall_ms = 0;
    b->kv_bytes = 0;
    b->candidates = 0;
    b->stopped = 0;
    b->plateaued = 0;
    b->best_score = -1e300;
    b->non_improve = 0;
    return 0;
}

static long long measured_ms(const wubu_ttc_budget_t *b)
{
    clock_t d = clock() - b->start;
    if (d < 0) d = 0;
    return (long long)((double)d * 1000.0 / (double)CLOCKS_PER_SEC);
}

long long wubu_ttc_budget_elapsed_ms(const wubu_ttc_budget_t *b)
{
    if (!b) return -1;
    return measured_ms(b) + b->injected_wall_ms;
}

int wubu_ttc_budget_add_wall_ms(wubu_ttc_budget_t *b, long long ms)
{
    if (!b || ms < 0) return -1;
    b->injected_wall_ms += ms;
    return 0;
}

int wubu_ttc_budget_add_kv(wubu_ttc_budget_t *b, size_t kv_bytes)
{
    if (!b) return -1;
    b->kv_bytes += kv_bytes;
    return 0;
}

size_t wubu_ttc_budget_kv_bytes(const wubu_ttc_budget_t *b)
{
    return b ? b->kv_bytes : 0;
}
int wubu_ttc_budget_candidates(const wubu_ttc_budget_t *b)
{
    return b ? b->candidates : -1;
}
int wubu_ttc_budget_plateaued(const wubu_ttc_budget_t *b)
{
    return b ? b->plateaued : 0;
}

int wubu_ttc_budget_try(wubu_ttc_budget_t *b, double score)
{
    if (!b || b->stopped) return 0;

    b->candidates++;
    if (score > b->best_score + b->min_gain) {
        b->best_score = score;
        b->non_improve = 0;
    } else {
        b->non_improve++;
    }

    int stop = 0;
    if (b->candidates >= b->max_candidates) stop = 1;
    if (wubu_ttc_budget_elapsed_ms(b) >= b->max_wall_ms) stop = 1;
    if (b->kv_bytes >= b->max_kv_bytes) stop = 1;
    if (b->non_improve >= b->plateau_window) {
        b->plateaued = 1;
        stop = 1;
    }

    if (stop) b->stopped = 1;
    return stop ? 0 : 1;
}
