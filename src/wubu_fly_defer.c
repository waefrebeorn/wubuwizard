/* wubu_fly_defer.c -- EG-80 FLy deferral policy gate. Strict C11. */
#include "wubu_fly_defer.h"

#include <stdlib.h>
#include <math.h>

struct wubu_fly_defer {
    double threshold;
    int window, enabled, deferred, rejections;
};

wubu_fly_defer_t *wubu_fly_defer_create(double threshold, int window,
                                        int enabled)
{
    wubu_fly_defer_t *p;
    if (!(threshold >= 0.0 && threshold <= 1.0) || window < 1 ||
        (enabled != 0 && enabled != 1)) return NULL;
    p = (wubu_fly_defer_t *)calloc(1, sizeof(*p));
    if (p) { p->threshold = threshold; p->window = window; p->enabled = enabled; }
    return p;
}

void wubu_fly_defer_free(wubu_fly_defer_t *p) { free(p); }

double wubu_fly_normalized_entropy(const double *scores, int k)
{
    double total = 0.0, h = 0.0;
    if (!scores || k < 2 || k > WUBU_FLY_MAX_K) return -1.0;
    for (int i = 0; i < k; ++i) {
        if (scores[i] < 0.0) return -1.0;
        total += scores[i];
    }
    if (!(total > 0.0)) return -1.0;
    for (int i = 0; i < k; ++i) {
        double p = scores[i] / total;
        if (p > 0.0) h -= p * log(p);
    }
    return h / log((double)k);
}

int wubu_fly_defer_decide(wubu_fly_defer_t *p, int rejected_now,
                          double norm_entropy, const unsigned char *align_next,
                          int window_n)
{
    int defer;
    if (!p || (rejected_now != 0 && rejected_now != 1) ||
        !(norm_entropy >= 0.0 && norm_entropy <= 1.0) ||
        !align_next || window_n != p->window) return -1;
    if (!rejected_now) return 0;
    defer = p->enabled && norm_entropy >= p->threshold;
    for (int i = 0; defer && i < window_n; ++i)
        if (align_next[i] != 1) defer = 0;
    if (defer) ++p->deferred;
    else ++p->rejections;
    return defer;
}

int wubu_fly_defer_deferred(const wubu_fly_defer_t *p)
{
    return p ? p->deferred : -1;
}

int wubu_fly_defer_rejections(const wubu_fly_defer_t *p)
{
    return p ? p->rejections : -1;
}
