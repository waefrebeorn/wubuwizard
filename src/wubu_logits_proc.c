/* wubu_logits_proc.c -- EG-34 typical-p + watermark logits processors. C11. */
#include "wubu_logits_proc.h"

#include <stdlib.h>
#include <math.h>

struct wubu_logits_proc {
    float typical_p, delta;
    uint64_t seed;
};

wubu_logits_proc_t *wubu_logits_proc_create(float typical_p, float delta,
                                            uint64_t seed)
{
    wubu_logits_proc_t *p;
    if (!(typical_p > 0.0f && typical_p <= 1.0f) || delta < 0.0f) return NULL;
    p = (wubu_logits_proc_t *)calloc(1, sizeof(*p));
    if (p) { p->typical_p = typical_p; p->delta = delta; p->seed = seed; }
    return p;
}

void wubu_logits_proc_free(wubu_logits_proc_t *p) { free(p); }

static uint64_t mix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

int wubu_logits_proc_is_green(const wubu_logits_proc_t *p, int pos, int token)
{
    uint64_t h;
    if (!p || pos < 0 || token < 0) return -1;
    h = mix64(p->seed ^ mix64((uint64_t)(unsigned)pos + 0x9e3779b9ULL) ^
              mix64((uint64_t)(unsigned)token + 0x85ebca6bULL));
    return (int)(h & 1ULL);
}

int wubu_logits_proc_apply(wubu_logits_proc_t *p, float *logits, int n, int pos)
{
    double probs[WUBU_LOGITS_MAX], total = 0.0, cum = 0.0;
    int order[WUBU_LOGITS_MAX], kept = n, cut = n - 1;
    if (!p || !logits || n < 1 || n > WUBU_LOGITS_MAX || pos < 0) return -1;
    for (int i = 0; i < n; ++i) {
        double v = logits[i];
        probs[i] = exp(v > 40.0 ? 40.0 : (v < -40.0 ? -40.0 : v));
        total += probs[i];
        order[i] = i;
    }
    if (!(total > 0.0)) return -1;
    for (int i = 1; i < n; ++i) {           /* descending by probability */
        int key = order[i], j = i - 1;
        while (j >= 0 && probs[order[j]] < probs[key]) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = key;
    }
    if (p->typical_p < 1.0f) {
        for (int i = 0; i < n; ++i) {
            cum += probs[order[i]] / total;
            if (cum >= (double)p->typical_p) { cut = i; break; }
        }
        kept = cut + 1;
        for (int i = cut + 1; i < n; ++i) logits[order[i]] = -1e31f;
    }
    if (p->delta > 0.0f) {
        for (int i = 0; i < n; ++i)
            if (logits[i] > -1e30f && wubu_logits_proc_is_green(p, pos, i))
                logits[i] += p->delta;
    }
    return kept;
}
