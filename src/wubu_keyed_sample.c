/* wubu_keyed_sample.c -- EG-89 position-keyed exponential race. Strict C11. */
#include "wubu_keyed_sample.h"

#include <stdlib.h>
#include <math.h>

struct wubu_keyed_sample {
    uint64_t seed;
};

static uint64_t mix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

/* Counter-based Exp(1): -ln(U) with U from a 53-bit hash field. */
static double exp_key(uint64_t seed, int pos, int v)
{
    uint64_t h = mix64(seed ^
                       mix64((uint64_t)(unsigned)pos + 0x9e3779b9ULL) ^
                       mix64((uint64_t)(unsigned)v + 0x85ebca6bULL));
    double u = ((double)(h >> 11) + 0.5) / 9007199254740992.0;
    return -log(u);
}

wubu_keyed_sample_t *wubu_keyed_sample_create(uint64_t seed)
{
    wubu_keyed_sample_t *p = (wubu_keyed_sample_t *)calloc(1, sizeof(*p));
    if (p) p->seed = seed;
    return p;
}

void wubu_keyed_sample_free(wubu_keyed_sample_t *p) { free(p); }

int wubu_keyed_sample(const wubu_keyed_sample_t *p, int pos,
                      const float *probs, int vocab)
{
    int best = -1;
    double best_key = 0.0, total = 0.0;
    if (!p || !probs || pos < 0 || vocab <= 0) return -1;
    for (int v = 0; v < vocab; ++v) {
        double pv = probs[v];
        if (pv < 0.0) return -1;
        total += pv;
    }
    if (!(total > 0.0)) return -1;
    for (int v = 0; v < vocab; ++v) {
        double key;
        if (!(probs[v] > 0.0f)) continue;
        key = exp_key(p->seed, pos, v) / (double)probs[v];
        if (best < 0 || key < best_key) {
            best = v;
            best_key = key;
        }
    }
    return best;
}

int wubu_keyed_sample_at(const wubu_keyed_sample_t *p,
                         int prefix_len, int depth,
                         const float *probs, int vocab)
{
    if (prefix_len < 0 || depth < 0 || (unsigned)prefix_len + (unsigned)depth > 0x7fffffffu)
        return -1;
    return wubu_keyed_sample(p, prefix_len + depth, probs, vocab);
}
