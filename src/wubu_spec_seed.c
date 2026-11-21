/* wubu_spec_seed.c -- LP-08: deterministic seeded rejection sampling.
 * Strict C11, status-as-value, no third-party deps.
 * See include/wubu_spec_seed.h. */
#include "wubu_spec_seed.h"

#define XS_DEFAULT_SEED 0x9E3779B97F4A7C15ULL
#define XS_MULTIPLIER   2685821657736338717ULL

static uint64_t xs_next(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * XS_MULTIPLIER;
}

/* Uniform double in [0,1) from the top 53 bits of the PRNG word. */
static double xs_unit(uint64_t *s)
{
    return (double)(xs_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

int wubu_spec_seed_accept(const float *probs, int n, uint64_t seed,
                          int *out_ids, int n_out)
{
    if (!probs || !out_ids || n <= 0 || n_out < 0) return -1;

    uint64_t s = seed ? seed : XS_DEFAULT_SEED;
    int accepted = 0;
    /* Bound work so a degenerate all-zero distribution cannot loop forever. */
    uint64_t max_attempts = (uint64_t)n_out * 64ULL + 1024ULL;
    uint64_t attempts = 0;

    while (accepted < n_out && attempts++ < max_attempts) {
        int idx = (int)(xs_next(&s) % (uint64_t)n);   /* uniform candidate */
        double p = (double)probs[idx];
        if (p < 0.0) p = 0.0;
        if (p > 1.0) p = 1.0;
        if (xs_unit(&s) < p) out_ids[accepted++] = idx; /* reject / accept */
    }
    return accepted;
}
