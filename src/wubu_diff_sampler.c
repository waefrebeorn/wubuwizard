/* wubu_diff_sampler.c -- LP-13: fused diffusion logits/sampler.
 *
 * The whole point of LP-13 is that fusion is numerically free.  Both entry
 * points below share the exact same per-row primitives and iteration order:
 *
 *   prep_row()  : max-subtract, expf, sum S  -> normalized-ready e[] (stable)
 *   entropy     : -sum (e_i/S) * log(e_i/S), accumulated 0..V-1
 *   stop        : entropy <= stop_entropy
 *   gumbel-max  : argmax_i ( (logit_i - m) + G_i ), G_i = -log(-log(U_i)),
 *                 U_i drawn from a deterministic per-row splitmix64 stream.
 *
 * (argmax of logit_i + G_i samples softmax(logits): the -m shift is a constant
 * across i and cancels; it is kept only for a well-scaled key.)
 *
 * Bit-exactness under -ffast-math is guaranteed structurally:
 *   - S and the entropy accumulator H are `volatile float` in BOTH paths, so
 *     no reassociation / vector reduction can reorder the 0..V-1 sum;
 *   - the RNG has a serial per-draw dependency, so draws are emitted in the
 *     same index order in both paths;
 *   - prep_row() is literally the same function in both paths;
 *   - the fused path merely interleaves entropy + keys + argmax into a single
 *     traversal, while the unfused reference spreads them over three loops.
 *
 * Pure C11, no stubs, no third-party.
 */
#include "wubu_diff_sampler.h"

#include <math.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Deterministic RNG: splitmix64 -> float uniform in the OPEN interval (0,1). */
/* ------------------------------------------------------------------ */
static uint64_t wubu_diff_mix(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

static void wubu_diff_seed(uint64_t base, int row, uint64_t *s)
{
    /* Each row owns an independent, reproducible draw stream. */
    *s = base + (uint64_t)(row + 1) * 0x9E3779B97F4A7C15ULL;
}

static float wubu_diff_uniform(uint64_t *s)
{
    uint64_t z;
    uint32_t m24;
    *s += 0x9E3779B97F4A7C15ULL;
    z = wubu_diff_mix(*s);
    m24 = (uint32_t)(z >> 40);              /* top 24 bits -> [0, 2^24) */
    /* (m+0.5)/2^24 is strictly inside (0,1): -log(U) is finite and > 0. */
    return ((float)m24 + 0.5f) * (1.0f / 16777216.0f);
}

/* ------------------------------------------------------------------ */
/* Shared numeric primitives.                                          */
/* ------------------------------------------------------------------ */

/* Stable softmax preparation: m = max(logits), e[i] = exp(logit_i - m),
 * S = sum e[i].  Identical in fused and unfused paths. */
static void wubu_diff_prep_row(const float *lg, int v,
                               float *e, float *m_out, float *S_out)
{
    float m = lg[0];
    volatile float S = 0.0f;
    int i;

    for (i = 1; i < v; ++i) {
        if (lg[i] > m) m = lg[i];
    }
    for (i = 0; i < v; ++i) {
        float a = lg[i] - m;
        float ex = expf(a);
        e[i] = ex;
        S += ex;               /* volatile: strictly serial, 0..v-1 */
    }
    *m_out = m;
    *S_out = S;
}

/* Gumbel-max key for one element, from the shifted logit a = logit_i - m. */
static float wubu_diff_gumbel_key(float a, uint64_t *s)
{
    float u = wubu_diff_uniform(s);
    float g = -logf(-logf(u));
    return a + g;
}

int wubu_diff_sampler_validate(const wubu_diff_sampler_config_t *cfg,
                               const float *logits,
                               const wubu_diff_result_t *out)
{
    if (cfg == NULL || logits == NULL || out == NULL) return WUBU_DIFF_ERR_ARG;
    if (cfg->canvas_rows <= 0 || cfg->vocab <= 0) return WUBU_DIFF_ERR_ARG;
    return WUBU_DIFF_OK;
}

/* ------------------------------------------------------------------ */
/* FUSED: entropy + stop + Gumbel sample in one combined per-row pass.  */
/* ------------------------------------------------------------------ */
static void wubu_diff_fused_row(const float *lg, int v, float stop_h,
                                uint64_t row_seed, float *e,
                                wubu_diff_result_t *out)
{
    float m, S;
    volatile float H = 0.0f;
    uint64_t s = row_seed; /* caller already mixed base seed with row index */
    int i, best = 0;
    float best_key = 0.0f;

    wubu_diff_prep_row(lg, v, e, &m, &S);

    for (i = 0; i < v; ++i) {
        float p = e[i] / S;
        float key;
        if (p > 0.0f) H -= p * logf(p);
        key = wubu_diff_gumbel_key(lg[i] - m, &s);
        if (i == 0 || key > best_key) {
            best_key = key;
            best = i;
        }
    }
    out->entropy = H;
    out->stopped = (H <= stop_h) ? 1 : 0;
    out->token = best;
    out->prob = e[best] / S;
}

int wubu_diff_sample_fused(const wubu_diff_sampler_config_t *cfg,
                           const float *logits,
                           wubu_diff_result_t *out)
{
    float *e;
    int r, v;

    if (wubu_diff_sampler_validate(cfg, logits, out) != WUBU_DIFF_OK)
        return WUBU_DIFF_ERR_ARG;

    v = cfg->vocab;
    e = (float *)malloc((size_t)v * sizeof(float));
    if (e == NULL) return WUBU_DIFF_ERR_NOMEM;

    for (r = 0; r < cfg->canvas_rows; ++r) {
        uint64_t seed;
        wubu_diff_seed(cfg->seed, r, &seed);
        wubu_diff_fused_row(logits + (size_t)r * (size_t)v, v,
                            cfg->stop_entropy, seed, e, &out[r]);
    }
    free(e);
    return WUBU_DIFF_OK;
}

/* ------------------------------------------------------------------ */
/* UNFUSED REFERENCE: separate softmax / entropy / stop / sample passes. */
/* ------------------------------------------------------------------ */

/* Pass 2: entropy over the normalized distribution. */
static float wubu_diff_ref_entropy(const float *e, float S, int v)
{
    volatile float H = 0.0f;
    int i;
    for (i = 0; i < v; ++i) {
        float p = e[i] / S;
        if (p > 0.0f) H -= p * logf(p);
    }
    return H;
}

/* Pass 4: Gumbel-max sample; returns the token, writes its probability. */
static int wubu_diff_ref_sample(const float *lg, float m,
                                const float *e, float S, int v,
                                uint64_t seed, float *prob_out)
{
    uint64_t s = seed;
    int i, best = 0;
    float best_key = 0.0f;

    for (i = 0; i < v; ++i) {
        float key = wubu_diff_gumbel_key(lg[i] - m, &s);
        if (i == 0 || key > best_key) {
            best_key = key;
            best = i;
        }
    }
    *prob_out = e[best] / S;
    return best;
}

int wubu_diff_sample_unfused(const wubu_diff_sampler_config_t *cfg,
                             const float *logits,
                             wubu_diff_result_t *out)
{
    float *e;
    int r, v;

    if (wubu_diff_sampler_validate(cfg, logits, out) != WUBU_DIFF_OK)
        return WUBU_DIFF_ERR_ARG;

    v = cfg->vocab;
    e = (float *)malloc((size_t)v * sizeof(float));
    if (e == NULL) return WUBU_DIFF_ERR_NOMEM;

    for (r = 0; r < cfg->canvas_rows; ++r) {
        const float *lg = logits + (size_t)r * (size_t)v;
        float m, S, H, prob;
        uint64_t seed;
        int token;

        /* Pass 1: softmax preparation (shared with fused). */
        wubu_diff_prep_row(lg, v, e, &m, &S);
        /* Pass 2: entropy. */
        H = wubu_diff_ref_entropy(e, S, v);
        /* Pass 3: stopping decision. */
        out[r].stopped = (H <= cfg->stop_entropy) ? 1 : 0;
        /* Pass 4: Gumbel-max sampling. */
        wubu_diff_seed(cfg->seed, r, &seed);
        token = wubu_diff_ref_sample(lg, m, e, S, v, seed, &prob);

        out[r].entropy = H;
        out[r].token = token;
        out[r].prob = prob;
    }
    free(e);
    return WUBU_DIFF_OK;
}
