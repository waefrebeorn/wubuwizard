/* wubu_ecosystem.c -- the Ecosystem of Spheres (THEORY/10).
 *
 * The colony: many small hyperbolic balls, each a miniature expert with
 * learnable curvature. Physics routes -- the input is attracted by each
 * ball's gravitational potential well (gyro-distance), and only the K
 * balls with the deepest wells fire their parameters. The router has
 * ZERO learned parameters: it is closed-form geometry evaluated on the
 * existing ball centers/curvatures.
 *
 *   active_params / total_params  is the design lever
 *   (DeepSeek V3: 671B/37B = 5.5%; WuBu1 target ~12%)
 *
 * The hive IS the ecosystem: grow (mitosis), shrink (apoptosis),
 * specialize (curvature drift) recycle hive slots.
 *
 * ADR-002: the colony struct is opaque (defined here, not the header).
 * Reuses wubu_mobius (proven Poincaré ops) + wubu_hive (lifecycle) +
 * topk_from_array (shared selection primitive, wubu_moe_hyperbolic.h).
 * C11.
 */
#include "wubu_ecosystem.h"
#include "wubu_mobius.h"
#include "wubu_hive.h"
#include "wubu_moe_hyperbolic.h"   /* topk_from_array (shared) */

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- the opaque colony (ADR-002: layout stays in the .c) ---- */

/* One sphere in the colony (internal layout -- the header exposes only
 * the opaque handle). The "expert core" is the set of params this ball
 * contributes when it fires; the router only ever touches the fired
 * balls' cores. */
typedef struct {
    float center[ECOSYSTEM_MAX_DIM];  /* ball center in the Poincaré ball */
    float curvature;                  /* learnable c_k > 0 (1/R^2 form) */
    uint64_t params;                  /* core param count (active-cost meter) */
    uint64_t fire_count;              /* utilization for grow/shrink/specialize */
    uint64_t last_fired_step;
    int   slot;                       /* hive slot index (stable) */
    bool  alive;
} wubu_ecosystem_ball_t;

struct wubu_ecosystem {
    int n_balls;              /* live balls */
    int k_active;             /* fired per input */
    int dim;                  /* tangent dimension (runtime, revolver) */
    uint64_t step;            /* tick counter (utilization window) */
    uint64_t active_params;   /* sum of fired balls' params last forward */
    uint64_t total_params;    /* sum over ALL balls (the huge-total) */
    wubu_ecosystem_ball_t balls[ECOSYSTEM_MAX_BALLS];
    wubu_hive_t hive;         /* the lifecycle: grow/shrink recycles slots */
};

/* ---- helpers ------------------------------------------------------ */

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static float rng_uniform(void) {          /* [0,1) */
    return (float)((rng_next() >> 40) * 0x1p-24);
}
static float rng_signed(void) {           /* [-1,1) */
    return 2.0f * rng_uniform() - 1.0f;
}

/* Lift an input into ball k's Poincaré ball, then measure the gyro
 * distance to the ball center -- the potential well depth.
 * phi_k = gyro-dist(x_ball, center_k)^2 * curvature_k
 * (THEORY/10 §2: "φ_k(x) = gyro-distance² from the ball center"). */
static float ball_potential(const wubu_ecosystem_ball_t *b,
                            const float *x, int dim) {
    float R = 1.0f / sqrtf(b->curvature + 1e-6f);
    float x_ball[ECOSYSTEM_MAX_DIM];

    wubu_exp_map(x, dim, R, x_ball);
    float d = wubu_poincare_dist(b->center, x_ball, dim, R);
    return d * d * b->curvature;
}

/* ---- colony lifecycle --------------------------------------------- */

wubu_ecosystem_t *wubu_ecosystem_init(int n_balls, int k_active,
                                      int dim, uint64_t seed) {
    if (n_balls < 2 || n_balls > ECOSYSTEM_MAX_BALLS ||
        k_active < 1 || k_active > n_balls || dim < 2 || dim > ECOSYSTEM_MAX_DIM)
        return NULL;

    wubu_ecosystem_t *eco = (wubu_ecosystem_t *)calloc(1, sizeof(*eco));
    if (!eco) return NULL;
    rng_state = seed ? seed : 0xDEADBEEFCAFEBABEull;

    eco->n_balls = n_balls;
    eco->k_active = k_active;
    eco->dim = dim;

    if (wubu_hive_init(&eco->hive) != 0) { free(eco); return NULL; }

    /* Seed the colony: centers spread on the ball (roughly orthogonal
     * random directions scaled to ~0.4/R), curvatures in [0.5, 2.0],
     * params = the sphere's core (q+k+v+o+g+u+d ~ 7 * dim * dim). */
    for (int k = 0; k < n_balls; k++) {
        wubu_ecosystem_ball_t *b = &eco->balls[k];
        memset(b, 0, sizeof(*b));
        float c = 0.5f + 1.5f * rng_uniform();
        b->curvature = c;
        float R = 1.0f / sqrtf(c + 1e-6f);
        float nrm = 0.0f;
        for (int i = 0; i < dim; i++) {
            b->center[i] = rng_signed();
            nrm += b->center[i] * b->center[i];
        }
        nrm = sqrtf(nrm) + 1e-6f;
        float scale = 0.35f * R / nrm;   /* keep centers inside the ball */
        for (int i = 0; i < dim; i++) b->center[i] *= scale;
        b->params = (uint64_t)(7 * dim * dim) + 1;
        eco->total_params += b->params;
        b->alive = true;
        b->slot = k;
        wubu_hive_insert(&eco->hive, &eco->balls[k]);
    }
    return eco;
}

void wubu_ecosystem_free(wubu_ecosystem_t *eco) {
    if (!eco) return;
    wubu_hive_clear(&eco->hive);
    free(eco);
}

/* ---- physics router ------------------------------------------------ */

int wubu_ecosystem_route(const wubu_ecosystem_t *eco, const float *x,
                         int k, int *out_idx, float *out_w) {
    if (!eco || !x || !out_idx || !out_w || k < 1) return -1;
    if (k > eco->k_active) k = eco->k_active;

    /* Potential wells: deeper well = smaller phi. Only LIVE balls rank;
     * dead (apoptosis) slots are skipped entirely. O(N) closed-form
     * potential evals -- cheaper than a learned router's softmax over
     * N experts with N*dim learned weights. */
    float wells[ECOSYSTEM_MAX_BALLS];
    int n_live = 0;
    for (int s = 0; s < ECOSYSTEM_MAX_BALLS; s++) {
        const wubu_ecosystem_ball_t *b = &eco->balls[s];
        if (!b->alive) continue;
        wells[n_live] = -ball_potential(b, x, eco->dim);  /* deep well = high score */
        n_live++;
    }
    if (n_live < k) return -1;

    /* Top-K by well depth (no learned router -- pure physics). The
     * selection primitive is shared with the Poincaré router. */
    int idx[ECOSYSTEM_MAX_BALLS];
    float val[ECOSYSTEM_MAX_BALLS];
    topk_from_array(wells, n_live, k, idx, val);

    /* Softmax over the K well depths for the composition weights. */
    float maxw = val[0];
    float sum = 0.0f;
    for (int t = 0; t < k; t++) {
        float e = expf(val[t] - maxw);
        val[t] = e;
        sum += e;
    }
    /* topk_from_array works on the LIVE sub-array; map back to real
     * ball indices via the skip walk. */
    int live_seen = 0;
    int real_idx[ECOSYSTEM_MAX_BALLS];
    for (int s = 0; s < ECOSYSTEM_MAX_BALLS && live_seen < n_live; s++) {
        if (eco->balls[s].alive) real_idx[live_seen++] = s;
    }
    for (int t = 0; t < k; t++) {
        out_idx[t] = real_idx[idx[t]];
        out_w[t] = val[t] / (sum + 1e-30f);
    }
    return 0;
}

int wubu_ecosystem_forward(wubu_ecosystem_t *eco, const float *x, float *out) {
    if (!eco || !x || !out) return -1;
    const int dim = eco->dim;

    int idx[ECOSYSTEM_MAX_BALLS];
    float w[ECOSYSTEM_MAX_BALLS];
    if (wubu_ecosystem_route(eco, x, eco->k_active, idx, w) != 0) return -1;

    /* Fired balls compose via Möbius addition of their lifted,
     * weight-scaled contributions, then project back to tangent. */
    float acc[ECOSYSTEM_MAX_DIM];
    memset(acc, 0, sizeof(acc));
    eco->active_params = 0;

    for (int t = 0; t < eco->k_active; t++) {
        wubu_ecosystem_ball_t *b = &eco->balls[idx[t]];
        float R = 1.0f / sqrtf(b->curvature + 1e-6f);
        float lifted[ECOSYSTEM_MAX_DIM];
        float scaled[ECOSYSTEM_MAX_DIM];
        float tmp[ECOSYSTEM_MAX_DIM];

        wubu_exp_map(x, dim, R, lifted);
        /* contribution = w_k ⊗ (lifted input, gyro-rotated toward center) */
        for (int i = 0; i < dim; i++) lifted[i] += 0.05f * b->center[i];
        wubu_mobius_scalar_mul(w[t], lifted, dim, R, scaled);
        wubu_mobius_add(acc, scaled, dim, R, tmp);
        memcpy(acc, tmp, (size_t)dim * sizeof(float));

        b->fire_count++;
        b->last_fired_step = eco->step;
        eco->active_params += b->params;
    }

    /* Log back to the tangent space for the next layer. */
    float R0 = 1.0f / sqrtf(1.0f + 1e-6f);
    wubu_log_map(acc, dim, R0, out);
    eco->step++;
    return 0;
}

uint64_t wubu_ecosystem_active_params(const wubu_ecosystem_t *eco) {
    return eco ? eco->active_params : 0;
}
uint64_t wubu_ecosystem_total_params(const wubu_ecosystem_t *eco) {
    return eco ? eco->total_params : 0;
}

int wubu_ecosystem_count(const wubu_ecosystem_t *eco) {
    return eco ? eco->n_balls : 0;
}
int wubu_ecosystem_dim(const wubu_ecosystem_t *eco) {
    return eco ? eco->dim : 0;
}
int wubu_ecosystem_k_active(const wubu_ecosystem_t *eco) {
    return eco ? eco->k_active : 0;
}

/* ---- lifecycle: the hive IS the ecosystem -------------------------- */

int wubu_ecosystem_grow(wubu_ecosystem_t *eco, int parent) {
    if (!eco || parent < 0 || parent >= ECOSYSTEM_MAX_BALLS) return -1;
    const wubu_ecosystem_ball_t *p = &eco->balls[parent];
    if (!p->alive) return -1;
    if (eco->n_balls >= ECOSYSTEM_MAX_BALLS) return -1;

    /* Mitosis: find a dead slot to recycle (apoptosis freed it), else
     * append at the end. The hive slot is stable; the hive recycles. */
    int child = -1;
    for (int k = 0; k < ECOSYSTEM_MAX_BALLS; k++)
        if (!eco->balls[k].alive) { child = k; break; }
    if (child < 0) {
        if (eco->n_balls >= ECOSYSTEM_MAX_BALLS) return -1;
        child = eco->n_balls;
    }

    wubu_ecosystem_ball_t *c = &eco->balls[child];
    memset(c, 0, sizeof(*c));

    /* Mitosis: curvature inherited, center perturbed along a random
     * direction, params split from the parent (function-preserving-ish:
     * total colony params stay the same -- one cell splits in two). */
    c->curvature = p->curvature * (0.9f + 0.2f * rng_uniform());
    float R = 1.0f / sqrtf(c->curvature + 1e-6f);
    float nrm = 0.0f;
    float dir[ECOSYSTEM_MAX_DIM];
    for (int i = 0; i < eco->dim; i++) {
        dir[i] = rng_signed();
        nrm += dir[i] * dir[i];
    }
    nrm = sqrtf(nrm) + 1e-6f;
    float step = 0.12f * R / nrm;
    for (int i = 0; i < eco->dim; i++) {
        c->center[i] = p->center[i] + step * dir[i];
        if (c->center[i] > 0.9f * R) c->center[i] = 0.9f * R;
        if (c->center[i] < -0.9f * R) c->center[i] = -0.9f * R;
    }
    uint64_t half = p->params / 2;
    wubu_ecosystem_ball_t *pp = &eco->balls[parent];
    pp->params -= half;            /* parent loses half (mitosis) */
    c->params = half + 1;
    eco->total_params = 0;
    for (int k = 0; k < ECOSYSTEM_MAX_BALLS; k++)
        if (eco->balls[k].alive) eco->total_params += eco->balls[k].params;

    c->alive = true;
    c->slot = child;
    eco->n_balls++;                /* live count */
    wubu_hive_insert(&eco->hive, c);
    return child;
}

int wubu_ecosystem_shrink(wubu_ecosystem_t *eco, int idx) {
    if (!eco || idx < 0 || idx >= ECOSYSTEM_MAX_BALLS) return -1;
    wubu_ecosystem_ball_t *b = &eco->balls[idx];
    if (!b->alive) return -1;

    b->alive = false;
    wubu_hive_erase(&eco->hive, b);
    eco->total_params -= b->params;
    eco->n_balls--;                /* live count */
    return 0;
}

void wubu_ecosystem_specialize(wubu_ecosystem_t *eco, float drift) {
    if (!eco) return;
    for (int k = 0; k < ECOSYSTEM_MAX_BALLS; k++) {
        wubu_ecosystem_ball_t *b = &eco->balls[k];
        if (!b->alive) continue;
        b->curvature *= 1.0f + drift * rng_signed();
        if (b->curvature < 0.3f) b->curvature = 0.3f;
        if (b->curvature > 3.0f) b->curvature = 3.0f;
    }
}

/* ---- prefetch: slow storage hides behind compute (research/063-E) ---- */

/* The next-likely balls are the ones ranked just below the current
 * top-K with the highest fire_count (utilization): they have been
 * firing and are likely to fire again. Touching their params warms
 * the dequant cache so the next token's top-K misses nothing. */
int wubu_ecosystem_prefetch(wubu_ecosystem_t *eco, int *top_k, int k) {
    if (!eco || !top_k || k < 0) return -1;
    if (k == 0) return 0;

    /* Mark the current top-K so we don't prefetch what's already hot. */
    uint8_t hot[ECOSYSTEM_MAX_BALLS];
    memset(hot, 0, sizeof(hot));
    for (int i = 0; i < k; i++)
        if (top_k[i] >= 0 && top_k[i] < ECOSYSTEM_MAX_BALLS)
            hot[top_k[i]] = 1;

    /* The next-likely candidates: alive, not hot, highest fire_count. */
    int cand[ECOSYSTEM_MAX_BALLS];
    int n_cand = 0;
    for (int s = 0; s < ECOSYSTEM_MAX_BALLS; s++) {
        const wubu_ecosystem_ball_t *b = &eco->balls[s];
        if (!b->alive || hot[s]) continue;
        cand[n_cand++] = s;
    }
    /* partial selection sort: top-k by fire_count */
    int warmed = 0;
    for (int pick = 0; pick < k && n_cand > 0; pick++) {
        int best = 0;
        for (int i = 1; i < n_cand; i++)
            if (eco->balls[cand[i]].fire_count >
                eco->balls[cand[best]].fire_count)
                best = i;
        int b = cand[best];
        /* warm: touch the params (a real memory touch, cache-line wide) */
        wubu_ecosystem_ball_t *wb = &eco->balls[b];
        volatile float sum = 0.0f;
        for (int i = 0; i < eco->dim; i += 16) sum += wb->center[i];
        (void)sum;
        warmed++;
        cand[best] = cand[--n_cand];
    }
    return warmed;
}
