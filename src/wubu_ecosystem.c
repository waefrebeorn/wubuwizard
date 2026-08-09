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
 * C11. Depends on wubu_mobius (proven Poincaré ops) + wubu_hive.
 */
#include "wubu_ecosystem.h"
#include "wubu_mobius.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

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

static int ball_cmp_desc(const void *a, const void *b) {
    const struct { int idx; float w; } *A = a, *B = b;
    return (A->w < B->w) - (A->w > B->w);
}

/* ---- colony lifecycle --------------------------------------------- */

int wubu_ecosystem_init(wubu_ecosystem_t *eco, int n_balls,
                        int k_active, int dim, uint64_t seed) {
    if (!eco || n_balls < 2 || n_balls > ECOSYSTEM_MAX_BALLS ||
        k_active < 1 || k_active > n_balls || dim < 2 || dim > ECOSYSTEM_MAX_DIM)
        return -1;

    memset(eco, 0, sizeof(*eco));
    rng_state = seed ? seed : 0xDEADBEEFCAFEBABEull;

    eco->n_balls = n_balls;
    eco->k_active = k_active;
    eco->dim = dim;
    eco->step = 0;
    eco->active_params = 0;
    eco->total_params = 0;

    if (wubu_hive_init(&eco->hive) != 0) return -1;

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
        /* small curvature-to-radius normalization: the well radius ~ R */
        b->params = (uint64_t)(7 * dim * dim) + 1;
        eco->total_params += b->params;
        b->alive = true;
        b->slot = k;
        wubu_hive_insert(&eco->hive, &eco->balls[k]);
    }
    return 0;
}

void wubu_ecosystem_free(wubu_ecosystem_t *eco) {
    if (!eco) return;
    wubu_hive_clear(&eco->hive);
    memset(eco, 0, sizeof(*eco));
}

/* ---- physics router ------------------------------------------------ */

int wubu_ecosystem_route(const wubu_ecosystem_t *eco, const float *x,
                         int *out_idx, float *out_w) {
    if (!eco || !x || !out_idx || !out_w) return -1;

    /* Potential wells: deeper well = smaller phi. Only LIVE balls rank;
     * dead (apoptosis) slots are skipped entirely. O(N) closed-form
     * potential evals -- cheaper than a learned router's softmax over
     * N experts with N*dim learned weights. */
    struct { int idx; float w; } ranked[ECOSYSTEM_MAX_BALLS];
    int n_live = 0;
    for (int k = 0; k < ECOSYSTEM_MAX_BALLS; k++) {
        const wubu_ecosystem_ball_t *b = &eco->balls[k];
        if (!b->alive) continue;
        ranked[n_live].idx = k;
        ranked[n_live].w = -ball_potential(b, x, eco->dim);  /* deep well = high score */
        n_live++;
    }
    if (n_live < eco->k_active) return -1;

    /* Top-K by well depth (no learned router -- pure physics). */
    qsort(ranked, (size_t)n_live, sizeof(ranked[0]), ball_cmp_desc);

    /* Softmax over the K well depths for the composition weights. */
    float maxw = ranked[0].w;
    float sum = 0.0f;
    for (int k = 0; k < eco->k_active; k++) {
        float e = expf(ranked[k].w - maxw);
        ranked[k].w = e;
        sum += e;
    }
    for (int k = 0; k < eco->k_active; k++) {
        out_idx[k] = ranked[k].idx;
        out_w[k] = ranked[k].w / (sum + 1e-30f);
    }
    return 0;
}

int wubu_ecosystem_forward(wubu_ecosystem_t *eco, const float *x, float *out) {
    if (!eco || !x || !out) return -1;
    const int dim = eco->dim;

    int idx[ECOSYSTEM_MAX_BALLS];
    float w[ECOSYSTEM_MAX_BALLS];
    if (wubu_ecosystem_route(eco, x, idx, w) != 0) return -1;

    /* Fired balls compose via Möbius addition of their lifted,
     * weight-scaled contributions, then project back to tangent. */
    float acc[ECOSYSTEM_MAX_DIM];
    memset(acc, 0, sizeof(acc));
    eco->active_params = 0;

    for (int k = 0; k < eco->k_active; k++) {
        wubu_ecosystem_ball_t *b = &eco->balls[idx[k]];
        float R = 1.0f / sqrtf(b->curvature + 1e-6f);
        float lifted[ECOSYSTEM_MAX_DIM];
        float scaled[ECOSYSTEM_MAX_DIM];
        float tmp[ECOSYSTEM_MAX_DIM];

        wubu_exp_map(x, dim, R, lifted);
        /* contribution = w_k ⊗ (lifted input, gyro-rotated toward center) */
        for (int i = 0; i < dim; i++) lifted[i] += 0.05f * b->center[i];
        wubu_mobius_scalar_mul(w[k], lifted, dim, R, scaled);
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
        /* all slots live: n_balls is the count of used slots */
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
    for (int k = 0; k < eco->n_balls; k++) {
        wubu_ecosystem_ball_t *b = &eco->balls[k];
        if (!b->alive) continue;
        b->curvature *= 1.0f + drift * rng_signed();
        if (b->curvature < 0.3f) b->curvature = 0.3f;
        if (b->curvature > 3.0f) b->curvature = 3.0f;
    }
}
