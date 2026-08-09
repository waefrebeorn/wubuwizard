/* wubu_ecosystem.h -- the Ecosystem of Spheres (THEORY/10).
 *
 * WuBu is not one model; it is a colony of small hyperbolic balls, each a
 * miniature expert with learnable curvature, interacting through physics:
 * the input is "attracted" by gravitational potential wells, and only the
 * spheres whose wells it falls into fire their parameters.
 *
 *   - total params can be enormous; active params stay tiny (DeepSeek V3:
 *     671B total / 37B active = 5.5%; our target ~12%)
 *   - the router is ZERO-parameter: closed-form gyro-potential wells
 *     evaluated on the existing geometry, no learned gate weights
 *   - the hive IS the ecosystem: grow (mitosis), shrink (apoptosis),
 *     specialize (curvature drift = ecological niches)
 *
 * C11, opaque-free (the colony is the seam). Depends on wubu_mobius
 * (proven Poincaré ops), wubu_hive (lifecycle), and libm.
 */
#ifndef WUBU_ECOSYSTEM_H
#define WUBU_ECOSYSTEM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "wubu_hive.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Colony sizing. N balls is the TOTAL (storage); K_active is the number
 * fired per input (compute). The active ratio K_active/N is the design
 * lever -- physics (not a learned router) sets which K. */
#define ECOSYSTEM_MAX_BALLS   256
#define ECOSYSTEM_MAX_DIM     512   /* tangent-space dim (probe-aligned) */
#define ECOSYSTEM_DEF_ACTIVE  4     /* K default: 4/256 = 1.6% active */

/* One sphere in the colony. The "expert core" is the set of params this
 * ball contributes when it fires; the router only ever touches the fired
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

/* The colony. n_balls = live balls; the array may hold dead (apoptosis)
 * entries between live ones (hive slots are stable; dead ones are
 * skipped by the router and recycled by grow). */
typedef struct {
    int n_balls;              /* live balls */
    int k_active;             /* fired per input */
    int dim;                  /* tangent dimension (runtime, revolver) */
    uint64_t step;            /* tick counter (utilization window) */
    uint64_t active_params;   /* sum of fired balls' params last forward */
    uint64_t total_params;    /* sum over ALL balls (the huge-total) */
    wubu_ecosystem_ball_t balls[ECOSYSTEM_MAX_BALLS];
    wubu_hive_t hive;         /* the lifecycle: grow/shrink recycles slots */
} wubu_ecosystem_t;

/* Init a colony: n_balls spheres in dim space, K_active fired per input.
 * Centers are seeded spread on the ball; curvatures randomized in
 * [0.5, 2.0]; per-ball params = dim*(q+k+v+o+g+u+d) estimate. */
int  wubu_ecosystem_init(wubu_ecosystem_t *eco, int n_balls,
                         int k_active, int dim, uint64_t seed);

/* ---- the physics router (zero learned parameters) ---- */

/* Potential well depth of ball k at input x:
 *   x_ball = exp_0^{c_k}(x)  (lift input into ball k's tangent)
 *   phi_k  = gyro-dist(x_ball, center_k)^2 * curvature_k
 * The input is "attracted" to the balls whose regions it resembles.
 * Returns the K_active balls with the deepest wells; out_idx[k] = ball
 * index, out_w[k] = softmax-normalized well weights. */
int wubu_ecosystem_route(const wubu_ecosystem_t *eco, const float *x,
                         int *out_idx, float *out_w);

/* Full physics forward for one token:
 *   fired balls contribute w_k * exp_0^{c_k}(proj_k(x)) via Möbius addition,
 *   then log back to the tangent space.
 * out: [dim]. Fills eco->active_params with the fired balls' param sum. */
int wubu_ecosystem_forward(wubu_ecosystem_t *eco, const float *x, float *out);

/* Active-parameter accounting (the small-active/huge-total meter). */
uint64_t wubu_ecosystem_active_params(const wubu_ecosystem_t *eco);
uint64_t wubu_ecosystem_total_params(const wubu_ecosystem_t *eco);

/* ---- the lifecycle (the hive IS the ecosystem) ---- */

/* GROW: mitosis -- a new ball splits from parent (curvature inherited,
 * center perturbed along the parent's gradient, params half each).
 * Returns the new ball index, or -1 at capacity. */
int wubu_ecosystem_grow(wubu_ecosystem_t *eco, int parent);

/* SHRINK: apoptosis -- a dead ball (low fire_count, stale) is erased;
 * its hive slot returns to the freelist. Returns 0 on success. */
int wubu_ecosystem_shrink(wubu_ecosystem_t *eco, int idx);

/* SPECIATE: drift curvatures apart (ecological niches). */
void wubu_ecosystem_specialize(wubu_ecosystem_t *eco, float drift);

/* Free all colony resources (hive blocks). */
void wubu_ecosystem_free(wubu_ecosystem_t *eco);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ECOSYSTEM_H */
