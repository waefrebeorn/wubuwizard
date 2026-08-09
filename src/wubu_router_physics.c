/* wubu_router_physics.c -- the physics routers registered into the slot.
 *
 * THE ROUTER SLOT (Revolver Doctrine): the engine probes a router by
 * name -- "ecosystem", "poincare", "gravity" -- and gets the same
 * interface back. This file registers all three physics:
 *
 *   ecosystem  -> wubu_ecosystem (THEORY/10 potential wells)
 *   poincare   -> the nested Poincaré centroid router (moe_hyperbolic)
 *   gravity    -> the polar gravity field (AN12)
 *
 * Each registration is a thin adapter: the physics keeps its own
 * opaque ctx; the vtable ops bridge to its API. Consumers never see
 * which physics is active -- they call wubu_router_route_named().
 *
 * C11. Depends on the physics modules + the registry.
 */
#include "wubu_router.h"
#include "wubu_ecosystem.h"
#include "wubu_moe_hyperbolic.h"
#include "wubu_gravity.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- ecosystem adapter ---- */

static int eco_route(void *ctx, const float *x, int dim, int k,
                     int *out_idx, float *out_w) {
    return wubu_ecosystem_route((const wubu_ecosystem_t *)ctx, x, k,
                                out_idx, out_w);
}
static int eco_forward(void *ctx, const float *x, int dim, float *out) {
    return wubu_ecosystem_forward((wubu_ecosystem_t *)ctx, x, out);
}
static int eco_grow(void *ctx, int parent) {
    return wubu_ecosystem_grow((wubu_ecosystem_t *)ctx, parent);
}
static int eco_shrink(void *ctx, int idx) {
    return wubu_ecosystem_shrink((wubu_ecosystem_t *)ctx, idx);
}
static void eco_specialize(void *ctx, float drift) {
    wubu_ecosystem_specialize((wubu_ecosystem_t *)ctx, drift);
}
static uint64_t eco_active(const void *ctx) {
    return wubu_ecosystem_active_params((const wubu_ecosystem_t *)ctx);
}
static uint64_t eco_total(const void *ctx) {
    return wubu_ecosystem_total_params((const wubu_ecosystem_t *)ctx);
}
static void eco_free(void *ctx) {
    wubu_ecosystem_free((wubu_ecosystem_t *)ctx);
}

static wubu_router_t g_eco_router;

/* ---- poincare adapter ---- */

/* The Poincaré centroid router is embedded in moe_hyperbolic; we adapt
 * a single-level poincare_router_t. Its API routes a whole batch to
 * scores; the adapter does one token at a time and top-k's the scores. */

typedef struct {
    poincare_router_t pr;
    int dim;              /* slot-side dim (the probe's tangent space) */
} poincare_adapter_t;

static void poincare_scores_to_topk(const float *scores, int n_experts,
                                    int k, int *out_idx, float *out_w) {
    /* reuse the shared selection primitive */
    int idx[256];
    float val[256];
    topk_from_array(scores, n_experts, k, idx, val);
    float sum = 0.0f;
    for (int t = 0; t < k; t++) sum += val[t];
    for (int t = 0; t < k; t++) {
        out_idx[t] = idx[t];
        out_w[t] = val[t] / (sum + 1e-30f);
    }
}

static int poincare_route(void *ctx, const float *x, int dim, int k,
                          int *out_idx, float *out_w) {
    poincare_adapter_t *pa = (poincare_adapter_t *)ctx;
    /* The centroid router is D_MODEL-native; the slot may probe at a
     * smaller dim. Zero-pad the tail so the physics sees its own space. */
    float padded[D_MODEL];
    memset(padded, 0, sizeof(padded));
    int n = dim < D_MODEL ? dim : D_MODEL;
    for (int i = 0; i < n; i++) padded[i] = x[i];

    float scores[N_EXPERTS];
    wubu_poincare_router_forward(padded, 1, 1, &pa->pr, scores);
    poincare_scores_to_topk(scores, N_EXPERTS, k, out_idx, out_w);
    return 0;
}
static uint64_t poincare_active(const void *ctx) {
    (void)ctx;
    return (uint64_t)N_ACTIVE_EXPTS;   /* k experts touched per token */
}
static uint64_t poincare_total(const void *ctx) {
    (void)ctx;
    /* centroids only: N_EXPERTS * D_MODEL weights */
    return (uint64_t)N_EXPERTS * (uint64_t)D_MODEL;
}
static void poincare_free(void *ctx) {
    poincare_adapter_t *pa = (poincare_adapter_t *)ctx;
    wubu_poincare_router_free(&pa->pr);
    free(pa);
}

static wubu_router_t g_poincare_router;

/* ---- gravity adapter ---- */

/* The gravity field routes on polar coordinates (r, theta); the adapter
 * projects the input's norm/direction onto the field's polar frame and
 * routes. One cell per route; K > 1 walks to the next-nearest orbit. */

typedef struct {
    wubu_gravity_t *g;
    int dim;
} gravity_adapter_t;

static void polar_of(const float *x, int dim, double *r, double *theta) {
    double nrm = 0.0, ang = 0.0;
    for (int i = 0; i < dim; i++) {
        double v = x[i];
        nrm += v * v;
        ang += (double)(i + 1) * v;   /* directional fingerprint */
    }
    *r = sqrt(nrm) * 0.1;             /* scale into the ball */
    *theta = fmod(fabs(ang) + 1.0, 2.0 * 3.141592653589793) ;
}

static int gravity_route(void *ctx, const float *x, int dim, int k,
                         int *out_idx, float *out_w) {
    gravity_adapter_t *ga = (gravity_adapter_t *)ctx;
    double r, theta;
    polar_of(x, dim, &r, &theta);
    int first = wubu_gravity_route(ga->g, r, theta);
    if (first < 0) return -1;
    out_idx[0] = first;
    out_w[0] = 1.0f;
    for (int t = 1; t < k; t++) { out_idx[t] = (first + t) % (int)wubu_gravity_count(ga->g); out_w[t] = 0.0f; }
    return 0;
}
static int gravity_grow(void *ctx, int parent) {
    gravity_adapter_t *ga = (gravity_adapter_t *)ctx;
    return wubu_gravity_grow(ga->g, parent, 0.05, 0.5);
}
static int gravity_shrink(void *ctx, int idx) {
    gravity_adapter_t *ga = (gravity_adapter_t *)ctx;
    return wubu_gravity_shrink(ga->g, idx);
}
static uint64_t gravity_total(const void *ctx) {
    gravity_adapter_t *ga = (gravity_adapter_t *)ctx;
    return wubu_gravity_count(ga->g) * 7 * 32 * 32;   /* core estimate */
}
static uint64_t gravity_active(const void *ctx) {
    (void)ctx;
    return 7 * 32 * 32;    /* one cell fired */
}
static void gravity_free(void *ctx) {
    gravity_adapter_t *ga = (gravity_adapter_t *)ctx;
    wubu_gravity_free(ga->g);
    free(ga);
}

static wubu_router_t g_gravity_router;

/* ---- registration ---- */

int wubu_router_register_physics(int n_balls, int dim, int k_active,
                                 uint64_t seed) {
    /* ecosystem */
    wubu_ecosystem_t *eco = wubu_ecosystem_init(n_balls, k_active, dim, seed);
    if (!eco) return -1;
    g_eco_router.name = "ecosystem";
    g_eco_router.ctx = eco;
    g_eco_router.route = eco_route;
    g_eco_router.forward = eco_forward;
    g_eco_router.grow = eco_grow;
    g_eco_router.shrink = eco_shrink;
    g_eco_router.specialize = eco_specialize;
    g_eco_router.active_params = eco_active;
    g_eco_router.total_params = eco_total;
    g_eco_router.free = eco_free;
    wubu_router_register(&g_eco_router);

    /* poincare */
    poincare_adapter_t *pa = (poincare_adapter_t *)calloc(1, sizeof(*pa));
    if (!pa) return -1;
    pa->dim = dim;
    /* The centroid router is D_MODEL-native: it inits N_EXPERTS*D_MODEL
     * floats regardless of the probe dim. Allocate its full space. */
    pa->pr.centroids = (float *)malloc((size_t)N_EXPERTS * D_MODEL * sizeof(float));
    if (!pa->pr.centroids) { free(pa); return -1; }
    wubu_poincare_router_init_random(pa->pr.centroids, (unsigned)seed);
    pa->pr.temperature = HYPERBOLIC_TEMPERATURE;
    pa->pr.loaded = true;
    g_poincare_router.name = "poincare";
    g_poincare_router.ctx = pa;
    g_poincare_router.route = poincare_route;
    g_poincare_router.forward = NULL;
    g_poincare_router.grow = NULL;
    g_poincare_router.shrink = NULL;
    g_poincare_router.specialize = NULL;
    g_poincare_router.active_params = poincare_active;
    g_poincare_router.total_params = poincare_total;
    g_poincare_router.free = poincare_free;
    wubu_router_register(&g_poincare_router);

    /* gravity */
    gravity_adapter_t *ga = (gravity_adapter_t *)calloc(1, sizeof(*ga));
    if (!ga) return -1;
    ga->dim = dim;
    wubu_gravity_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.G = 1.0;
    cfg.M = 1000.0;
    cfg.c = 1.0;
    cfg.n_cells = (size_t)n_balls;
    cfg.boot_r = 0.1;
    ga->g = wubu_gravity_init(&cfg);
    if (!ga->g) { free(ga); return -1; }
    g_gravity_router.name = "gravity";
    g_gravity_router.ctx = ga;
    g_gravity_router.route = gravity_route;
    g_gravity_router.forward = NULL;
    g_gravity_router.grow = gravity_grow;
    g_gravity_router.shrink = gravity_shrink;
    g_gravity_router.specialize = NULL;
    g_gravity_router.active_params = gravity_active;
    g_gravity_router.total_params = gravity_total;
    g_gravity_router.free = gravity_free;
    wubu_router_register(&g_gravity_router);

    return 0;
}
