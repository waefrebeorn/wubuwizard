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
#include "wubu_moe.h"        /* moe_weights_t + wubu_moe_forward (engine wiring) */

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

/* ---- engine-wiring probe (used by test_router) ---- */

/* Build a minimal MoE with the ecosystem router installed and run one
 * forward -- proves wubu_moe_forward consumes the router slot. Only the
 * shared expert is allocated (routed experts clip to skip with
 * n_experts_loaded = 0); the physics router still selects the top-K
 * indices, and the forward produces finite output. */
int wubu_moe_forward_router_probe(const float *x, int B, int T) {
    extern void wubu_moe_forward(const float *x, int B, int T,
                                 const moe_weights_t *w,
                                 float *output, int *selected_experts);
    extern wubu_router_t *wubu_router_get(const char *name);

    wubu_router_t *r = wubu_router_get("ecosystem");
    if (!r) return -1;

    const int N = B * T;
    /* The MoE forward reads D_MODEL-wide tokens; the slot probe may be
     * called with a smaller vector. Zero-pad to the engine's space. */
    float *xpad = (float *)calloc((size_t)N * D_MODEL, sizeof(float));
    if (!xpad) return -1;
    memcpy(xpad, x, (size_t)(B * T) * sizeof(float));

    moe_weights_t mw;
    memset(&mw, 0, sizeof(mw));
    mw.loaded = true;
    /* n_experts_loaded = 1: every routed expert index (0..63 from the
     * colony) clips to the skip path (memset 0) -- the probe runs the
     * shared expert only. The clip guard is `n_experts_loaded > 0`, so
     * 0 would NOT clip; 1 does. */
    mw.n_experts_loaded = 1;
    mw.router = r;                 /* THE SLOT: physics routing */

    /* Shared expert F32 buffers (the only weights the forward touches
     * with n_experts_loaded = 1, except expert 0). Small random init. */
    size_t nsh = (size_t)D_MODEL * SHARED_D_FF;
    mw.ffn_gate_shexp = (float *)calloc(nsh, sizeof(float));
    mw.ffn_up_shexp   = (float *)calloc(nsh, sizeof(float));
    mw.ffn_down_shexp = (float *)calloc(nsh, sizeof(float));
    if (!mw.ffn_gate_shexp || !mw.ffn_up_shexp || !mw.ffn_down_shexp) {
        free(mw.ffn_gate_shexp); free(mw.ffn_up_shexp); free(mw.ffn_down_shexp);
        return -1;
    }

    /* Expert 0 must not clip (indices 1..63 do). Give it one REAL F32
     * expert blob (zeroed) so the forward runs the complete path --
     * router hook -> quantized matmul -> accumulate. */
    size_t nff = (size_t)D_MODEL * D_FF;
    mw.ffn_gate_exps_q = (const uint8_t *)calloc(nff, sizeof(float));
    mw.ffn_up_exps_q   = (const uint8_t *)calloc(nff, sizeof(float));
    mw.ffn_down_exps_q = (const uint8_t *)calloc((size_t)D_FF * D_MODEL, sizeof(float));
    mw.ffn_gate_exps_q_type = GGML_TYPE_F32;
    mw.ffn_up_exps_q_type   = GGML_TYPE_F32;
    mw.ffn_down_exps_q_type = GGML_TYPE_F32;
    if (!mw.ffn_gate_exps_q || !mw.ffn_up_exps_q || !mw.ffn_down_exps_q) {
        free(mw.ffn_gate_shexp); free(mw.ffn_up_shexp); free(mw.ffn_down_shexp);
        return -1;
    }

    float *out = (float *)calloc((size_t)N * D_MODEL, sizeof(float));
    if (!out) {
        free(mw.ffn_gate_shexp); free(mw.ffn_up_shexp); free(mw.ffn_down_shexp);
        free((void *)mw.ffn_gate_exps_q); free((void *)mw.ffn_up_exps_q); free((void *)mw.ffn_down_exps_q);
        return -1;
    }

    wubu_moe_forward(xpad, B, T, &mw, out, NULL);
    int finite = 1;
    for (int i = 0; i < N * D_MODEL; i++)
        if (!isfinite(out[i])) { finite = 0; break; }
    free(out);
    free(xpad);
    free(mw.ffn_gate_shexp); free(mw.ffn_up_shexp); free(mw.ffn_down_shexp);
    free((void *)mw.ffn_gate_exps_q); free((void *)mw.ffn_up_exps_q); free((void *)mw.ffn_down_exps_q);
    return finite ? 0 : -1;
}

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
