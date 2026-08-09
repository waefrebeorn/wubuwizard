/* wubu_router.h -- THE ROUTER SLOT (Revolver Doctrine, THEORY/06).
 *
 * WuBu has many routing physics: the gravity field (AN12 polar orbits),
 * the Poincaré centroid router (moe_hyperbolic), the ecosystem potential
 * wells (THEORY/10). THE SLOT PRECEDES THE MODULE: one interface, any
 * physics can register into it. Consumers probe the live router -- they
 * never hardcode which physics is active.
 *
 *   route(x) -> top-K indices + weights   (the physics selects the K)
 *   grow/shrink/specialize                (the lifecycle, if supported)
 *   active/total params                   (the small-active/huge-total meter)
 *
 * C11, opaque. A router is a (ops, ctx) pair; the ops vtable is the
 * contract, the ctx is the physics instance. Minimal includes: no
 * engine headers here -- routers are self-contained physics modules.
 */
#ifndef WUBU_ROUTER_H
#define WUBU_ROUTER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- the vtable (the contract every physics implements) ---- */

typedef struct wubu_router wubu_router_t;

struct wubu_router {
    const char *name;             /* "gravity", "poincare", "ecosystem" */
    void *ctx;                    /* the physics instance (opaque) */

    /* Route an input x [dim] to the K deepest wells. out_idx[k] = cell
     * index, out_w[k] = normalized weight. Returns 0 on success. */
    int  (*route)(void *ctx, const float *x, int dim, int k,
                  int *out_idx, float *out_w);

    /* Optional full forward through the fired cells (compose). If NULL,
     * callers use route() only. */
    int  (*forward)(void *ctx, const float *x, int dim, float *out);

    /* Lifecycle (may be NULL when the physics has no grow/shrink). */
    int  (*grow)(void *ctx, int parent);
    int  (*shrink)(void *ctx, int idx);
    void (*specialize)(void *ctx, float drift);

    /* Accounting. */
    uint64_t (*active_params)(const void *ctx);
    uint64_t (*total_params)(const void *ctx);

    /* Destructor (may be NULL for static/borrowed contexts). */
    void (*free)(void *ctx);
};

/* ---- the registry: name -> router ---- */

/* Register a router under a name (idempotent: replacing an existing
 * name frees the old ctx via its free fn). Returns 0 on success. */
int wubu_router_register(wubu_router_t *router);

/* Look up a router by name. Returns NULL if not registered. */
wubu_router_t *wubu_router_get(const char *name);

/* The registered router count + names (for probing / listing). */
size_t wubu_router_count(void);
const char *wubu_router_name_at(size_t i);

/* Convenience: route through a named router (NULL-safe). Returns 0 on
 * success, -1 if the router or its route op is missing. */
int wubu_router_route_named(const char *name, const float *x, int dim, int k,
                            int *out_idx, float *out_w);

/* Register the built-in physics routers ("ecosystem", "poincare",
 * "gravity") under the slot. n_balls = colony size, dim = tangent
 * space dim, k_active = fired per input. Returns 0 on success. */
int wubu_router_register_physics(int n_balls, int dim, int k_active,
                                 uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ROUTER_H */
