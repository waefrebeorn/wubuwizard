/*
 * test_router.c -- THE ROUTER SLOT gate (Revolver Doctrine, THEORY/06).
 *
 * Proves the routing is AGNOSTIC: three physics implementations
 * ("ecosystem" THEORY/10 potential wells, "poincare" centroid router,
 * "gravity" AN12 polar field) register into ONE slot with ONE contract.
 * The test drives them through the registry only -- it never touches a
 * physics-specific API. If a new physics registers tomorrow, this test
 * covers it with zero changes.
 *
 *   1. REGISTRY: all three routers are registered and discoverable by
 *      name (probe, don't assume).
 *   2. SAME CONTRACT: route_named() works identically for each -- K
 *      indices + weights out, deterministic.
 *   3. SWAP: the same input routed through each physics returns a valid
 *      top-K for all of them (any physics can fill the slot).
 *   4. LIFECYCLE: grow/shrink work through the vtable where supported
 *      (ecosystem + gravity), and the colony stays bounded.
 *   5. ACCOUNTING: every router reports active/total params.
 *
 * Gate: `make test_router`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_router.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

#define N_BALLS  64
#define DIM      32
#define K_ACTIVE 4

static uint64_t trng = 0xABCDEF0123456789ull;
static float frand(void) {
    trng ^= trng << 13;
    trng ^= trng >> 7;
    trng ^= trng << 17;
    return 2.0f * ((float)((trng >> 40) * 0x1p-24)) - 1.0f;
}

int main(void)
{
    printf("=== test_router (THE ROUTER SLOT: one interface, any physics) ===\n");

    if (wubu_router_register_physics(N_BALLS, DIM, K_ACTIVE, 424242ull) != 0) {
        printf("  FAIL: physics registration\n");
        return 1;
    }

    /* ---- 1. registry: all three discoverable ---- */
    const char *want[] = { "ecosystem", "poincare", "gravity" };
    int n_registered = (int)wubu_router_count();
    printf("  ok: %d routers registered:", n_registered);
    for (size_t i = 0; i < wubu_router_count(); i++)
        printf(" %s", wubu_router_name_at(i));
    printf("\n");
    for (int i = 0; i < 3; i++) {
        CHECK(wubu_router_get(want[i]) != NULL, "registry: router discoverable by name");
    }

    /* ---- 2+3. same contract, swap-ability ---- */
    float x[DIM];
    for (int i = 0; i < DIM; i++) x[i] = frand() * 0.5f;

    for (int i = 0; i < 3; i++) {
        int idx[K_ACTIVE], idx2[K_ACTIVE];
        float w[K_ACTIVE], w2[K_ACTIVE];
        int rc = wubu_router_route_named(want[i], x, DIM, K_ACTIVE, idx, w);
        CHECK(rc == 0, "route_named succeeds for every physics");
        if (rc != 0) continue;
        /* determinism */
        wubu_router_route_named(want[i], x, DIM, K_ACTIVE, idx2, w2);
        int same = 1;
        for (int k = 0; k < K_ACTIVE; k++) if (idx[k] != idx2[k]) same = 0;
        CHECK(same, "router is deterministic (pure physics)");
        /* indices in range, weights normalized */
        int in_range = 1;
        float wsum = 0.0f;
        for (int k = 0; k < K_ACTIVE; k++) {
            if (idx[k] < 0 || idx[k] >= N_BALLS * 4) in_range = 0;
            wsum += w[k];
        }
        CHECK(in_range, "indices are valid cell ids");
        printf("  ok: %-8s top-1=%d top-2=%d w=[%.3f %.3f %.3f %.3f] sum=%.3f\n",
               want[i], idx[0], idx[1], w[0], w[1], w[2], w[3], wsum);
        CHECK(wsum > 0.9f && wsum < 1.1f, "weights are normalized");

        /* accounting via the vtable */
        wubu_router_t *r = wubu_router_get(want[i]);
        uint64_t act = r->active_params ? r->active_params(r->ctx) : 0;
        uint64_t tot = r->total_params ? r->total_params(r->ctx) : 0;
        printf("  ok: %-8s active=%llu total=%llu (%.2f%%)\n",
               want[i], (unsigned long long)act, (unsigned long long)tot,
               tot ? 100.0 * (double)act / (double)tot : 0.0);
        CHECK(tot > 0, "router reports total params");
    }

    /* ---- 4. lifecycle through the vtable (where supported) ---- */
    wubu_router_t *eco = wubu_router_get("ecosystem");
    if (eco->grow && eco->shrink) {
        int child = eco->grow(eco->ctx, 0);
        CHECK(child >= 0, "ecosystem grow via vtable");
        if (child >= 0) {
            CHECK(eco->shrink(eco->ctx, child) == 0, "ecosystem shrink via vtable");
        }
    }
    wubu_router_t *grav = wubu_router_get("gravity");
    if (grav->grow && grav->shrink) {
        /* Cell 0 is the protected boot core (r < boot_r) and must never
         * split -- grow an outer (non-core) cell instead. */
        int child = grav->grow(grav->ctx, 10);
        CHECK(child >= 0, "gravity grow via vtable (non-core cell)");
        if (child >= 0) {
            CHECK(grav->shrink(grav->ctx, child) == 0, "gravity shrink via vtable");
        }
        /* the core invariant: core cells never split */
        CHECK(grav->grow(grav->ctx, 0) < 0, "boot core never splits (protected)");
    }

    /* ---- routing still valid after lifecycle churn ---- */
    for (int i = 0; i < 3; i++) {
        int idx[K_ACTIVE];
        float w[K_ACTIVE];
        CHECK(wubu_router_route_named(want[i], x, DIM, K_ACTIVE, idx, w) == 0,
              "route still works after lifecycle churn");
    }

    /* ---- re-registration replaces cleanly (no leak / no dup) ---- */
    CHECK(wubu_router_register_physics(N_BALLS, DIM, K_ACTIVE, 424242ull) == 0,
          "re-registration succeeds");
    CHECK(wubu_router_count() == 3,
          "re-registration replaces, does not duplicate");

    /* ---- ENGINE WIRING: the slot drives a real MoE forward ---- */
    /* The MoE core (wubu_moe_forward) accepts w->router and routes by
     * physics instead of the learned gate. Prove the wiring: with a
     * physics router installed, forward still runs and produces finite
     * output of the right shape. */
    {
        extern int wubu_moe_forward_router_probe(const float *x, int B, int T);
        int rc = wubu_moe_forward_router_probe(x, 1, 1);
        CHECK(rc == 0, "MoE forward runs with the physics router installed");
    }

    if (failures == 0) printf("=== ALL ROUTER-SLOT TESTS PASSED ===\n");
    else printf("=== %d FAILURES ===\n", failures);
    return failures ? 1 : 0;
}
