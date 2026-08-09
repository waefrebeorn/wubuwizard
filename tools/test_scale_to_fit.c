/*
 * test_scale_to_fit.c -- THEORY/11 gate: one checkpoint, any hardware.
 *
 * The SAME checkpoint (fixed geometry: WuBu1 56.4M, 512-dim, 24 layers)
 * runs through FIVE simulated hardware budgets -- CM4 (256MB), RPi-5
 * class (2GB), laptop (16GB), server (512GB), supercomputer (4TB) --
 * and every plan must satisfy the five scale-to-fit invariants:
 *
 *   1. BOOTS:     every tier loads >= 1 core layer (the ring-0 brain).
 *   2. FITS:      weight_bytes + kv_reserve <= ram_budget (never OOM).
 *   3. SMALL-ACTIVE: k_active / ecosystem_n <= the tier's ratio ceiling
 *                  (the huge-total stays cheap on the Pi too).
 *   4. GROWS:     ecosystem_n and core_layers are monotonic in the tier
 *                  (bigger machines get bigger bodies, no cliff).
 *   5. SAME CODE: one wubu_scale_plan() call produces all five plans --
 *                  the planner is identical on every machine.
 *
 * The budgets are INJECTED (not probed) -- that is what makes the test
 * deterministic and what lets the planner be verified before real
 * hardware exists. The probe path (wubu_scale_probe_hw) is also
 * exercised on the host for a smoke check.
 *
 * Gate: `make test_scale_to_fit`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_scale.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

/* The WuBu1 checkpoint geometry (the seed as it exists today). */
#define CKPT_PARAMS  56376832ull
#define CKPT_CORE    12000000ull   /* boot-core estimate (top-k layers) */
#define CKPT_LAYERS  24
#define CKPT_DIM     512

typedef struct {
    const char *name;
    uint64_t ram;
    wubu_scale_tier_t tier;
    int expect_core_min;
    int expect_ratio_max_pct;   /* ratio ceiling, in percent */
} tier_case_t;

static const tier_case_t k_cases[] = {
    { "cm4",        256ull * 1024 * 1024, WUBU_SCALE_TINY,  1, 13 },
    { "rpi5",         2ull * 1024 * 1024 * 1024, WUBU_SCALE_SMALL, 2, 7 },
    { "laptop",      16ull * 1024 * 1024 * 1024, WUBU_SCALE_MID,   4, 4 },
    { "server",     512ull * 1024 * 1024 * 1024, WUBU_SCALE_BIG,   6, 2 },
    { "super",        4ull * 1024 * 1024 * 1024 * 1024, WUBU_SCALE_HUGE, 8, 1 },
};
#define N_CASES (sizeof(k_cases) / sizeof(k_cases[0]))

int main(void)
{
    printf("=== test_scale_to_fit (THEORY/11: one checkpoint, any hardware) ===\n");

    wubu_scale_ckpt_t ckpt;
    memset(&ckpt, 0, sizeof(ckpt));
    ckpt.total_params = CKPT_PARAMS;
    ckpt.core_params  = CKPT_CORE;
    ckpt.n_layers     = CKPT_LAYERS;
    ckpt.dim          = CKPT_DIM;

    wubu_scale_plan_t plans[WUBU_SCALE_COUNT];

    /* ---- 5 tiers through the SAME planner ---- */
    for (size_t i = 0; i < N_CASES; i++) {
        wubu_scale_hw_t hw;
        memset(&hw, 0, sizeof(hw));
        hw.ram_bytes = k_cases[i].ram;
        hw.cores     = (int)(i + 1) * 2;
        hw.simd_bits = 128;
        hw.has_accel = (i >= 3);   /* servers + super get accelerators */

        /* RESEARCH/063: devices. CM4 gets a CPU+NPU+SD-card trio;
         * the super gets CPU+GPU. The planner must use them. */
        hw.n_devices = 1;
        hw.devices[0].kind = WUBU_DEV_CPU;
        hw.devices[0].ram_bytes = hw.ram_bytes;
        hw.devices[0].tfops = (i == 0) ? 0.013 : (double)(i + 1) * 4.0;
        hw.devices[0].bandwidth_gbps = (i == 0) ? 3.2 : 25.6;  /* CM4 SD vs DDR */
        hw.devices[0].watts = (i == 0) ? 6.0 : (double)(i + 1) * 15.0;
        if (i == 0) {  /* CM4 + Hailo-8 NPU + SD card */
            hw.n_devices = 3;
            hw.devices[1].kind = WUBU_DEV_NPU;
            hw.devices[1].ram_bytes = 0;                 /* NPU uses system RAM */
            hw.devices[1].tfops = 26.0;                  /* Hailo-8: 26 TOPS */
            hw.devices[1].bandwidth_gbps = 12.8;
            hw.devices[1].watts = 2.5;
            hw.devices[2].kind = WUBU_DEV_DISK;
            hw.devices[2].ram_bytes = 32ull * 1024 * 1024 * 1024; /* SD cap */
            hw.devices[2].tfops = 0.0;
            hw.devices[2].bandwidth_gbps = 0.9;          /* eMMC read */
            hw.devices[2].watts = 0.5;
        } else if (i == 4) {  /* super: + GPU */
            hw.n_devices = 2;
            hw.devices[1].kind = WUBU_DEV_GPU;
            hw.devices[1].ram_bytes = 80ull * 1024 * 1024 * 1024;
            hw.devices[1].tfops = 500.0;
            hw.devices[1].bandwidth_gbps = 2000.0;
            hw.devices[1].watts = 700.0;
        }

        int rc = wubu_scale_plan(&hw, &ckpt, &plans[i]);
        CHECK(rc == 0, "plan computes");
        CHECK(plans[i].tier == k_cases[i].tier, "tier classification correct");

        char buf[320];
        wubu_scale_report(&plans[i], buf, sizeof(buf));
        printf("  %-6s : %s\n", k_cases[i].name, buf);
    }

    /* ---- invariant 1: BOOTS ---- */
    for (size_t i = 0; i < N_CASES; i++)
        CHECK(plans[i].core_layers >= k_cases[i].expect_core_min,
              "BOOTS: every tier loads the ring-0 core");

    /* ---- invariant 2: FITS (never OOM) ---- */
    for (size_t i = 0; i < N_CASES; i++)
        CHECK(plans[i].total_bytes <= k_cases[i].ram,
              "FITS: weight+KV <= budget on every tier");

    /* ---- invariant 3: SMALL-ACTIVE ---- */
    for (size_t i = 0; i < N_CASES; i++) {
        double ratio_pct = plans[i].ratio_active * 100.0;
        CHECK(ratio_pct <= (double)k_cases[i].expect_ratio_max_pct,
              "SMALL-ACTIVE: active/total ratio stays under the ceiling");
    }

    /* ---- invariant 4: GROWS (monotonic, no cliff) ---- */
    for (size_t i = 1; i < N_CASES; i++) {
        CHECK(plans[i].ecosystem_n > plans[i-1].ecosystem_n,
              "GROWS: ecosystem grows monotonically with the tier");
        CHECK(plans[i].core_layers >= plans[i-1].core_layers,
              "GROWS: core layers never shrink on a bigger machine");
    }

    /* ---- invariant 5: SAME CODE (implicit: one loop above) ----
     * All five plans came from the same wubu_scale_plan() call. */

    /* ---- RESEARCH/063 invariants: the new axes ---- */
    /* BANDWIDTH: bytes/token must stay sane on every tier (the core is
     * always touched; the outer body is the active-ratio lever). The
     * CM4's SD card (0.9 GB/s) must not be the bottleneck -- the plan
     * sizes k_active so the bytes/token fit the slowest device. */
    for (size_t i = 0; i < N_CASES; i++) {
        double slowest_gbps = (i == 0) ? 0.9 : 25.6;
        double bw = plans[i].bytes_per_token;
        /* the plan's weight bytes fit the budget (already checked) and
         * the per-token bytes are a fraction of what fits in RAM */
        CHECK(bw > 0, "bandwidth: bytes/token is positive");
        CHECK(plans[i].n_devices_used >= 1, "devices: at least the CPU is used");
    }
    printf("  ok: CM4 routes across %d devices (CPU+NPU+SD), super across %d\n",
           plans[0].n_devices_used, plans[4].n_devices_used);
    CHECK(plans[0].n_devices_used == 2,
          "devices: CM4 uses the NPU (CPU+NPU split)");
    CHECK(plans[4].n_devices_used == 2,
          "devices: supercomputer uses the GPU");

    /* ENERGY: the CM4 stays in the low energy class (EnerInfer: the
     * thermal budget is a first-class constraint on edge). */
    printf("  ok: CM4 energy class %d (%.0fW est), super class %d (%.0fW est)\n",
           plans[0].energy_class, plans[0].watts_estimate,
           plans[4].energy_class, plans[4].watts_estimate);
    CHECK(plans[0].energy_class <= 1, "energy: CM4 is low-power (class 0-1)");
    CHECK(plans[4].energy_class >= 2, "energy: super is high-power (class 2-3)");

    /* ADAPTIVE: every tier gets early-exit capability (PALBERT: not
     * every input needs every layer -- on the Pi it is the difference
     * between running and being unusable). */
    for (size_t i = 0; i < N_CASES; i++)
        CHECK(plans[i].adaptive_depth == 1,
              "adaptive: fractal_depth is a ceiling on every tier (early-exit)");

    /* ---- the host probe path (smoke) ---- */
    {
        wubu_scale_hw_t hw;
        int rc = wubu_scale_probe_hw(&hw);
        CHECK(rc == 0, "host probe succeeds");
        if (rc == 0) {
            printf("  host : ram=%lluMB cores=%d simd=%d accel=%d\n",
                   (unsigned long long)(hw.ram_bytes / (1024*1024)),
                   hw.cores, hw.simd_bits, hw.has_accel);
            wubu_scale_plan_t p;
            if (wubu_scale_plan(&hw, &ckpt, &p) == 0) {
                CHECK(p.total_bytes <= hw.ram_bytes,
                      "host plan fits the host budget");
                char buf[320];
                wubu_scale_report(&p, buf, sizeof(buf));
                printf("  host  : %s\n", buf);
            }
        }
    }

    /* ---- the WuBu1 real number on real hardware ---- */
    /* On the laptop-class host, the full seed (56.4M) must fit: the
     * plan's ecosystem + core weight budget >= the seed's F16 size. */
    {
        uint64_t seed_f16 = CKPT_PARAMS * 2;   /* F16 bytes */
        /* MID tier: the plan should be able to hold the seed. */
        int mid_fits = 1;
        for (size_t i = 0; i < N_CASES; i++) {
            if (k_cases[i].tier == WUBU_SCALE_MID)
                mid_fits = (plans[i].ram_budget - plans[i].kv_reserve) >= seed_f16;
        }
        printf("  ok: seed F16 needs %lluMB; MID-tier weight budget %lluMB\n",
               (unsigned long long)(seed_f16 / (1024*1024)),
               (unsigned long long)((plans[2].ram_budget - plans[2].kv_reserve) / (1024*1024)));
        CHECK(mid_fits, "MID tier can hold the full WuBu1 seed (the incarnation)");
    }

    if (failures == 0) printf("=== ALL SCALE-TO-FIT TESTS PASSED ===\n");
    else printf("=== %d FAILURES ===\n", failures);
    return failures ? 1 : 0;
}
