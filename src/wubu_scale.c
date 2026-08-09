/* wubu_scale.c -- SCALE-TO-FIT planner (THEORY/11).
 *
 * One checkpoint, any hardware: probe -> tier -> plan. The plan is
 * closed-form: given the budget and the checkpoint geometry, solve for
 * the largest body (core layers + ecosystem balls) that fits, honoring
 * the small-active/huge-total ratio and the precision cascade.
 *
 * C11, opaque. Self-contained (stdlib + optional hwcaps/mem_budget
 * probes; the probes degrade gracefully when those modules are absent
 * at link time -- the planner core is pure math on the hw struct).
 */
#include "wubu_scale.h"
#include "wubu_hwcaps.h"    /* SIMD ladder probe (minimal header) */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if !defined(_WIN32)
#include <unistd.h>
#else
#include <windows.h>
#endif

/* ---- tier classification (computed, not a static policy table) ---- */

static wubu_scale_tier_t tier_of(uint64_t ram) {
    if (ram <= 512ull * 1024 * 1024) return WUBU_SCALE_TINY;
    if (ram <= 4ull   * 1024 * 1024 * 1024) return WUBU_SCALE_SMALL;
    if (ram <= 32ull  * 1024 * 1024 * 1024) return WUBU_SCALE_MID;
    if (ram <= 1024ull * 1024 * 1024 * 1024) return WUBU_SCALE_BIG;
    return WUBU_SCALE_HUGE;
}

static const char *tier_name(wubu_scale_tier_t t) {
    switch (t) {
        case WUBU_SCALE_TINY:  return "tiny";
        case WUBU_SCALE_SMALL: return "small";
        case WUBU_SCALE_MID:   return "mid";
        case WUBU_SCALE_BIG:   return "big";
        case WUBU_SCALE_HUGE:  return "huge";
        default:               return "?";
    }
}

/* bytes per param at a precision */
static uint64_t prec_bytes(wubu_scale_prec_t p) {
    switch (p) {
        case WUBU_SCALE_F32: return 4;
        case WUBU_SCALE_F16: return 2;
        case WUBU_SCALE_Q8:  return 1;
        case WUBU_SCALE_Q4:  return 1;   /* ~0.5 -> round up to 1 */
        case WUBU_SCALE_Q2:  return 1;   /* ~0.25 -> round up to 1 */
        default:             return 4;
    }
}

/* The per-tier defaults (the ratio lever + the cascade). */
typedef struct {
    int    core_layers;    /* boot-core depth */
    int    n_balls;        /* ecosystem colony size */
    int    k_active;       /* fired per token */
    int    fractal_depth;  /* fractal layers beyond the core */
    wubu_scale_prec_t cascade;
    double ratio_target;   /* k_active / n_balls ceiling */
} tier_defaults_t;

static const tier_defaults_t k_tiers[WUBU_SCALE_COUNT] = {
    /* TINY  */ { 4,  16,   2, 0, WUBU_SCALE_Q8, 0.125 },
    /* SMALL */ { 6,  64,   4, 1, WUBU_SCALE_Q8, 0.0625 },
    /* MID   */ { 8, 256,   8, 2, WUBU_SCALE_F16, 0.03125 },
    /* BIG   */ { 10, 1024, 16, 3, WUBU_SCALE_F16, 0.015625 },
    /* HUGE  */ { 12, 4096, 32, 4, WUBU_SCALE_F32, 0.0078125 },
};

/* ---- host probe (degrades gracefully) ---- */

int wubu_scale_probe_hw(wubu_scale_hw_t *hw) {
    if (!hw) return -1;
    memset(hw, 0, sizeof(*hw));

    /* RAM: try the real mem_budget probe if linked, else /proc/meminfo. */
    extern size_t wubu_mem_detect_available_ram(void);
    hw->ram_bytes = wubu_mem_detect_available_ram();
    if (hw->ram_bytes == 0) {
        FILE *f = fopen("/proc/meminfo", "r");
        if (f) {
            char line[256];
            size_t avail = 0;
            while (fgets(line, sizeof(line), f))
                if (strncmp(line, "MemAvailable:", 13) == 0) {
                    sscanf(line + 13, "%zu", &avail);
                    break;
                }
            fclose(f);
            hw->ram_bytes = avail * 1024;
        }
    }

    /* cores */
#if defined(_WIN32)
    {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        hw->cores = (int)si.dwNumberOfProcessors;
    }
#else
    {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        hw->cores = n > 0 ? (int)n : 1;
    }
#endif

    /* SIMD: wubu_hwcaps if linked; else 128 (portable baseline). */
    {
        extern const wubu_hwcaps_t *wubu_hwcaps_get(void);
        const wubu_hwcaps_t *h = wubu_hwcaps_get();
        if (h) {
            hw->simd_bits = h->simd_bits;
            if (hw->simd_bits < 128) hw->simd_bits = 128;
        } else {
            hw->simd_bits = 128;
        }
    }

    /* accelerator: presence probes (no driver calls, no crashes). */
    FILE *f = fopen("/dev/dxg", "r");
    if (f) { hw->has_accel = 1; fclose(f); }
    if (!hw->has_accel) {
        f = fopen("/dev/nvidia0", "r");
        if (f) { hw->has_accel = 1; fclose(f); }
    }
    return 0;
}

/* ---- the planner (closed-form, O(1)) ---- */

int wubu_scale_plan(const wubu_scale_hw_t *hw,
                    const wubu_scale_ckpt_t *ckpt,
                    wubu_scale_plan_t *plan) {
    if (!hw || !ckpt || !plan || ckpt->total_params == 0) return -1;

    memset(plan, 0, sizeof(*plan));
    plan->tier = tier_of(hw->ram_bytes);
    plan->ram_budget = hw->ram_bytes;
    plan->tier_name = tier_name(plan->tier);

    const tier_defaults_t *d = &k_tiers[plan->tier];

    /* KV reserve: ~10% of budget, capped so tiny machines keep a core. */
    uint64_t kv = hw->ram_bytes / 10;
    uint64_t min_kv = 16ull * 1024 * 1024;         /* 16MB floor */
    uint64_t max_kv = 64ull * 1024 * 1024 * 1024;  /* 64GB cap */
    if (kv < min_kv) kv = min_kv;
    if (kv > max_kv) kv = max_kv;
    plan->kv_reserve = kv;

    uint64_t weight_budget = hw->ram_bytes > kv ? hw->ram_bytes - kv : 0;

    /* Core layers: clamp to the checkpoint's real layer count. */
    int core = d->core_layers;
    if (ckpt->n_layers > 0 && core > ckpt->n_layers) core = ckpt->n_layers;
    if (core < 1) core = 1;
    plan->core_layers = core;

    /* Ecosystem: the largest N whose weight fits the remaining budget.
     * Each ball ~ 7*dim*dim params. Closed-form solve. */
    int n = d->n_balls;
    uint64_t ball_params = 7ull * (uint64_t)ckpt->dim * (uint64_t)ckpt->dim + 1;
    uint64_t ball_bytes = ball_params * prec_bytes(d->cascade);
    while (n > 1 && ball_bytes * (uint64_t)n > weight_budget / 2) {
        n /= 2;                    /* halve until it fits (O(log) at most) */
    }
    if (n < 1) n = 1;
    plan->ecosystem_n = n;

    /* k_active: keep the ratio, floor at 1, cap at N. */
    int k = (int)((double)n * d->ratio_target);
    if (k < 1) k = 1;
    if (k > n) k = n;
    plan->k_active = k;

    /* Fractal depth: 0 on tiny (core only); deeper as the budget grows,
     * clamped by the checkpoint. */
    int fd = d->fractal_depth;
    if (ckpt->n_layers > 0 && fd > ckpt->n_layers - core) fd = ckpt->n_layers - core;
    if (fd < 0) fd = 0;
    plan->fractal_depth = fd;
    plan->cascade = d->cascade;

    /* Weight bytes: core (estimate core_params at cascade) + ecosystem. */
    uint64_t core_bytes = 0;
    if (ckpt->core_params > 0)
        core_bytes = ckpt->core_params * prec_bytes(WUBU_SCALE_F32);
    uint64_t eco_bytes = ball_bytes * (uint64_t)n;
    plan->weight_bytes = core_bytes + eco_bytes;
    plan->total_bytes = plan->weight_bytes + kv;
    plan->ratio_active = (double)k / (double)n;
    return 0;
}

void wubu_scale_report(const wubu_scale_plan_t *plan, char *buf, size_t buflen) {
    if (!plan || !buf || buflen < 1) return;
    snprintf(buf, buflen,
             "tier=%-5s ram=%lluMB core=%d balls=%d k=%d frac=%d "
             "prec=%d w=%lluMB tot=%lluMB ratio=%.4f",
             plan->tier_name,
             (unsigned long long)(plan->ram_budget / (1024*1024)),
             plan->core_layers, plan->ecosystem_n, plan->k_active,
             plan->fractal_depth, (int)plan->cascade,
             (unsigned long long)(plan->weight_bytes / (1024*1024)),
             (unsigned long long)(plan->total_bytes / (1024*1024)),
             plan->ratio_active);
}
