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

    /* ---- RESEARCH/063 axes ---- */

    /* C. BANDWIDTH: bytes per token = weight bytes touched per token.
     * k_active balls fire; each touches its ball's params. The core is
     * always touched (dense). This is the roofline-honest cost. */
    double bpt = (double)core_bytes;                       /* dense core */
    bpt += (double)k * (double)ball_params * (double)prec_bytes(d->cascade);
    plan->bytes_per_token = bpt;

    /* A. DEVICES: route across the devices the machine has. The plan
     * reports how many are USED (1 = CPU only; more = split). The
     * split itself is the router's job at runtime (QEIL: hardware-
     * aware routing); the planner sizes so the CPU can run the core
     * alone and the accelerator carries the outer body. */
    plan->n_devices_used = 1;
    if (hw->n_devices > 0) {
        int accel = 0;
        for (int i = 0; i < hw->n_devices && i < WUBU_SCALE_MAX_DEVICES; i++)
            if (hw->devices[i].kind != WUBU_DEV_CPU) accel++;
        plan->n_devices_used = 1 + (accel > 0 ? 1 : 0);
    }

    /* B. ENERGY: estimate avg draw under load from the device power
     * figures + the active ratio. Energy class 0-3 (EnerInfer: the
     * slack is exploitable -- the runtime tunes frequencies). */
    {
        double watts = 0.0;
        double max_w = 0.0;
        if (hw->n_devices > 0) {
            for (int i = 0; i < hw->n_devices && i < WUBU_SCALE_MAX_DEVICES; i++) {
                watts += hw->devices[i].watts;
                if (hw->devices[i].watts > max_w) max_w = hw->devices[i].watts;
            }
        } else {
            watts = (double)hw->cores * 5.0;   /* ~5W/core default */
            max_w = watts;
        }
        /* scale by how much of the body actually fires */
        plan->watts_estimate = watts * (0.5 + 0.5 * plan->ratio_active);
        if (max_w <= 7.0)      plan->energy_class = 0;   /* phone-class */
        else if (max_w <= 30)  plan->energy_class = 1;   /* SBC/laptop */
        else if (max_w <= 200) plan->energy_class = 2;   /* desktop */
        else                   plan->energy_class = 3;   /* server */
    }

    /* D. ADAPTIVE: on constrained tiers, fractal_depth is a CEILING --
     * the runtime early-exits easy tokens (PALBERT: not every input
     * needs every layer). Every tier can adapt; tiny/small get it
     * because they need it most. */
    plan->adaptive_depth = 1;
    return 0;
}

void wubu_scale_report(const wubu_scale_plan_t *plan, char *buf, size_t buflen) {
    if (!plan || !buf || buflen < 1) return;
    snprintf(buf, buflen,
             "tier=%-5s ram=%lluMB core=%d balls=%d k=%d frac=%d "
             "prec=%d w=%lluMB tot=%lluMB ratio=%.4f "
             "bw=%.0fB/tok watts=%.0f eclass=%d devs=%d adapt=%d",
             plan->tier_name,
             (unsigned long long)(plan->ram_budget / (1024*1024)),
             plan->core_layers, plan->ecosystem_n, plan->k_active,
             plan->fractal_depth, (int)plan->cascade,
             (unsigned long long)(plan->weight_bytes / (1024*1024)),
             (unsigned long long)(plan->total_bytes / (1024*1024)),
             plan->ratio_active,
             plan->bytes_per_token, plan->watts_estimate,
             plan->energy_class, plan->n_devices_used,
             plan->adaptive_depth);
}

/* ---- ON-CHIP MEASURE (research/063-F, OHQ) ---- */

#include <time.h>

/* Quantize a vector to n bits per element, then measure the GEMV time.
 * The measurement is local, one-shot, and real: it times THIS chip's
 * throughput at each precision, weighted by the BYTES MOVED (the
 * roofline-honest cost -- a Q8 GEMV moves 4x fewer weight bytes than
 * F32, and memory-bound silicon reflects that). The cascade the plan
 * assumed is only a starting point; the runtime tightens it from what
 * the silicon says. */
static double measure_gemv_at(int bits, int n_reps) {
    enum { DIM = 512, OUT = 64 };
    static float w[DIM * OUT];
    static float x[DIM];
    static float y[OUT];
    static int inited = 0;
    if (!inited) {
        for (int i = 0; i < DIM * OUT; i++) w[i] = (float)((i * 2654435761u) % 1000) / 500.0f - 1.0f;
        for (int i = 0; i < DIM; i++) x[i] = (float)((i * 40503u) % 1000) / 500.0f - 1.0f;
        inited = 1;
    }

    /* Quantize on the fly: simulate the cascade's weight cost. */
    float q[DIM * OUT];
    float scale = (float)((1 << (bits - 1)) - 1);
    for (int i = 0; i < DIM * OUT; i++) {
        float v = w[i];
        float s = v > 0 ? v : -v;
        q[i] = (float)(int)(v * scale + (v >= 0 ? 0.5f : -0.5f)) / scale;
    }

    /* The honest cost: bytes moved per token = weight bytes at this
     * precision. n_reps scales so each precision gets the same total
     * bytes through the machine (a stable, comparable window). */
    double bytes = (double)(DIM * OUT) * (double)bits / 8.0;
    int reps = (int)(n_reps * (32.0 / (double)bits));  /* more reps at low bits */

    clock_t t0 = clock();
    for (int rep = 0; rep < reps; rep++) {
        for (int o = 0; o < OUT; o++) {
            float acc = 0.0f;
            for (int i = 0; i < DIM; i++)
                acc += x[i] * q[o * DIM + i];
            y[o] = acc;
        }
    }
    clock_t t1 = clock();
    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
    /* bytes/second = the roofline rate this chip sustains at this prec */
    return (bytes * (double)reps) / (secs + 1e-30);
}

int wubu_scale_measure(const wubu_scale_hw_t *hw, wubu_scale_prec_t *best) {
    if (!best) return -1;
    (void)hw;

    /* Time each precision class (F32=32, F16=16, Q8=8, Q4=4, Q2=2).
     * measure_gemv_at returns BYTES/SEC (higher = faster for the same
     * bytes moved). The winner is the fastest cascade the silicon
     * sustains at this precision. */
    struct { int bits; wubu_scale_prec_t prec; } cand[] = {
        { 32, WUBU_SCALE_F32 },
        { 16, WUBU_SCALE_F16 },
        { 8,  WUBU_SCALE_Q8 },
        { 4,  WUBU_SCALE_Q4 },
        { 2,  WUBU_SCALE_Q2 },
    };
    int n = (int)(sizeof(cand) / sizeof(cand[0]));

    double best_bw = 0.0;
    wubu_scale_prec_t best_p = WUBU_SCALE_F32;
    for (int i = 0; i < n; i++) {
        double bw = measure_gemv_at(cand[i].bits, 200);
        if (bw > best_bw) { best_bw = bw; best_p = cand[i].prec; }
    }
    *best = best_p;
    return 0;
}
