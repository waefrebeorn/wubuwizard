/* wubu_scale.h -- SCALE-TO-FIT (THEORY/11): one checkpoint, any hardware.
 *
 * WuBu is the alive homogeneous amoeba: no fixed number. The loader
 * probes the machine (RAM, SIMD, cores), classifies it into a tier, and
 * computes a load plan -- core layers, ecosystem N, precision cascade,
 * active ratio -- such that the SAME checkpoint boots on a 256MB CM4
 * AND a 1TB supercomputer.
 *
 *   wubu_scale_plan()  is the ONE function every machine calls.
 *   The plan is O(1) closed-form: solve for the largest body that fits
 *   the budget, honoring the small-active/huge-total ratio.
 *
 * C11, opaque. Reuses wubu_mem_budget (RAM probe) + wubu_hwcaps (SIMD
 * ladder) when the caller lets it probe; the test injects budgets to
 * simulate tiers. No engine headers -- the planner is self-contained.
 */
#ifndef WUBU_SCALE_H
#define WUBU_SCALE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The hardware tiers (computed from the probe, never a static table).
 * TINY = 256MB CM4-class ... HUGE = 1TB+ supercomputer. */
typedef enum {
    WUBU_SCALE_TINY = 0,   /* <= 512 MB  : boot core only, N=16, K=1-2 */
    WUBU_SCALE_SMALL,      /* <= 4 GB    : + trunk, N=64, K=4 */
    WUBU_SCALE_MID,        /* <= 32 GB   : full seed, N=256, K=8 */
    WUBU_SCALE_BIG,        /* <= 1 TB    : + fractal layers, N=1024 */
    WUBU_SCALE_HUGE,       /* > 1 TB     : everything + training-grow */
    WUBU_SCALE_COUNT
} wubu_scale_tier_t;

/* The precision cascade: per-depth weight format. F32 at the core,
 * Q2_K at the leaves (AN23 adaptive precision). */
typedef enum {
    WUBU_SCALE_F32 = 0,
    WUBU_SCALE_F16,
    WUBU_SCALE_Q8,
    WUBU_SCALE_Q4,
    WUBU_SCALE_Q2
} wubu_scale_prec_t;

/* The probed machine (injected by tests; probed for real on the host). */
typedef struct {
    uint64_t ram_bytes;        /* available RAM */
    int      cores;            /* CPU cores */
    int      simd_bits;        /* 128/256/512 (from wubu_hwcaps) */
    int      has_accel;        /* CUDA/Vulkan present */
} wubu_scale_hw_t;

/* The checkpoint geometry (probed from tensor shapes -- Revolver). */
typedef struct {
    uint64_t total_params;     /* all weights in the checkpoint */
    uint64_t core_params;      /* boot-core estimate (top-k layers) */
    int      n_layers;         /* total layers in the checkpoint */
    int      dim;              /* model dim (WUBU_DIMS.d_model) */
} wubu_scale_ckpt_t;

/* THE LOAD PLAN: what scale-to-fit computes. */
typedef struct {
    wubu_scale_tier_t  tier;
    uint64_t ram_budget;       /* bytes available for weights + KV */
    uint64_t kv_reserve;       /* bytes kept for the KV namespace */
    int      core_layers;      /* boot-core depth (wubu_boot top-k) */
    int      ecosystem_n;      /* colony size: how many balls fit */
    int      k_active;         /* balls fired per token */
    int      fractal_depth;    /* how many fractal layers load (0 = core only) */
    wubu_scale_prec_t cascade; /* deepest precision in the plan */
    uint64_t weight_bytes;     /* bytes the plan needs for weights */
    uint64_t total_bytes;      /* weight + kv */
    double   ratio_active;     /* k_active / ecosystem_n */
    const char *tier_name;     /* "tiny" ... "huge" */
} wubu_scale_plan_t;

/* Opaque planner handle. */
typedef struct wubu_scale wubu_scale_t;

/* Probe the host machine (RAM via /proc/meminfo + Windows, cores via
 * sysconf, SIMD via wubu_hwcaps, accel via filesystem probes). Returns
 * 0 on success. */
int wubu_scale_probe_hw(wubu_scale_hw_t *hw);

/* Compute the load plan for a checkpoint on a hardware config.
 * hw may be probed (wubu_scale_probe_hw) or injected (tests). The one
 * function every machine calls. Returns 0 on success, -1 on bad input. */
int wubu_scale_plan(const wubu_scale_hw_t *hw,
                    const wubu_scale_ckpt_t *ckpt,
                    wubu_scale_plan_t *plan);

/* Human-readable one-line summary of a plan. buf >= 256 bytes. */
void wubu_scale_report(const wubu_scale_plan_t *plan, char *buf, size_t buflen);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SCALE_H */
