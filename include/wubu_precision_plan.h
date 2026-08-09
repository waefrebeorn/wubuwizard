/* wubu_precision_plan.h -- HETEROGENEOUS PRECISION (AM03, research/046).
 *
 * "Run on all hardware" is a PRECISION PLAN, not a hardware feature.
 * One canonical artifact, a per-family bit ladder chosen by sensitivity:
 *
 *   Escha defaults (research/046, proven at 35B-A3B scale):
 *     gate/up   -> 2 bits (outputs filtered downstream — errors don't
 *                  compound into the residual stream)
 *     down      -> 3 bits (writes into the residual stream)
 *     dense     -> INT8  (q/k/v/o, embedding, non-MoE weights)
 *     norms     -> F32   (tiny but critical)
 *
 *   wubu_hw_profile: detect SIMD/RAM/GPU/storage ONCE; the ladder is a
 *   PURE FUNCTION of the profile (Revolver: probe, don't assume).
 *     big box  -> full Escha
 *     small box-> dense INT4 (AWQ-class)
 *     no SIMD  -> F32 fallback
 *
 *   quality-density gate: a plan is accepted only if quality/byte beats
 *   the current plan (the Escha result reproduced at our scale).
 *
 * C11, opaque, self-contained (reuses wubu_hwcaps for SIMD + the scale
 * probe for RAM/GPU). No engine headers.
 */
#ifndef WUBU_PRECISION_PLAN_H
#define WUBU_PRECISION_PLAN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The matrix families (per-role sensitivity — research/057 ladder). */
typedef enum {
    WUBU_FAM_GATE_UP = 0,   /* MoE/expert gate+up (filtered downstream) */
    WUBU_FAM_DOWN,          /* writes back into the residual stream */
    WUBU_FAM_DENSE,         /* q/k/v/o projections + non-MoE weights */
    WUBU_FAM_EMBEDDING,     /* token embeddings (tied head) */
    WUBU_FAM_NORM,          /* norms + selectors (tiny, critical) */
    WUBU_FAM_COUNT
} wubu_family_t;

/* Bits per value for a family (2/3/4/8/16/32). */
typedef struct {
    int bits[WUBU_FAM_COUNT];   /* per-family bit-width */
    uint64_t total_params;      /* params the plan covers */
    uint64_t total_bytes;       /* bytes at this ladder */
    double   quality_estimate;  /* 0..1 predicted quality (sensitivity) */
    double   density;           /* quality / byte (the gate metric) */
    const char *profile_name;   /* "escha" | "edge" | "fallback" */
} wubu_precision_plan_t;

/* The hardware profile (computed once from the probe). */
typedef struct {
    int simd_bits;          /* 128/256/512 from wubu_hwcaps */
    uint64_t ram_bytes;     /* available RAM */
    int has_gpu;            /* /dev/dxg or CUDA present */
    int cores;
} wubu_hw_profile_t;

/* P1: probe the hardware once (SIMD via wubu_hwcaps, RAM/GPU via the
 * scale probe). Returns 0 on success. */
int wubu_hw_profile_probe(wubu_hw_profile_t *prof);

/* P2: the ladder as a pure function of the profile. The profile picks
 * the plan: big box -> Escha defaults; small box -> dense INT4; no
 * SIMD -> F32 fallback. Always returns a valid plan (never NULL out).
 * Returns 0 on success. */
int wubu_precision_plan_for_profile(const wubu_hw_profile_t *prof,
                                    wubu_precision_plan_t *plan);

/* P3: the quality-density gate. Returns 1 if candidate's density beats
 * current's (quality/byte improved), 0 otherwise. */
int wubu_precision_density_better(const wubu_precision_plan_t *candidate,
                                  const wubu_precision_plan_t *current);

/* P4: bytes for the plan (total_bytes recomputed from the ladder). */
uint64_t wubu_precision_plan_bytes(const wubu_precision_plan_t *plan);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_PRECISION_PLAN_H */
