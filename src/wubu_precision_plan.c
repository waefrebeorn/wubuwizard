/* wubu_precision_plan.c -- HETEROGENEOUS PRECISION (AM03, research/046).
 *
 * The ladder is a pure function of the hardware profile:
 *   big box  -> Escha defaults (gate/up 2b, down 3b, dense 8b)
 *   small box-> dense drops to INT4 (AWQ-class), experts stay low
 *   no SIMD  -> F32 fallback (correctness over density)
 * The quality-density gate decides acceptance (quality/byte).
 *
 * C11, opaque, self-contained. Reuses wubu_hwcaps (SIMD) + wubu_scale
 * (RAM/GPU probe) -- both degrade to defaults when not linked.
 */
#include "wubu_precision_plan.h"

#include <stdlib.h>
#include <string.h>

/* ---- the three ladder profiles ---- */

/* Escha defaults (the proven 35B-A3B split): gate/up 2b, down 3b,
 * dense INT8, embedding INT8, norms F32. */
static void ladder_escha(wubu_precision_plan_t *p, uint64_t total) {
    p->bits[WUBU_FAM_GATE_UP] = 2;
    p->bits[WUBU_FAM_DOWN] = 3;
    p->bits[WUBU_FAM_DENSE] = 8;
    p->bits[WUBU_FAM_EMBEDDING] = 8;
    p->bits[WUBU_FAM_NORM] = 32;
    p->total_params = total;
    p->profile_name = "escha";
    p->quality_estimate = 0.92;   /* Escha W2: within ~2pp of Q8 on code */
}

/* Edge: small RAM -> dense drops to INT4 (AWQ-class), gate/up stays 2b. */
static void ladder_edge(wubu_precision_plan_t *p, uint64_t total) {
    ladder_escha(p, total);
    p->bits[WUBU_FAM_DENSE] = 4;
    p->bits[WUBU_FAM_EMBEDDING] = 4;
    p->profile_name = "edge";
    p->quality_estimate = 0.85;
}

/* Fallback: no SIMD -> F32 everywhere (correctness over density). */
static void ladder_fallback(wubu_precision_plan_t *p, uint64_t total) {
    for (int i = 0; i < WUBU_FAM_COUNT; i++)
        p->bits[i] = 32;
    p->total_params = total;
    p->profile_name = "fallback";
    p->quality_estimate = 1.0;
}

/* ---- hardware probe ---- */

int wubu_hw_profile_probe(wubu_hw_profile_t *prof) {
    if (!prof) return -1;
    memset(prof, 0, sizeof(*prof));

    /* SIMD: wubu_hwcaps (degrades to 128 if not linked). */
    {
        extern const void *wubu_hwcaps_get(void);
        const void *h = wubu_hwcaps_get();
        if (h) {
            /* read simd_bits via a tiny offset-free accessor: the
             * struct's first ints are flags then simd_bits; instead of
             * depending on the layout, default to 128 and let the
             * caller (or the scale probe) refine. */
            prof->simd_bits = 128;
        } else {
            prof->simd_bits = 128;
        }
    }

    /* RAM + GPU via the scale probe (degrades to /proc/meminfo). */
    {
        extern int wubu_scale_probe_hw(void *hw);
        /* the scale hw struct starts with ram_bytes (u64) then cores
         * (int) then simd_bits (int); we only need ram + accel. */
        struct { uint64_t ram; int cores; int simd; int accel; } shw;
        memset(&shw, 0, sizeof(shw));
        if (wubu_scale_probe_hw(&shw) == 0) {
            prof->ram_bytes = shw.ram;
            prof->cores = shw.cores;
            prof->has_gpu = shw.accel;
            if (shw.simd >= 256) prof->simd_bits = shw.simd;
        }
    }
    return 0;
}

/* ---- the ladder: pure function of the profile ---- */

int wubu_precision_plan_for_profile(const wubu_hw_profile_t *prof,
                                    wubu_precision_plan_t *plan) {
    if (!prof || !plan) return -1;
    memset(plan, 0, sizeof(*plan));

    /* The checkpoint geometry: WuBu1 (56.4M) family split. The family
     * proportions are probed from the real checkpoint by the loader;
     * here we use the WuBu1 fractions as the default (the plan is a
     * starting point — the runtime tightens it). */
    uint64_t total = 56376832ull;
    uint64_t gate_up = total * 55 / 100;      /* MoE/expert bulk */
    uint64_t down    = total * 18 / 100;
    uint64_t dense   = total * 15 / 100;
    uint64_t emb     = total * 10 / 100;
    uint64_t norm    = total - gate_up - down - dense - emb;

    /* profile -> ladder */
    if (prof->simd_bits < 128) {
        ladder_fallback(plan, total);
    } else if (prof->ram_bytes > 0 &&
               prof->ram_bytes <= 2ull * 1024 * 1024 * 1024) {
        ladder_edge(plan, total);             /* small box: <= 2GB */
    } else {
        ladder_escha(plan, total);            /* big box */
    }

    /* bytes at the ladder (per-family params x bits/8, round up). */
    uint64_t fam_params[WUBU_FAM_COUNT] = {
        gate_up, down, dense, emb, norm
    };
    plan->total_bytes = 0;
    for (int i = 0; i < WUBU_FAM_COUNT; i++) {
        uint64_t bytes = fam_params[i] * (uint64_t)plan->bits[i] / 8;
        if (fam_params[i] * (uint64_t)plan->bits[i] % 8) bytes++;
        plan->total_bytes += bytes;
    }
    plan->total_params = total;
    plan->density = (double)plan->quality_estimate /
                    (double)(plan->total_bytes ? plan->total_bytes : 1);
    return 0;
}

uint64_t wubu_precision_plan_bytes(const wubu_precision_plan_t *plan) {
    return plan ? plan->total_bytes : 0;
}

/* ---- the quality-density gate ---- */

int wubu_precision_density_better(const wubu_precision_plan_t *candidate,
                                  const wubu_precision_plan_t *current) {
    if (!candidate || !current) return 0;
    /* a plan is accepted only if quality/byte beats the current plan */
    double c_dens = candidate->total_bytes
        ? (double)candidate->quality_estimate /
          (double)candidate->total_bytes : 0.0;
    double cur_dens = current->total_bytes
        ? (double)current->quality_estimate /
          (double)current->total_bytes : 0.0;
    return c_dens > cur_dens * 1.05;   /* 5% improvement to switch */
}
