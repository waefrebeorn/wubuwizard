/*
 * test_precision_plan.c -- AM03 gate: HETEROGENEOUS PRECISION.
 *
 * "Run on all hardware" is a precision plan, not a hardware feature.
 * Proves the Escha ladder (research/046):
 *
 *   1. PURE FUNCTION: the ladder is a pure function of the profile --
 *      the same profile always yields the same plan (no state, no
 *      assumptions).
 *   2. SENSITIVITY: per-family bits follow the Escha doctrine --
 *      gate/up get the FEWEST bits (filtered downstream), norms the
 *      MOST (tiny but critical). One canonical artifact, not one
 *      uniform bit-width.
 *   3. PROFILE DRIVES: a big box (SIMD 512, 16GB, GPU) gets the full
 *      Escha ladder; a small box (SIMD 128, 512MB, no GPU -- the CM4)
 *      gets the edge ladder (dense drops to INT4); no SIMD -> F32
 *      fallback (correctness over density). The SAME checkpoint, three
 *      plans.
 *   4. DENSITY GATE: the plan is accepted only if quality/byte beats
 *      the incumbent (the Escha result reproduced at our scale).
 *
 * Gate: `make test_precision_plan`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_precision_plan.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

static void show(const char *tag, const wubu_precision_plan_t *p) {
    printf("  %-8s %s: gate/up=%db down=%db dense=%db emb=%db norm=%db "
           "bytes=%lluMB q=%.2f dens=%.3e\n",
           tag, p->profile_name,
           p->bits[WUBU_FAM_GATE_UP], p->bits[WUBU_FAM_DOWN],
           p->bits[WUBU_FAM_DENSE], p->bits[WUBU_FAM_EMBEDDING],
           p->bits[WUBU_FAM_NORM],
           (unsigned long long)(p->total_bytes / (1024*1024)),
           p->quality_estimate, p->density);
}

int main(void)
{
    printf("=== test_precision_plan (AM03: heterogeneous precision) ===\n");

    /* ---- 1+3. profile -> ladder (the pure function) ---- */
    wubu_hw_profile_t big, small, tiny, nosimd;
    memset(&big, 0, sizeof(big)); big.simd_bits = 512;
    big.ram_bytes = 16ull * 1024 * 1024 * 1024; big.has_gpu = 1; big.cores = 16;
    memset(&small, 0, sizeof(small)); small.simd_bits = 128;
    small.ram_bytes = 512ull * 1024 * 1024; small.has_gpu = 0; small.cores = 4;
    memset(&tiny, 0, sizeof(tiny)); tiny.simd_bits = 128;
    tiny.ram_bytes = 256ull * 1024 * 1024; tiny.has_gpu = 0; tiny.cores = 4;
    memset(&nosimd, 0, sizeof(nosimd)); nosimd.simd_bits = 0;
    nosimd.ram_bytes = 1024ull * 1024 * 1024; nosimd.cores = 1;

    wubu_precision_plan_t p_big, p_small, p_tiny, p_nosimd, p_big2;
    CHECK(wubu_precision_plan_for_profile(&big, &p_big) == 0, "big box plans");
    CHECK(wubu_precision_plan_for_profile(&small, &p_small) == 0, "small box plans");
    CHECK(wubu_precision_plan_for_profile(&tiny, &p_tiny) == 0, "tiny box plans");
    CHECK(wubu_precision_plan_for_profile(&nosimd, &p_nosimd) == 0, "no-SIMD plans");
    wubu_precision_plan_for_profile(&big, &p_big2);

    show("big", &p_big);
    show("small", &p_small);
    show("tiny", &p_tiny);
    show("nosimd", &p_nosimd);

    /* the pure function: same profile -> identical plan */
    CHECK(memcmp(&p_big, &p_big2, sizeof(p_big)) == 0,
          "ladder is a pure function of the profile (deterministic)");

    /* ---- 2. sensitivity: the Escha ordering ---- */
    CHECK(p_big.bits[WUBU_FAM_GATE_UP] <= p_big.bits[WUBU_FAM_DOWN],
          "sensitivity: gate/up <= down bits (filtered downstream)");
    CHECK(p_big.bits[WUBU_FAM_DOWN] <= p_big.bits[WUBU_FAM_DENSE],
          "sensitivity: down <= dense bits (residual write)");
    CHECK(p_big.bits[WUBU_FAM_NORM] >= p_big.bits[WUBU_FAM_DENSE],
          "sensitivity: norms get the most bits (tiny but critical)");
    CHECK(p_big.bits[WUBU_FAM_DENSE] == 8,
          "big box: dense is INT8 (the Escha proven split)");
    CHECK(p_big.bits[WUBU_FAM_GATE_UP] == 2,
          "big box: gate/up is 2-bit (Escha)");

    /* ---- 3. profile drives the ladder ---- */
    /* small box (CM4-class, <=2GB): dense drops to INT4 (AWQ-class) */
    CHECK(p_small.bits[WUBU_FAM_DENSE] == 4,
          "small box: dense drops to INT4 (AWQ-class)");
    CHECK(p_tiny.bits[WUBU_FAM_DENSE] == 4,
          "tiny box (256MB): also edge ladder (dense INT4)");
    /* no SIMD: F32 everywhere (correctness over density) */
    CHECK(p_nosimd.bits[WUBU_FAM_GATE_UP] == 32 &&
          p_nosimd.bits[WUBU_FAM_DENSE] == 32,
          "no SIMD: F32 fallback everywhere");
    /* the same checkpoint shrinks with the plan */
    CHECK(p_big.total_bytes > p_small.total_bytes,
          "the SAME checkpoint is smaller on the small box");
    CHECK(p_big.total_bytes > p_tiny.total_bytes ||
          p_big.total_bytes == p_tiny.total_bytes,
          "big box >= edge size (same ladder family)");

    /* ---- 4. the quality-density gate ---- */
    /* the edge ladder is denser (smaller bytes, slight quality drop) --
     * on a 512MB box it must WIN on density */
    CHECK(wubu_precision_density_better(&p_small, &p_big) == 1,
          "density gate: edge ladder beats Escha on the small box");
    /* but the fallback (F32, same quality, huge bytes) must LOSE */
    CHECK(wubu_precision_density_better(&p_nosimd, &p_big) == 0,
          "density gate: F32 fallback loses on density (correctness only)");

    /* the real WuBu1 numbers */
    printf("  ok: WuBu1 (56.4M) at Escha = %lluMB, at edge = %lluMB, "
           "fallback = %lluMB\n",
           (unsigned long long)(p_big.total_bytes / (1024*1024)),
           (unsigned long long)(p_small.total_bytes / (1024*1024)),
           (unsigned long long)(p_nosimd.total_bytes / (1024*1024)));

    if (failures == 0) printf("=== ALL PRECISION-PLAN TESTS PASSED ===\n");
    else printf("=== %d FAILURES ===\n", failures);
    return failures ? 1 : 0;
}
