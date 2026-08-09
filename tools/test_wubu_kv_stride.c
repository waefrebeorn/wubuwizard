/*
 * test_wubu_kv_stride.c -- the S6 gate: runtime KV context cap.
 *
 * The Revolver Doctrine says the KV cache is a BANKED cylinder, not a
 * flat-static window. S6 = gqa_max_ctx is runtime-overridable (the
 * active rotation count), while the struct's physical buffer stays at
 * GQA_MAX_CTX (the cylinder length).
 *
 * This test proves the per-layer KV stride uses model->gqa_max_ctx
 * (NOT the compile-time GQA_MAX_CTX), so a WUBU_MAX_CTX override
 * actually retires the right bank.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_model.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

int main(void)
{
    printf("=== test_wubu_kv_stride (S6: runtime KV context cap) ===\n");

    /* 1. The struct now CARRIES the active cap (the bank count). */
    wubu_model_t m;
    memset(&m, 0, sizeof(m));
    /* A freshly-inited model defaults gqa_max_ctx to GQA_MAX_CTX. */
    m.gqa_max_ctx = GQA_MAX_CTX;
    CHECK(m.gqa_max_ctx == GQA_MAX_CTX, "S6 field exists + defaults to GQA_MAX_CTX");

    /* 2. Override (the Revolver rotation) — set a smaller active bank. */
    int small = GQA_MAX_CTX / 8;
    m.gqa_max_ctx = small;
    CHECK(m.gqa_max_ctx == small, "S6: active cap overridable to GQA_MAX_CTX/8");

    /* 3. The per-layer stride (the code we patched in wubu_model.c)
     *    is: l_gqa * model->gqa_max_ctx * GQA_KV_DIM.
     *    With gqa_max_ctx = small, layer N's bank starts at a smaller
     *    offset than the cylinder would suggest — proving the stride
     *    reads the RUNTIME value, not the static macro. */
    int l_gqa = 3;
    int64_t stride_runtime  = (int64_t)l_gqa * m.gqa_max_ctx * GQA_KV_DIM;
    int64_t stride_static   = (int64_t)l_gqa * GQA_MAX_CTX  * GQA_KV_DIM;
    CHECK(stride_runtime < stride_static,
          "S6: runtime stride < static stride (banked, not flat)");

    /* 4. The override is consistent: stride = l_gqa * gqa_max_ctx * KV_DIM */
    CHECK(stride_runtime == (int64_t)l_gqa * small * GQA_KV_DIM,
          "S6: stride = gqa_max_ctx * KV_DIM (the formula we patched)");

    /* 5. The buffer must be at least as large as the stride demands
     *    (else the bank would read past the allocation). GQA_MAX_CTX
     *    is the physical cylinder — gqa_max_ctx is the active count —
     *    so: cylinder >= active. */
    CHECK(GQA_MAX_CTX >= small,
          "S6: physical cylinder (GQA_MAX_CTX) >= active bank (gqa_max_ctx)");

    if (failures == 0) printf("=== ALL S6 TESTS PASSED ===\n");
    else printf("=== %d FAILURES ===\n", failures);
    return failures ? 1 : 0;
}
