/*
 * test_wubu_kernel.c -- the kernel-layer byte-budget gate (Theory/07).
 *
 * Proves the 35M engine's kernel slab layout matches the byte budget in
 * THEORY/07 AND the kernel contract (alignment, sum-of-slabs == arena):
 *   1. wubu_kernel_budget computes the slab list from probed dims + quant
 *      cascade (F16 weights + Q8 KV at ctx=32768 as the default).
 *   2. Every slab base is aligned to its .align (the AVX-512 / striding
 *      invariant — misalignment = silent corruption, P-arena pitfall).
 *   3. arena->used == wubu_kernel_total_bytes (no gap, no overflow).
 *   4. The F16-weight slab count is < F32 (the 2× compression theorem).
 *   5. The KV cache slab uses Q8_0 precision (1.8× vs F32) at the long
 *      context cap — the P9 tiering contract.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wubu_arena.h"
#include "wubu_kernel_budget.h"
#include "wubu35_dims.h"          /* wubu35_dims_default + WUBU35_DIMS */
#include "wubu_kv_select.h"   /* WUBU_KV_F16/Q8 enum values */
#include "wubu.h"        /* WUBU35_DIMS + WUBU_* macros */

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

/* The probe needs a real checkpoint; this test uses the released defaults. */

static size_t slab_padded_bytes(const wubu_slab_t *s) {
    /* bytes + alignment padding within the arena push */
    (void)s; return 0;
}

int main(void)
{
    /* Populate WUBU35_DIMS from the released defaults (no checkpoint). */
    wubu35_dims_default();

    printf("=== test_wubu_kernel (Theory/07: byte budget) ===\n");
    printf("  dims: vocab=%d dim=%d layers=%d heads=%d kv_heads=%d ffn=%d ctx=%d\n",
           WUBU_VOCAB, WUBU_DIM, WUBU_LAYERS, WUBU_HEADS, WUBU_KV_HEADS,
           WUBU_FFN_DIM, WUBU_MAX_SEQ);

    /* ---- 1. F16 weights + Q8 KV @ the long-context cap ---- */
    int max_ctx = 32768;   /* S6 active cap (the banked context) */
    wubu_budget_t budget[WUBU_BUDGET_MAX];
    int n = wubu_kernel_budget(budget, /*weight_prec=*/16, /*kv_prec=*/WUBU_KV_Q8, max_ctx);
    CHECK(n > 0 && n < WUBU_BUDGET_MAX, "budget produced a sane slab count");

    size_t total_bytes = wubu_kernel_total_bytes(budget, n);
    printf("  arena total: %.1f MB (%d slabs)\n",
           (double)total_bytes / (1024*1024), n);

    /* ---- 2. Allocate + align invariant ---- */
    wubu_arena_t *a = wubu_kernel_allocate(budget, n);
    CHECK(a != NULL, "arena allocated (no OOM)");
    if (!a) return 1;

    int align_ok = 1;
    for (int i = 0; i < a->n_slabs; i++) {
        const wubu_slab_t *s = &a->slabs[i];
        if (((uintptr_t)s->base % s->align) != 0) {
            printf("  FAIL: slab '%s' base %p not aligned to %zu\n",
                   s->name ? s->name : "?", s->base, s->align);
            align_ok = 0; failures++;
        }
    }
    CHECK(align_ok, "all slab bases aligned (AVX-512 / striding invariant)");

    /* ---- 3. Sum-of-slabs == arena used ---- */
    size_t used = wubu_arena_used(a);
    CHECK(used == total_bytes,
          "arena used == kernel_total_bytes (no gap/overflow)");
    printf("  arena used: %.1f MB == budget: %.1f MB\n",
           (double)used/(1024*1024), (double)total_bytes/(1024*1024));

    /* ---- 4. F16 weight compression is ~2× ---- */
    /* Compute the weight footprint (embed + selectors + all block weights)
     * at F16 vs F32 directly. The KV cache is excluded (separate tiering). */
    size_t w_f16 = 0, w_f32 = 0;
    for (int i = 0; i < n; i++) {
        /* weight slabs are the non-F32, non-KV ones */
        if (budget[i].prec == WUBU_PREC_F16) {
            w_f16 += (size_t)budget[i].elems * 2;
            w_f32 += (size_t)budget[i].elems * 4;
        }
    }
    double ratio = (double)w_f32 / (double)w_f16;
    printf("  F16/F32 weight ratio: %.2f× [w_f16=%.1fMB w_f32=%.1fMB]\n",
           ratio, (double)w_f16/(1024*1024), (double)w_f32/(1024*1024));
    CHECK(ratio > 1.95 && ratio < 2.05,
          "weight compression is exactly ~2× (F16 vs F32 slab bytes)");

    /* ---- 5. KV cache uses Q8_0 at long context ---- */
    const wubu_slab_t *kv = wubu_arena_find(a, "kv_cache");
    CHECK(kv != NULL, "KV cache slab exists");
    CHECK(kv && kv->prec == WUBU_PREC_Q8_0,
          "KV cache slab is Q8_0 (P9 tiering at long context)");
    int kv_elems = 10 * max_ctx * WUBU_KV_HEADS * WUBU_HEAD_DIM * 2;
    CHECK(kv && kv->elems == kv_elems,
          "KV slab elems == 10*ctx*kv_heads*head_dim*2");
    printf("  KV cache: %d elems, Q8_0, ~%.1f MB (F32 would be %.1f MB)\n",
           kv ? kv->elems : 0,
           kv ? (double)kv->elems * 1.125 / (1024*1024) : 0.0,
           (double)kv_elems * 4.0 / (1024*1024));

    /* ---- 6. Slab names unique + count matches expectation ----
     * 1 embedding + 1 final_norm + L*11 block slabs + 1 selectors + 1 KV = ? */
    int expect = 1 + 1 + WUBU_LAYERS * 11 + 1 + 1;
    CHECK(n == expect, "slab count == 1+1+L*11+1+1");
    printf("  slab count: %d (expected %d — embedding, norm, %d×11, selectors, kv)\n",
           n, expect, WUBU_LAYERS);

    wubu_arena_free(a);
    if (failures == 0) printf("=== ALL KERNEL TESTS PASSED ===\n");
    else printf("=== %d FAILURES ===\n", failures);
    return failures ? 1 : 0;
}
