#include <stdio.h>
/*
 * wubu_kernel_budget.c -- byte-budget compiler for the kernel-layer model.
 *
 * Computes the slab layout from WUBU_RUNTIME_DIMS (probed) + the quant cascade
 * (wubu_kv_select). The layout is FIXED order so Styx paths are stable:
 *
 *   embedding        [vocab, dim]      weights
 *   final_norm       [dim]            f32
 *   blocks           ...              (see below, per-block)
 *     q_proj         [dim, heads*hdim]  weights
 *     k_proj         [dim, kv_heads*hdim]
 *     v_proj         [dim, kv_heads*hdim]
 *     o_proj         [dim, dim]
 *     g_proj         [dim, dim]
 *     q_norm         [kv_heads*hdim]
 *     k_norm         [kv_heads*hdim]
 *     attn_norm      [dim]
 *     gate_up        [dim, 2*ffn_dim]
 *     down           [ffn_dim, dim]
 *     ffn_norm       [dim]
 *   selectors        [n_layers, dim]
 *   kv_cache         [10 * max_ctx * kv_heads * hdim * 2]  → tiered
 *
 * The OLD layout (14 separate calloc per block, F32 KV) is the reference
 * the test checks against. The NEW layout (arena, F16/int4 weights) is
 * the kernel default. Both must forward-invariant.
 */
#include "wubu_kernel_budget.h"
#include "wubu.h"            /* WUBU_RUNTIME_DIMS + block shapes */
#include "wubu_kv_select.h"   /* WUBU_KV_* */
#include <stddef.h>

static size_t slab_bytes(int elems, int prec) {
    switch (prec) {
        case WUBU_PREC_F32:  return (size_t)elems * 4;
        case WUBU_PREC_F16:  return (size_t)elems * 2;
        case WUBU_PREC_INT4: return ((size_t)elems + 1) / 2;  /* nybble pack */
        default:             return (size_t)elems * 4;
    }
}

static size_t align_up(size_t n, size_t a) {
    return (n + (a - 1)) & ~(a - 1);
}

int wubu_kernel_budget(wubu_budget_t *out, int weight_prec,
                       int kv_prec, int max_ctx)
{
    if (!out) return 0;
    int p = (weight_prec == 4) ? WUBU_PREC_INT4 : WUBU_PREC_F16;
    int kv_p = WUBU_PREC_F32;
    switch (kv_prec) {
        case WUBU_KV_F16: kv_p = WUBU_PREC_F16; break;
        case WUBU_KV_Q8:  kv_p = WUBU_PREC_Q8_0; break;
        case WUBU_KV_Q4_0: kv_p = WUBU_PREC_Q4_0; break;
        case WUBU_KV_F32: kv_p = WUBU_PREC_F32; break;
    }

    /* Slab names must be unique for Styx path lookup. We embed the
     * block index into the name field via a shared name-pool. */
    static char names[WUBU_BUDGET_MAX][32];
    static char nameoff[WUBU_BUDGET_MAX];   /* offset into the slab name */
    int ni = 0;
    int i = 0;
    int L = WUBU_LAYERS, D = WUBU_DIM, H = WUBU_HEADS,
        KH = WUBU_KV_HEADS, HD = WUBU_HEAD_DIM, F = WUBU_FFN_DIM;
    (void)p; (void)kv_p; (void)max_ctx; (void)L; (void)D; (void)H;
    (void)KH; (void)HD; (void)F;
#define NAMED(prefix, layer, tag) \
    (snprintf(names[ni], sizeof(names[ni]), "%s%d.%s", prefix, layer, tag), names[ni++])

    /* Embedding (tied with lm_head — ONE slab) */
    out[i++] = (wubu_budget_t){"embedding", WUBU_VOCAB * D, p, 64};

    /* Final norm (always F32 — tiny) */
    out[i++] = (wubu_budget_t){"final_norm", D, WUBU_PREC_F32, 4};

    /* Per-block weights — names embed the block index so each slab is
     * path-addressable (/kv/blk_02/q_proj, /kv/blk_02/gate_up, etc.). */
    for (int l = 0; l < L; l++) {
        out[i++] = (wubu_budget_t){NAMED("blk_", l, "q_proj"),  D * (H*HD), p, 64};
        out[i++] = (wubu_budget_t){NAMED("blk_", l, "k_proj"),  D * (KH*HD), p, 64};
        out[i++] = (wubu_budget_t){NAMED("blk_", l, "v_proj"),  D * (KH*HD), p, 64};
        out[i++] = (wubu_budget_t){NAMED("blk_", l, "o_proj"),  D * D,         p, 64};
        out[i++] = (wubu_budget_t){NAMED("blk_", l, "g_proj"),  D * D,         p, 64};
        out[i++] = (wubu_budget_t){NAMED("blk_", l, "q_norm"),  KH*HD, WUBU_PREC_F32, 4};
        out[i++] = (wubu_budget_t){NAMED("blk_", l, "k_norm"),  KH*HD, WUBU_PREC_F32, 4};
        out[i++] = (wubu_budget_t){NAMED("blk_", l, "attn_norm"), D, WUBU_PREC_F32, 4};
        out[i++] = (wubu_budget_t){NAMED("blk_", l, "gate_up"), D * (2*F),     p, 64};
        out[i++] = (wubu_budget_t){NAMED("blk_", l, "down"),    F * D,         p, 64};
        out[i++] = (wubu_budget_t){NAMED("blk_", l, "ffn_norm"), D, WUBU_PREC_F32, 4};
    }

    /* Selectors (residual routing weight per block) */
    out[i++] = (wubu_budget_t){"selectors", L * D, p, 64};

    /* KV cache: 10 GQA layers × max_ctx × kv_heads × head_dim × 2 (K+V) */
    int kv_elems = 10 * max_ctx * KH * HD * 2;
    out[i++] = (wubu_budget_t){"kv_cache", kv_elems, kv_p, 64};
#undef NAMED

    return i;
}

size_t wubu_kernel_total_bytes(const wubu_budget_t *budget, int n)
{
    size_t total = 0;
    for (int i = 0; i < n && i < WUBU_BUDGET_MAX; i++) {
        size_t bytes = slab_bytes(budget[i].elems, budget[i].prec);
        total += align_up(bytes, budget[i].align);
        /* inter-slab alignment padding */
    }
    return align_up(total, 64);   /* arena itself is 64-aligned */
}

wubu_arena_t *wubu_kernel_allocate(const wubu_budget_t *budget, int n)
{
    size_t total = wubu_kernel_total_bytes(budget, n);
    wubu_arena_t *a = wubu_arena_create(total);
    if (!a) return NULL;
    /* Push each slab — if any fails, abort (the budget was computed
     * exactly, so this should never happen). */
    for (int i = 0; i < n && i < WUBU_BUDGET_MAX; i++) {
        size_t bytes = slab_bytes(budget[i].elems, budget[i].prec);
        if (!wubu_arena_push(a, budget[i].name, bytes, budget[i].align,
                             budget[i].prec, budget[i].elems)) {
            wubu_arena_free(a);
            return NULL;
        }
    }
    return a;
}
