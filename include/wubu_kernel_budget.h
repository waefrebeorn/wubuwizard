/*
 * wubu_kernel_budget.h -- byte-budget compiler for the kernel-layer model.
 *
 * Theory/07: the kernel does NOT keep the 35M in F32 (202 MB resident).
 * It negotiates a quant cascade from the roofline (wubu_kv_select) and
 * carves ONE arena. This module computes the slab layout + total bytes
 * from the probed dims (WUBU_RUNTIME_DIMS) and the KV/weight precision choice.
 *
 * The budget is a list of (name, elems, prec, align) descriptors that
 * wubu_arena_push turns into aligned slabs. A Styx export walks the slab
 * table for path-addressable access.
 *
 * No third-party deps; only WUBU_RUNTIME_DIMS + wubu_kv_select + wubu_arena.
 */
#ifndef WUBU_KERNEL_BUDGET_H
#define WUBU_KERNEL_BUDGET_H

#include "wubu_arena.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One budget descriptor: wubu_arena_push pushes it with these params. */
typedef struct {
    const char *name;     /* slab name */
    int         elems;    /* element count */
    int         prec;     /* wubu_prec_t */
    size_t      align;    /* alignment (bytes) */
} wubu_budget_t;

/* Compute the kernel slab layout for the 35M model.
 *   weight_prec: 16 (F16) or 4 (INT4) — from wubu_kv_select().weight_bits
 *   kv_prec:     the KV scheme (WUBU_KV_F32/F16/Q8/Q4_0)
 *   max_ctx:     active KV context cap (S6: gqa_max_ctx)
 *
 * Returns the number of budget descriptors written to `out` (capped at
 * WUBU_BUDGET_MAX). The descriptors are in FIXED order so a Styx export
 * has stable paths. The caller allocates arena = wubu_kernel_total_bytes.
 */
#define WUBU_BUDGET_MAX 256   /* 12 layers × 11 slabs + overhead */
int wubu_kernel_budget(wubu_budget_t *out, int weight_prec,
                       int kv_prec, int max_ctx);

/* Total bytes needed for a budget (incl. alignment padding). Pass the
 * returned descriptor count from wubu_kernel_budget. */
size_t wubu_kernel_total_bytes(const wubu_budget_t *budget, int n);

/* Carve an arena from a budget: push every slab, return the arena.
 * NULL on OOM. The caller frees via wubu_arena_free. */
wubu_arena_t *wubu_kernel_allocate(const wubu_budget_t *budget, int n);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KERNEL_BUDGET_H */
