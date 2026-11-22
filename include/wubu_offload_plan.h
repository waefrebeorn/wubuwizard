#ifndef WUBU_OFFLOAD_PLAN_H
#define WUBU_OFFLOAD_PLAN_H

/*
 * wubu_offload_plan.h -- EG-06: placement solver and zigzag block schedule
 * (FlexGen ICML 2023 §4 Algorithm 1 lineage), reimplemented in C11.
 *
 * Given tier bandwidths/capacities and per-layer sizes/hotness, place each
 * layer on a tier so per-token read time is minimized under capacity.
 * The zigzag schedule alternates forward and reverse layer order across
 * decode iterations so adjacent steps reuse the opposite end of the chain.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { WUBU_OFFLOAD_GPU = 0, WUBU_OFFLOAD_RAM = 1,
       WUBU_OFFLOAD_SSD = 2, WUBU_OFFLOAD_TIERS = 3 };

typedef struct {
    uint64_t bandwidth_bps[WUBU_OFFLOAD_TIERS];  /* bytes per second */
    uint64_t capacity[WUBU_OFFLOAD_TIERS];       /* bytes */
} wubu_offload_hw_t;

typedef struct {
    uint64_t bytes;      /* layer weight size */
    uint64_t hotness;    /* observed activations; higher = prefer faster tier */
} wubu_offload_layer_t;

/* Greedy hot-first placement. assign_out[n] receives a tier id.
 * Returns 0 on success, -1 on bad args or if the layers cannot fit. */
int wubu_offload_solve(const wubu_offload_hw_t *hw,
                       const wubu_offload_layer_t *layers, int n,
                       int *assign_out);

/* Per-token read time in seconds for a placement (lower is better).
 * Returns -1.0 on bad args. */
double wubu_offload_cost(const wubu_offload_hw_t *hw,
                         const wubu_offload_layer_t *layers, int n,
                         const int *assign);

/* Fill out with `iters` forward/reverse passes of n layer ids.
 * Needs cap >= 2*n*iters. Returns 0 or -1. */
int wubu_offload_zigzag(int n, int iters, int *out, int cap);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_OFFLOAD_PLAN_H */
