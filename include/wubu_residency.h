#ifndef WUBU_RESIDENCY_H
#define WUBU_RESIDENCY_H

/*
 * wubu_residency.h -- EG-110: MNN-style DRAM/Flash hybrid residency planner.
 * Mobile engines keep attention/hot weights in DRAM and stream the remaining
 * layers from Flash in sequential order. This C11 gate plans which layers fit
 * in a DRAM budget and accounts for bytes left on Flash. DRAM must hold every
 * hot layer plus one staging buffer sized for the largest streamed layer, so a
 * plan that cannot stage a streamed layer fails closed instead of overflowing.
 * It models residency, not measured Flash bandwidth or model quality.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_RESIDENCY_MAX_LAYERS 256

typedef struct {
    int hot;            /* 1 = keep resident in DRAM, 0 = stream from Flash */
    uint64_t bytes;
} wubu_residency_layer_t;

/* plan_out[i] = 1 DRAM, 0 Flash. 0 when all hot layers fit, -1 otherwise. */
int wubu_residency_plan(const wubu_residency_layer_t *layers, int n,
                        uint64_t dram_budget, int *plan_out);

uint64_t wubu_residency_dram_bytes(const wubu_residency_layer_t *layers,
                                   const int *plan, int n);
uint64_t wubu_residency_flash_bytes(const wubu_residency_layer_t *layers,
                                    const int *plan, int n);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_RESIDENCY_H */
