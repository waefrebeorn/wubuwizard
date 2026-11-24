#ifndef WUBU_LOOP_PLACE_H
#define WUBU_LOOP_PLACE_H

/*
 * wubu_loop_place.h -- EG-105: loop-aware expert placement.
 * The Inkling looped core (layers 6..17, three passes per token) must stay on
 * the GPU; non-loop experts are the correct CPU spill candidates. A naive
 * `--n-cpu-moe 12` spills layers 0..11, putting half the loop core in system
 * RAM and re-reading it on every pass. This gates the placement rule and the
 * per-token CPU byte count. Byte counts are synthetic, not a VRAM measurement.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_LOOP_MAX_LAYERS 64

typedef struct {
    int layers;                 /* physical layers */
    int loop_start, loop_end;   /* inclusive looped range */
    int passes;                 /* 3 for released Inkling 2 Mini */
    uint64_t gpu_budget;        /* bytes available for hot loop core */
    uint64_t bytes_per_layer;   /* synthetic expert weight bytes */
} wubu_loop_cfg_t;

/* assign_out[i] = 0 GPU, 1 CPU. 0 when the loop core fits, -1 otherwise. */
int wubu_loop_place_plan(const wubu_loop_cfg_t *cfg, int *assign_out);

int wubu_loop_place_loop_layers(const wubu_loop_cfg_t *cfg);

/* CPU bytes read per token given an assignment; 0 on bad input. */
uint64_t wubu_loop_place_cpu_reads(const wubu_loop_cfg_t *cfg,
                                   const int *assign);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_LOOP_PLACE_H */
