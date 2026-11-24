/* wubu_loop_place.c -- EG-105 loop-aware expert placement gate. Strict C11. */
#include "wubu_loop_place.h"

static int valid(const wubu_loop_cfg_t *cfg)
{
    return cfg && cfg->layers >= 2 && cfg->layers <= WUBU_LOOP_MAX_LAYERS &&
           cfg->loop_start >= 0 && cfg->loop_end >= cfg->loop_start &&
           cfg->loop_end < cfg->layers && cfg->passes >= 1 &&
           cfg->passes <= 10 && cfg->bytes_per_layer > 0;
}

int wubu_loop_place_plan(const wubu_loop_cfg_t *cfg, int *assign_out)
{
    int looped;
    if (!valid(cfg) || !assign_out) return -1;
    looped = cfg->loop_end - cfg->loop_start + 1;
    if ((uint64_t)looped * cfg->bytes_per_layer > cfg->gpu_budget) return -1;
    for (int i = 0; i < cfg->layers; ++i)
        assign_out[i] = (i >= cfg->loop_start && i <= cfg->loop_end) ? 0 : 1;
    return 0;
}

int wubu_loop_place_loop_layers(const wubu_loop_cfg_t *cfg)
{
    if (!valid(cfg)) return -1;
    return cfg->loop_end - cfg->loop_start + 1;
}

uint64_t wubu_loop_place_cpu_reads(const wubu_loop_cfg_t *cfg,
                                   const int *assign)
{
    uint64_t reads = 0;
    if (!valid(cfg) || !assign) return 0;
    for (int i = 0; i < cfg->layers; ++i) {
        if (!assign[i]) continue;
        int passes = (i >= cfg->loop_start && i <= cfg->loop_end) ? cfg->passes : 1;
        reads += cfg->bytes_per_layer * (uint64_t)passes;
    }
    return reads;
}
