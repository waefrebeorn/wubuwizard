/* wubu_residency.c -- EG-110 DRAM/Flash residency gate. Strict C11. */
#include "wubu_residency.h"

static int valid(const wubu_residency_layer_t *layers, int n, const int *plan)
{
    if (!layers || !plan || n < 1 || n > WUBU_RESIDENCY_MAX_LAYERS) return 0;
    for (int i = 0; i < n; ++i) {
        if (layers[i].bytes == 0 || (layers[i].hot != 0 && layers[i].hot != 1)) return 0;
        if (plan[i] != 0 && plan[i] != 1) return 0;
    }
    return 1;
}

int wubu_residency_plan(const wubu_residency_layer_t *layers, int n,
                        uint64_t dram_budget, int *plan_out)
{
    uint64_t hot = 0;
    if (!layers || !plan_out || n < 1 || n > WUBU_RESIDENCY_MAX_LAYERS) return -1;
    for (int i = 0; i < n; ++i)
        if (layers[i].bytes == 0 || (layers[i].hot != 0 && layers[i].hot != 1)) return -1;
    for (int i = 0; i < n; ++i) hot += layers[i].hot ? layers[i].bytes : 0;
    /* Streaming a layer needs a DRAM staging buffer at least as large as it. */
    for (int i = 0; i < n; ++i) {
        uint64_t need = hot + (layers[i].hot ? 0 : layers[i].bytes);
        if (need > dram_budget) return -1;
    }
    if (hot > dram_budget) return -1;
    for (int i = 0; i < n; ++i) plan_out[i] = layers[i].hot;
    return 0;
}

uint64_t wubu_residency_dram_bytes(const wubu_residency_layer_t *layers,
                                   const int *plan, int n)
{
    uint64_t total = 0;
    if (!valid(layers, n, plan)) return 0;
    for (int i = 0; i < n; ++i)
        if (plan[i]) total += layers[i].bytes;
    return total;
}

uint64_t wubu_residency_flash_bytes(const wubu_residency_layer_t *layers,
                                    const int *plan, int n)
{
    uint64_t total = 0;
    if (!valid(layers, n, plan)) return 0;
    for (int i = 0; i < n; ++i)
        if (!plan[i]) total += layers[i].bytes;
    return total;
}
