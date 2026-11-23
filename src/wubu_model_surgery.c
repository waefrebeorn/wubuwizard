/* wubu_model_surgery.c -- EG-39b surgery planner gate. Strict C11. */
#include "wubu_model_surgery.h"

#include <stdlib.h>

static int valid(const wubu_surgery_layer_t *layers, int n,
                 const int *keep)
{
    if (!layers || !keep || n < 1 || n > WUBU_SURGERY_MAX_LAYERS) return 0;
    for (int i = 0; i < n; ++i)
        if (layers[i].bytes == 0 || !(layers[i].importance > 0.0)) return 0;
    return 1;
}

int wubu_surgery_plan(const wubu_surgery_layer_t *layers, int n,
                      uint64_t budget, double max_loss_frac, int *keep_out)
{
    int rank[WUBU_SURGERY_MAX_LAYERS];
    uint64_t total = 0, remaining;
    double importance_sum = 0.0, lost = 0.0;
    if (!valid(layers, n, keep_out) || budget == 0 ||
        max_loss_frac < 0.0 || max_loss_frac > 1.0) return -1;
    for (int i = 0; i < n; ++i) {
        rank[i] = i;
        total += layers[i].bytes;
        importance_sum += layers[i].importance;
    }
    if (!(importance_sum > 0.0)) return -1;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double a = layers[rank[i]].importance / (double)layers[rank[i]].bytes;
            double b = layers[rank[j]].importance / (double)layers[rank[j]].bytes;
            if (b < a) { int t = rank[i]; rank[i] = rank[j]; rank[j] = t; }
        }
    for (int i = 0; i < n; ++i) keep_out[i] = 1;
    remaining = total;
    for (int k = 0; k < n && remaining > budget; ++k) {
        int i = rank[k];
        keep_out[i] = 0;
        remaining -= layers[i].bytes;
        lost += layers[i].importance;
    }
    if (remaining > budget || lost / importance_sum > max_loss_frac) return -1;
    return 0;
}

uint64_t wubu_surgery_bytes_saved(const wubu_surgery_layer_t *layers,
                                  const int *keep, int n)
{
    uint64_t saved = 0;
    if (!valid(layers, n, keep)) return 0;
    for (int i = 0; i < n; ++i)
        if (!keep[i]) saved += layers[i].bytes;
    return saved;
}

uint64_t wubu_surgery_remaining_bytes(const wubu_surgery_layer_t *layers,
                                      const int *keep, int n)
{
    uint64_t total = 0;
    if (!valid(layers, n, keep)) return 0;
    for (int i = 0; i < n; ++i) {
        total += layers[i].bytes;
        if (!keep[i]) total -= layers[i].bytes;
    }
    return total;
}

double wubu_surgery_loss_fraction(const wubu_surgery_layer_t *layers,
                                  const int *keep, int n)
{
    double total = 0.0, lost = 0.0;
    if (!valid(layers, n, keep)) return -1.0;
    for (int i = 0; i < n; ++i) {
        total += layers[i].importance;
        if (!keep[i]) lost += layers[i].importance;
    }
    return total > 0.0 ? lost / total : -1.0;
}
