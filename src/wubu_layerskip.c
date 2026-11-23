/*
 * wubu_layerskip.c -- EG-60 layer dropout + exit-cache remainder. Pure C11.
 */
#include "wubu_layerskip.h"

#include <math.h>

static int skipped(int layer, float p) {
    unsigned int h = (unsigned int)(layer * 2654435761u + 12345u);
    h ^= h >> 13;
    h *= 1274126177u;
    h ^= h >> 16;
    return (double)(h & 0xffffu) / 65536.0 < (double)p;
}

static void layer_apply(float *h, int dim, int layer) {
    float prev[64];
    if (dim > 64) return;
    for (int d = 0; d < dim; d++) prev[d] = h[d];
    for (int d = 0; d < dim; d++) {
        double s = 0.0;
        for (int j = 0; j < dim; j++) {
            double w = sin(0.37 * (layer + 1) + 0.11 * (d + 1) + 0.07 * (j + 1));
            s += w * prev[j];
        }
        h[d] = (float)(prev[d] + 0.05 * tanh(s / dim));
    }
}

int wubu_layerskip_forward(const float *x, int dim, int layers,
                           const float *dropout, int exit_layer,
                           float *out, size_t *kv_saved) {
    float h[64], cache[64];
    size_t saved = 0;
    if (!x || !dropout || !out || !kv_saved || dim <= 0 || dim > 64 ||
        layers <= 0 || exit_layer < -1 || exit_layer >= layers) return -1;
    for (int d = 0; d < dim; d++) h[d] = x[d];

    int boundary = exit_layer >= 0 ? exit_layer : layers - 1;
    for (int l = 0; l <= boundary; l++) {
        if (skipped(l, dropout[l])) {
            saved += (size_t)dim * sizeof(float);
            continue;
        }
        layer_apply(h, dim, l);
    }
    if (exit_layer >= 0)
        for (int d = 0; d < dim; d++) cache[d] = h[d];

    /* Remainder pass resumes from the exit-layer cache, not a recompute. */
    if (exit_layer >= 0)
        for (int d = 0; d < dim; d++) h[d] = cache[d];
    for (int l = boundary + 1; l < layers; l++) {
        if (skipped(l, dropout[l])) {
            saved += (size_t)dim * sizeof(float);
            continue;
        }
        layer_apply(h, dim, l);
    }
    for (int d = 0; d < dim; d++) out[d] = h[d];
    *kv_saved = saved;
    return 0;
}
