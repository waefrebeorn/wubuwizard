/*
 * wubu_neuron_cluster.c -- EG-05 runtime neuron clustering. Pure C11.
 * The predictor uses observed activation counts; no model-specific code.
 */
#include "wubu_neuron_cluster.h"

#include <stdlib.h>
#include <string.h>

struct wubu_neuron_clusters {
    int      n_layers;
    int      n_neurons;
    uint32_t *counts;          /* [layer * n_neurons + neuron] */
    uint8_t  *hot;             /* same shape */
    uint8_t  *picked;          /* scratch for deterministic partition */
    int      cluster_size;
    uint64_t observations;
};

wubu_neuron_clusters_t *wubu_neuron_clusters_create(int n_layers, int n_neurons) {
    wubu_neuron_clusters_t *c;
    size_t n;
    if (n_layers <= 0 || n_neurons <= 0) return NULL;
    if ((size_t)n_layers > ((size_t)-1) / (size_t)n_neurons) return NULL;
    n = (size_t)n_layers * (size_t)n_neurons;
    c = (wubu_neuron_clusters_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->counts = (uint32_t *)calloc(n, sizeof(uint32_t));
    c->hot    = (uint8_t *)calloc(n, sizeof(uint8_t));
    c->picked = (uint8_t *)calloc(n, sizeof(uint8_t));
    if (!c->counts || !c->hot || !c->picked) {
        wubu_neuron_clusters_free(c);
        return NULL;
    }
    c->n_layers = n_layers;
    c->n_neurons = n_neurons;
    c->cluster_size = 1;
    return c;
}

void wubu_neuron_clusters_free(wubu_neuron_clusters_t *c) {
    if (!c) return;
    free(c->counts);
    free(c->hot);
    free(c->picked);
    free(c);
}

int wubu_neuron_clusters_observe(wubu_neuron_clusters_t *c,
                                 int layer, int neuron) {
    if (!c || layer < 0 || layer >= c->n_layers ||
        neuron < 0 || neuron >= c->n_neurons) return -1;
    size_t idx = (size_t)layer * (size_t)c->n_neurons + (size_t)neuron;
    if (c->counts[idx] != UINT32_MAX - 1) c->counts[idx]++;
    c->observations++;
    return 0;
}

int wubu_neuron_clusters_partition(wubu_neuron_clusters_t *c, int hot_percent) {
    int hot_count, total = 0;
    if (!c || hot_percent < 0 || hot_percent > 100) return -1;
    memset(c->hot, 0, (size_t)c->n_layers * (size_t)c->n_neurons);
    memset(c->picked, 0, (size_t)c->n_layers * (size_t)c->n_neurons);
    if (hot_percent == 0) return 0;
    hot_count = (c->n_neurons * hot_percent + 99) / 100;
    if (hot_count < 1) hot_count = 1;
    if (hot_count > c->n_neurons) hot_count = c->n_neurons;

    for (int l = 0; l < c->n_layers; l++) {
        size_t base = (size_t)l * (size_t)c->n_neurons;
        for (int k = 0; k < hot_count; k++) {
            int best = -1;
            uint32_t best_count = 0;
            for (int n = 0; n < c->n_neurons; n++) {
                if (c->picked[base + (size_t)n]) continue;
                uint32_t v = c->counts[base + (size_t)n];
                if (best < 0 || v > best_count) {
                    best = n;
                    best_count = v;
                }
            }
            if (best < 0) break;
            c->picked[base + (size_t)best] = 1;
            c->hot[base + (size_t)best] = 1;
            total++;
        }
    }
    return total;
}

int wubu_neuron_clusters_set_target_bytes(wubu_neuron_clusters_t *c,
                                          int bytes_per_neuron,
                                          size_t target_bytes) {
    if (!c || bytes_per_neuron <= 0) return -1;
    c->cluster_size = (int)(target_bytes / (size_t)bytes_per_neuron);
    if (c->cluster_size < 1) c->cluster_size = 1;
    return c->cluster_size;
}

int wubu_neuron_clusters_cluster_size(const wubu_neuron_clusters_t *c) {
    return c ? c->cluster_size : -1;
}

int wubu_neuron_clusters_hot(const wubu_neuron_clusters_t *c,
                              int layer, int neuron) {
    if (!c || layer < 0 || layer >= c->n_layers ||
        neuron < 0 || neuron >= c->n_neurons) return -1;
    return c->hot[(size_t)layer * (size_t)c->n_neurons + (size_t)neuron] ? 1 : 0;
}

uint64_t wubu_neuron_clusters_predicted_reads(const wubu_neuron_clusters_t *c,
                                              int tokens) {
    if (!c || tokens <= 0) return 0;
    uint64_t clusters = 0;
    for (int l = 0; l < c->n_layers; l++) {
        size_t base = (size_t)l * (size_t)c->n_neurons;
        int n_clusters = (c->n_neurons + c->cluster_size - 1) / c->cluster_size;
        for (int cl = 0; cl < n_clusters; cl++) {
            int cold_active = 0;
            int start = cl * c->cluster_size;
            int end = start + c->cluster_size;
            if (end > c->n_neurons) end = c->n_neurons;
            for (int n = start; n < end; n++) {
                size_t idx = base + (size_t)n;
                if (c->counts[idx] > 0 && !c->hot[idx]) { cold_active = 1; break; }
            }
            if (cold_active) clusters++;
        }
    }
    return clusters * (uint64_t)tokens;
}

uint64_t wubu_neuron_clusters_observations(const wubu_neuron_clusters_t *c) {
    return c ? c->observations : 0;
}
