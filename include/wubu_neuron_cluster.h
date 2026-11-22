#ifndef WUBU_NEURON_CLUSTER_H
#define WUBU_NEURON_CLUSTER_H

/*
 * wubu_neuron_cluster.h -- EG-05: activation-locality predictor and
 * neuron-cluster abstraction (PowerInfer lineage), reimplemented in C11.
 *
 * Observe activations, partition a small hot set (resident in fast memory)
 * from the cold remainder (fetched in clusters), and account predicted
 * cluster reads. Deterministic: ties are broken by neuron index.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_neuron_clusters wubu_neuron_clusters_t;

wubu_neuron_clusters_t *wubu_neuron_clusters_create(int n_layers, int n_neurons);
void wubu_neuron_clusters_free(wubu_neuron_clusters_t *c);

/* Record one activation of (layer, neuron). Returns 0 or -1 on bad args. */
int wubu_neuron_clusters_observe(wubu_neuron_clusters_t *c,
                                 int layer, int neuron);

/* Hot set = top hot_percent of neurons by observed count. Returns the
 * number of hot neurons across all layers, or -1 on bad args. */
int wubu_neuron_clusters_partition(wubu_neuron_clusters_t *c, int hot_percent);

/* Runtime cluster sizing: neurons per cluster = max(1, target_bytes /
 * bytes_per_neuron). Returns the chosen cluster size. */
int wubu_neuron_clusters_set_target_bytes(wubu_neuron_clusters_t *c,
                                          int bytes_per_neuron,
                                          size_t target_bytes);

int  wubu_neuron_clusters_cluster_size(const wubu_neuron_clusters_t *c);
int  wubu_neuron_clusters_hot(const wubu_neuron_clusters_t *c,
                              int layer, int neuron);
uint64_t wubu_neuron_clusters_predicted_reads(const wubu_neuron_clusters_t *c,
                                              int tokens);
uint64_t wubu_neuron_clusters_observations(const wubu_neuron_clusters_t *c);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_NEURON_CLUSTER_H */
