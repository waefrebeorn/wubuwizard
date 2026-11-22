/* test_eg05_neuron.c -- EG-05 gate: activation-locality predictor, runtime
 * neuron-cluster sizing and cluster-level I/O accounting. Prints:
 * test_eg05_neuron: PASS. */
#include <stdio.h>

#include "wubu_neuron_cluster.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    enum { LAYERS = 2, NEURONS = 32 };
    wubu_neuron_clusters_t *c = wubu_neuron_clusters_create(LAYERS, NEURONS);
    CHECK(c != NULL);
    if (!c) return 1;

    /* Power-law activation: four neurons dominate each layer. */
    const int hot_counts[4] = { 100, 80, 60, 40 };
    for (int l = 0; l < LAYERS; l++) {
        for (int n = 0; n < 4; n++)
            for (int i = 0; i < hot_counts[n]; i++)
                CHECK(wubu_neuron_clusters_observe(c, l, n) == 0);
        for (int n = 4; n < NEURONS; n++)
            CHECK(wubu_neuron_clusters_observe(c, l, n) == 0);
    }
    printf("EG-05 observations=%llu\n",
           (unsigned long long)wubu_neuron_clusters_observations(c));
    CHECK(wubu_neuron_clusters_observations(c) == 616);

    int n_hot = wubu_neuron_clusters_partition(c, 10);
    printf("EG-05 hot_percent=10 hot_neurons=%d\n", n_hot);
    CHECK(n_hot == 8); /* 4 per layer */
    for (int l = 0; l < LAYERS; l++) {
        for (int n = 0; n < 4; n++)
            CHECK(wubu_neuron_clusters_hot(c, l, n) == 1);
        CHECK(wubu_neuron_clusters_hot(c, l, 10) == 0);
    }

    /* Runtime sizing from a target cluster byte budget. */
    CHECK(wubu_neuron_clusters_set_target_bytes(c, 4, 64) == 16);
    CHECK(wubu_neuron_clusters_cluster_size(c) == 16);
    CHECK(wubu_neuron_clusters_set_target_bytes(c, 4, 16) == 4);
    CHECK(wubu_neuron_clusters_cluster_size(c) == 4);

    /* Cluster-level reads: hot set removes cluster 0 from the fetch path. */
    uint64_t with_predictor = wubu_neuron_clusters_predicted_reads(c, 10);
    CHECK(wubu_neuron_clusters_partition(c, 0) == 0);
    uint64_t without_predictor = wubu_neuron_clusters_predicted_reads(c, 10);
    printf("EG-05 predicted_reads: hot=%llu no_hot=%llu (10 tokens)\n",
           (unsigned long long)with_predictor,
           (unsigned long long)without_predictor);
    CHECK(with_predictor == 140);
    CHECK(without_predictor == 160);
    CHECK(with_predictor < without_predictor);

    /* Fail-closed validation. */
    CHECK(wubu_neuron_clusters_observe(c, -1, 0) == -1);
    CHECK(wubu_neuron_clusters_observe(c, 0, NEURONS) == -1);
    CHECK(wubu_neuron_clusters_partition(c, 101) == -1);
    CHECK(wubu_neuron_clusters_set_target_bytes(c, 0, 64) == -1);
    CHECK(wubu_neuron_clusters_hot(c, LAYERS, 0) == -1);
    CHECK(wubu_neuron_clusters_predicted_reads(c, 0) == 0);

    wubu_neuron_clusters_free(c);
    if (failures) {
        fprintf(stderr, "test_eg05_neuron: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg05_neuron: PASS\n");
    return 0;
}
