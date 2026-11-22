/*
 * wubu_offload_plan.c -- EG-06 placement solver + zigzag schedule. Pure C11.
 */
#include "wubu_offload_plan.h"

#include <stdlib.h>
#include <string.h>

typedef struct { uint64_t hotness; int index; } rank_t;

static int rank_cmp(const void *a, const void *b) {
    const rank_t *x = (const rank_t *)a;
    const rank_t *y = (const rank_t *)b;
    if (x->hotness > y->hotness) return -1;
    if (x->hotness < y->hotness) return 1;
    return x->index < y->index ? -1 : (x->index > y->index ? 1 : 0);
}

int wubu_offload_solve(const wubu_offload_hw_t *hw,
                       const wubu_offload_layer_t *layers, int n,
                       int *assign_out) {
    rank_t *rank;
    uint64_t used[WUBU_OFFLOAD_TIERS] = {0, 0, 0};
    int i, t, placed = 0;

    if (!hw || !layers || !assign_out || n <= 0) return -1;
    for (t = 0; t < WUBU_OFFLOAD_TIERS; t++)
        if (hw->bandwidth_bps[t] == 0) return -1;

    rank = (rank_t *)malloc((size_t)n * sizeof(rank_t));
    if (!rank) return -1;
    for (i = 0; i < n; i++) {
        rank[i].hotness = layers[i].hotness;
        rank[i].index = i;
        assign_out[i] = -1;
    }
    qsort(rank, (size_t)n, sizeof(rank_t), rank_cmp);

    /* Hot-first: every layer takes the fastest tier that still has room. */
    for (i = 0; i < n; i++) {
        int li = rank[i].index;
        for (t = 0; t < WUBU_OFFLOAD_TIERS; t++) {
            if (used[t] + layers[li].bytes <= hw->capacity[t]) {
                used[t] += layers[li].bytes;
                assign_out[li] = t;
                placed++;
                break;
            }
        }
    }
    free(rank);
    return placed == n ? 0 : -1;
}

double wubu_offload_cost(const wubu_offload_hw_t *hw,
                         const wubu_offload_layer_t *layers, int n,
                         const int *assign) {
    double total = 0.0;
    int i;
    if (!hw || !layers || !assign || n <= 0) return -1.0;
    for (i = 0; i < n; i++) {
        int t = assign[i];
        if (t < 0 || t >= WUBU_OFFLOAD_TIERS ||
            hw->bandwidth_bps[t] == 0) return -1.0;
        total += (double)layers[i].bytes / (double)hw->bandwidth_bps[t];
    }
    return total;
}

int wubu_offload_zigzag(int n, int iters, int *out, int cap) {
    int k = 0, it, i;
    if (!out || n <= 0 || iters <= 0 || cap < 2 * n * iters) return -1;
    for (it = 0; it < iters; it++) {
        for (i = 0; i < n; i++) out[k++] = i;             /* forward */
        for (i = n - 1; i >= 0; i--) out[k++] = i;        /* reverse */
    }
    return k;
}
