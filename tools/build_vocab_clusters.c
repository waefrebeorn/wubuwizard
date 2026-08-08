/**
 * build_vocab_clusters.c — Cluster 248k output weight columns into groups.
 * Output: centroids file + group assignment file for hierarchical output projection.
 *
 * Hierarchical output proj flow:
 *   1. Compute h · centroid[0..99] → find winning group
 *   2. Compute h · W[:, group_columns] only (was 248k, now ~2500)
 *   3. Find argmax from subset
 *   4. If argmax found → done. If not → fall back to full output proj.
 *
 * Build: gcc -O3 -I include -o build_vocab_clusters build_vocab_clusters.c src/gguf_reader.o -lm
 * Usage: ./build_vocab_clusters ~/models/model.gguf centroids.bin groups.bin 100
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_CLUSTERS 256
#define D_MODEL 2048

static double wall_clock(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Simple random initialization of centroids
static void init_centroids(float *centroids, int n_clusters, int n_cols, const float *data) {
    for (int c = 0; c < n_clusters; c++) {
        int idx = rand() % n_cols;
        memcpy(&centroids[c * D_MODEL], &data[idx * D_MODEL], D_MODEL * sizeof(float));
    }
}

// k-means one iteration: assign each column to nearest centroid
// Returns total distance (for convergence check)
static double kmeans_iter(float *centroids, int *assignments, int n_clusters, int n_cols, const float *data) {
    double total_dist = 0;
    #pragma omp parallel for reduction(+:total_dist)
    for (int j = 0; j < n_cols; j++) {
        const float *col = &data[j * D_MODEL];
        int best_c = 0;
        float best_d = 0;
        for (int c = 0; c < n_clusters; c++) {
            const float *cent = &centroids[c * D_MODEL];
            float dot = 0;
            for (int k = 0; k < D_MODEL; k++) dot += col[k] * cent[k];
            if (c == 0 || dot > best_d) { best_d = dot; best_c = c; }
        }
        assignments[j] = best_c;
        total_dist += best_d;
    }
    // Update centroids
    float *new_cent = (float *)calloc(n_clusters * D_MODEL, sizeof(float));
    int *counts = (int *)calloc(n_clusters, sizeof(int));
    for (int j = 0; j < n_cols; j++) {
        int c = assignments[j];
        for (int k = 0; k < D_MODEL; k++) new_cent[c * D_MODEL + k] += data[j * D_MODEL + k];
        counts[c]++;
    }
    for (int c = 0; c < n_clusters; c++) {
        if (counts[c] > 0) {
            float inv = 1.0f / counts[c];
            for (int k = 0; k < D_MODEL; k++) centroids[c * D_MODEL + k] = new_cent[c * D_MODEL + k] * inv;
        }
    }
    free(new_cent);
    free(counts);
    return total_dist;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <model.gguf> <centroids.bin> <groups.bin> <n_clusters>\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *cent_path = argv[2];
    const char *group_path = argv[3];
    int n_clusters = atoi(argv[4]);
    if (n_clusters < 2 || n_clusters > MAX_CLUSTERS) {
        fprintf(stderr, "n_clusters 2-%d\n", MAX_CLUSTERS); return 1;
    }

    gguf_ctx *ctx = gguf_open(model_path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);

    // Find output.weight tensor
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    if (!t) { fprintf(stderr, "output.weight not found\n"); return 1; }
    
    int64_t n_rows = t->dims[0];  // D_MODEL
    int64_t n_cols = t->dims[1];  // vocab_size
    int type = t->ggml_type;
    
    printf("output.weight: [%lld, %lld] type=%d\n", (long long)n_rows, (long long)n_cols, type);
    
    // Dequantize to F32
    int64_t total = n_rows * n_cols;
    float *f32 = (float *)malloc(total * sizeof(float));
    const uint8_t *blob = (const uint8_t *)ctx->data_blob + t->data_offset;
    
    double t0 = wall_clock();
    gguf_dequantize(blob, type, total, f32);
    printf("Dequantized: %.1fs\n", wall_clock() - t0);
    
    // L2-normalize each column for cosine similarity
    #pragma omp parallel for
    for (int j = 0; j < n_cols; j++) {
        float *col = &f32[j * n_rows];
        float norm = 0;
        for (int k = 0; k < n_rows; k++) norm += col[k] * col[k];
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            float inv = 1.0f / norm;
            for (int k = 0; k < n_rows; k++) col[k] *= inv;
        }
    }
    printf("Normalized\n");
    
    // k-means clustering
    srand(42);
    float *centroids = (float *)calloc(n_clusters * D_MODEL, sizeof(float));
    int *assignments = (int *)malloc(n_cols * sizeof(int));
    
    init_centroids(centroids, n_clusters, (int)n_cols, f32);
    
    double prev_dist = 0;
    for (int iter = 0; iter < 20; iter++) {
        double t1 = wall_clock();
        double dist = kmeans_iter(centroids, assignments, n_clusters, (int)n_cols, f32);
        double change = (prev_dist > 0) ? fabs(dist - prev_dist) / prev_dist : 1.0;
        printf("Iter %d: dist=%.0f change=%.4f (%.1fs)\n", iter, dist, change, wall_clock() - t1);
        if (change < 0.001) break;
        prev_dist = dist;
    }
    
    // Save centroids
    FILE *fc = fopen(cent_path, "wb");
    if (!fc) { perror("centroids"); return 1; }
    fwrite(centroids, sizeof(float), n_clusters * D_MODEL, fc);
    fclose(fc);
    printf("Centroids saved: %s (%.1f MB)\n", cent_path, n_clusters * D_MODEL * 4.0 / 1e6);
    
    // Save group assignments (token_id → group_id)
    FILE *fg = fopen(group_path, "wb");
    if (!fg) { perror("groups"); return 1; }
    fwrite(assignments, sizeof(int), n_cols, fg);
    fclose(fg);
    printf("Groups saved: %s\n", group_path);
    
    // Stats per group
    int *counts = (int *)calloc(n_clusters, sizeof(int));
    for (int j = 0; j < n_cols; j++) counts[assignments[j]]++;
    printf("\nGroup sizes:\n");
    for (int c = 0; c < n_clusters; c++)
        printf("  Group %d: %d tokens\n", c, counts[c]);
    free(counts);
    
    free(centroids);
    free(assignments);
    free(f32);
    gguf_close(ctx);
    return 0;
}
