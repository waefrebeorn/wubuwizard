#include "wubu_moe_hyperbolic.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include "wubu_activations.h"

// ============================================================
// Helper: map Euclidean vector to Poincaré ball via exp_map
// ============================================================
void euclidean_to_poincare_ball(const float *v, int d, float R, float *out) {
    // exp_map(v) = tanh(||v||/R) * v/||v|| * R
    float norm = 0.0f;
    for (int i = 0; i < d; i++) norm += v[i] * v[i];
    norm = sqrtf(norm);

    if (norm < 1e-30f) {
        // Zero vector maps to origin
        memset(out, 0, d * sizeof(float));
        return;
    }

    float ratio = norm / R;
    if (ratio >= 0.9999f) ratio = 0.9999f;  // clamp for tanh stability
    float tanh_r = tanhf(ratio);
    float scale = tanh_r * R / norm;

    for (int i = 0; i < d; i++) {
        out[i] = v[i] * scale;
    }
}

// ============================================================
// Softmax helper
// ============================================================
static void softmax_inplace(float *vals, int n) {
    // Find max for stability
    float maxv = vals[0];
    for (int i = 1; i < n; i++)
        if (vals[i] > maxv) maxv = vals[i];

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        vals[i] = expf(vals[i] - maxv);
        sum += vals[i];
    }

    float inv_sum = 1.0f / (sum + 1e-30f);
    for (int i = 0; i < n; i++)
        vals[i] *= inv_sum;
}

// ============================================================
// Top-k selection (indices + values from an array)
// ============================================================
void topk_from_array(const float *vals, int n, int k,
                     int *out_indices, float *out_vals) {
    // Simple repeated max search (n is small: 16 or 256)
    int used[256]; // max 256 experts
    if (n > 256) n = 256;
    memset(used, 0, n * sizeof(int));

    for (int t = 0; t < k; t++) {
        int best_idx = -1;
        float best_val = -1e30f;
        for (int i = 0; i < n; i++) {
            if (!used[i] && vals[i] > best_val) {
                best_val = vals[i];
                best_idx = i;
            }
        }
        if (best_idx >= 0) {
            out_indices[t] = best_idx;
            out_vals[t] = best_val;
            used[best_idx] = 1;
        } else {
            out_indices[t] = 0;
            out_vals[t] = 0.0f;
        }
    }
}

// ============================================================
// Poincaré distance router initialization (synthetic)
// ============================================================
void wubu_poincare_router_init_random(float *centroids_out, unsigned int seed) {
    srand(seed);
    int n = N_EXPERTS * D_MODEL;
    for (int i = 0; i < n; i++) {
        // Random in [-0.5, 0.5], scaled to ~R/2 to stay well inside the ball
        centroids_out[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * R_POINCARE_INPUT * 0.8f;
    }
}

// ============================================================
// Poincaré distance router forward
// ============================================================
void wubu_poincare_router_forward(const float *x, int B, int T,
                                  const poincare_router_t *router,
                                  float *scores) {
    if (!router || !router->loaded) return;
    int N = B * T;
    int d = D_MODEL;
    float R = R_POINCARE_INPUT;
    float temp = router->temperature;
    const float *centroids = router->centroids;

    // Per-token scratch: mapped input + per-expert distances
    float *x_ball = (float *)malloc(d * sizeof(float));
    float *dists = (float *)malloc(N_EXPERTS * sizeof(float));

    if (!x_ball || !dists) {
        fprintf(stderr, "Poincaré router: allocation failed\n");
        free(x_ball); free(dists);
        return;
    }

    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * d;
        float *score_s = scores + s * N_EXPERTS;

        // Step 1: Map input to Poincaré ball
        euclidean_to_poincare_ball(x_s, d, R, x_ball);

        // Step 2: Compute Poincaré distance to each centroid
        for (int e = 0; e < N_EXPERTS; e++) {
            const float *cent_e = centroids + e * d;
            dists[e] = wubu_poincare_dist(x_ball, cent_e, d, R);
        }

        // Step 3: Convert distances to scores: score = -distance / temperature
        for (int e = 0; e < N_EXPERTS; e++) {
            score_s[e] = -dists[e] / temp;
        }
    }

    free(x_ball);
    free(dists);
}

// ============================================================
// Nested MoE router initialization (synthetic)
// ============================================================
void wubu_nested_moe_router_init_random(float *coarse_out, float *fine_out,
                                        unsigned int seed) {
    srand(seed);
    int n_coarse = N_HYPERBOLIC_GROUPS * D_MODEL;
    int n_fine = N_EXPERTS * D_MODEL;

    // Coarse centroids: randomly placed in large ball (R=1.5)
    for (int i = 0; i < n_coarse; i++) {
        coarse_out[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * R_POINCARE_COARSE * 0.8f;
    }

    // Fine centroids: randomly placed in small ball (R=0.5)
    for (int i = 0; i < n_fine; i++) {
        fine_out[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * R_POINCARE_FINE * 0.8f;
    }
}

// ============================================================
// Two-level hierarchical routing
// ============================================================
void wubu_nested_moe_router_forward(const float *x, int B, int T,
                                    const nested_moe_router_t *router,
                                    int *out_indices, float *out_weights) {
    if (!router || !router->loaded) return;
    int N = B * T;
    int d = D_MODEL;
    float temp = router->temperature;

    // Scratch buffers
    float *x_ball_coarse = (float *)malloc(d * sizeof(float));
    float *x_ball_fine = (float *)malloc(d * sizeof(float));
    float *coarse_dists = (float *)malloc(N_HYPERBOLIC_GROUPS * sizeof(float));
    float *coarse_scores = (float *)malloc(N_HYPERBOLIC_GROUPS * sizeof(float));
    float *fine_dists = (float *)malloc(N_EXPERTS_PER_GROUP * sizeof(float));
    float *fine_scores = (float *)malloc(N_EXPERTS_PER_GROUP * sizeof(float));
    float *all_scores = (float *)malloc(N_EXPERTS * sizeof(float));
    int *all_indices = (int *)malloc(N_EXPERTS * sizeof(int));

    if (!x_ball_coarse || !x_ball_fine || !coarse_dists || !coarse_scores ||
        !fine_dists || !fine_scores || !all_scores || !all_indices) {
        fprintf(stderr, "Nested MoE router: allocation failed\n");
        free(x_ball_coarse); free(x_ball_fine);
        free(coarse_dists); free(coarse_scores);
        free(fine_dists); free(fine_scores);
        free(all_scores); free(all_indices);
        return;
    }

    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * d;
        int *inds_s = out_indices + s * N_ACTIVE_EXPTS;
        float *wts_s = out_weights + s * N_ACTIVE_EXPTS;

        // Map input to both balls (same vector, different radii)
        euclidean_to_poincare_ball(x_s, d, R_POINCARE_COARSE, x_ball_coarse);
        euclidean_to_poincare_ball(x_s, d, R_POINCARE_FINE, x_ball_fine);

        // --- Level 1: Coarse routing ---
        // Compute Poincaré distance to each of 16 coarse centroids
        for (int g = 0; g < N_HYPERBOLIC_GROUPS; g++) {
            const float *cent_g = router->coarse_centroids + g * d;
            coarse_dists[g] = wubu_poincare_dist(x_ball_coarse, cent_g, d, R_POINCARE_COARSE);
        }

        // Convert to scores: -dist / temp
        for (int g = 0; g < N_HYPERBOLIC_GROUPS; g++) {
            coarse_scores[g] = -coarse_dists[g] / temp;
        }
        softmax_inplace(coarse_scores, N_HYPERBOLIC_GROUPS);

        // Select top-2 groups (n_candidates = 2 groups × 16 experts = 32)
        int sel_groups[2];
        float sel_group_scores[2];
        topk_from_array(coarse_scores, N_HYPERBOLIC_GROUPS, 2, sel_groups, sel_group_scores);

        // --- Level 2: Fine routing within selected groups ---
        int n_candidates = 0;

        for (int g_idx = 0; g_idx < 2; g_idx++) {
            int group = sel_groups[g_idx];
            float group_score = sel_group_scores[g_idx];

            for (int e = 0; e < N_EXPERTS_PER_GROUP; e++) {
                int expert_id = group * N_EXPERTS_PER_GROUP + e;
                const float *cent_e = router->fine_centroids + expert_id * d;

                float dist = wubu_poincare_dist(x_ball_fine, cent_e, d, R_POINCARE_FINE);
                float score = -dist / temp;

                // Final score = group_score * 0.3 + expert_score * 0.7 (blend)
                all_scores[n_candidates] = 0.3f * group_score + 0.7f * score;
                all_indices[n_candidates] = expert_id;
                n_candidates++;
            }
        }

        // Apply softmax to candidate scores to get positive weights summing to 1
        if (n_candidates > 0) {
            float max_s = all_scores[0];
            for (int i = 1; i < n_candidates; i++)
                if (all_scores[i] > max_s) max_s = all_scores[i];
            float sum_exp = 0.0f;
            for (int i = 0; i < n_candidates; i++) {
                all_scores[i] = expf(all_scores[i] - max_s);
                sum_exp += all_scores[i];
            }
            float inv_sum_exp = 1.0f / (sum_exp + 1e-30f);
            for (int i = 0; i < n_candidates; i++)
                all_scores[i] *= inv_sum_exp;
        }

        // Select top-8 from 32 candidates
        if (n_candidates == 0) {
            // Fallback: select first N_ACTIVE_EXPTS experts
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                inds_s[k] = k;
                wts_s[k] = 1.0f / N_ACTIVE_EXPTS;
            }
        } else {
            // Selection
            int used[32]; // max 32 candidates
            memset(used, 0, n_candidates * sizeof(int));

            for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                int best_idx = -1;
                float best_val = -1e30f;
                for (int i = 0; i < n_candidates; i++) {
                    if (!used[i] && all_scores[i] > best_val) {
                        best_val = all_scores[i];
                        best_idx = i;
                    }
                }
                if (best_idx >= 0) {
                    inds_s[k] = all_indices[best_idx];
                    wts_s[k] = best_val;
                    used[best_idx] = 1;
                } else {
                    inds_s[k] = 0;
                    wts_s[k] = 0.0f;
                }
            }

            // Normalize weights to sum to 1 (handle negative scores)
            float sum_w = 0.0f;
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) sum_w += wts_s[k];
            if (fabsf(sum_w) > 1e-30f) {
                float inv_sum = 1.0f / sum_w;
                for (int k = 0; k < N_ACTIVE_EXPTS; k++) wts_s[k] *= inv_sum;
            }
        }
    }

    free(x_ball_coarse);
    free(x_ball_fine);
    free(coarse_dists);
    free(coarse_scores);
    free(fine_dists);
    free(fine_scores);
    free(all_scores);
    free(all_indices);
}

// ============================================================
// Free functions
// ============================================================
void wubu_poincare_router_free(poincare_router_t *router) {
    if (!router) return;
    free(router->centroids);
    router->centroids = NULL;
    router->loaded = false;
}

void wubu_nested_moe_router_free(nested_moe_router_t *router) {
    if (!router) return;
    free(router->coarse_centroids);
    free(router->fine_centroids);
    router->coarse_centroids = NULL;
    router->fine_centroids = NULL;
    router->loaded = false;
}
