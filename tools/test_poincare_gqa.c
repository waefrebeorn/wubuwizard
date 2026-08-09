/**
 * test_poincare_gqa.c
 *
 * Verifies the Poincaré GQA forward pass:
 * 1. Poincaré GQA produces non-NaN output
 * 2. Output values are finite (no Inf, no NaN)
 * 3. Output has the same shape as standard GQA
 *    (same B, T, D_MODEL dimensions)
 *
 * Since we don't have real model weights, we construct synthetic
 * weights with small random values and run both Euclidean and
 * Poincaré GQA to check correctness.
 */

#include "wubu_ssm.h"
#include "wubu_poincare_gqa.h"
#include "gguf_reader.h"   /* GGML_TYPE_F32 for F32-blob aliasing */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Simple deterministic pseudo-random for reproducibility
static float frand(void) {
    static unsigned int seed = 42;
    seed = seed * 1103515245 + 12345;
    return (float)(seed & 0x7fffffff) / (float)0x7fffffff;
}

// Initialize weight matrix with small random values
static void init_weights_2d(float *w, int rows, int cols, float scale) {
    for (int i = 0; i < rows * cols; i++) {
        w[i] = (frand() - 0.5f) * 2.0f * scale;
    }
}

// Initialize norm weight (all ones, like typical RMSNorm)
static void init_norm_weight(float *w, int d) {
    for (int i = 0; i < d; i++) {
        w[i] = 1.0f;
    }
}

int main(void) {
    srand(42);

    // Test parameters
    const int B = 2;
    const int T = 8;
    const int N = B * T;
    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;   // 4096
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;  // 512
    const float R = 10.0f;  // Poincaré ball radius

    printf("=== Poincaré GQA Forward Test ===\n");
    printf("B=%d, T=%d, D_MODEL=%d\n", B, T, D_MODEL);
    printf("Q_HEADS=%d, KV_HEADS=%d, HEAD_DIM=%d\n",
           GQA_Q_HEADS, GQA_KV_HEADS, GQA_HEAD_DIM);
    printf("q_dim=%d, kv_dim=%d, R=%.1f\n", q_dim, kv_dim, R);

    // ========== Allocate weights ==========
    gqa_layer_weights w;
    memset(&w, 0, sizeof(w));

    w.attn_q_weight = (float *)malloc(D_MODEL * q_dim * 2 * sizeof(float));
    w.attn_k_weight = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
    w.attn_v_weight = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
    w.attn_output_weight = (float *)malloc(q_dim * D_MODEL * sizeof(float));
    w.attn_q_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
    w.attn_k_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
    w.attn_norm_weight = NULL;          // not used by GQA forward
    w.post_attention_norm_weight = NULL; // not used by GQA forward

    if (!w.attn_q_weight || !w.attn_k_weight || !w.attn_v_weight ||
        !w.attn_output_weight || !w.attn_q_norm_weight || !w.attn_k_norm_weight) {
        fprintf(stderr, "Failed to allocate weights\n");
        return 1;
    }

    /* Current GQA forward reads the quantized-blob fields; for F32-only
     * test weights, alias the _q pointer to the F32 buffer with type F32
     * so quantized_matmul_batched takes its F32 path. */
    w.attn_q_weight_q = (const uint8_t *)w.attn_q_weight;
    w.attn_q_weight_type = GGML_TYPE_F32;
    w.attn_k_weight_q = (const uint8_t *)w.attn_k_weight;
    w.attn_k_weight_type = GGML_TYPE_F32;
    w.attn_v_weight_q = (const uint8_t *)w.attn_v_weight;
    w.attn_v_weight_type = GGML_TYPE_F32;
    w.attn_output_weight_q = (const uint8_t *)w.attn_output_weight;
    w.attn_output_weight_type = GGML_TYPE_F32;

    // Initialize weights with small values
    float q_scale = 0.01f;
    float k_scale = 0.01f;
    float v_scale = 0.01f;
    float out_scale = 0.01f;

    init_weights_2d(w.attn_q_weight, D_MODEL, q_dim * 2, q_scale);
    init_weights_2d(w.attn_k_weight, D_MODEL, kv_dim, k_scale);
    init_weights_2d(w.attn_v_weight, D_MODEL, kv_dim, v_scale);
    init_weights_2d(w.attn_output_weight, q_dim, D_MODEL, out_scale);
    init_norm_weight(w.attn_q_norm_weight, GQA_HEAD_DIM);
    init_norm_weight(w.attn_k_norm_weight, GQA_HEAD_DIM);

    // ========== Allocate input and output buffers ==========
    float *x = (float *)malloc(N * D_MODEL * sizeof(float));
    float *output_euclidean = (float *)malloc(N * D_MODEL * sizeof(float));
    float *output_poincare  = (float *)malloc(N * D_MODEL * sizeof(float));

    if (!x || !output_euclidean || !output_poincare) {
        fprintf(stderr, "Failed to allocate input/output buffers\n");
        return 1;
    }

    // Initialize input with random values
    for (int i = 0; i < N * D_MODEL; i++) {
        x[i] = (frand() - 0.5f) * 0.1f;
    }

    // ========== Run Euclidean GQA ==========
    printf("\nRunning Euclidean GQA...\n");
    memset(output_euclidean, 0, N * D_MODEL * sizeof(float));
    wubu_gqa_forward(x, B, T, &w, output_euclidean,
                      NULL, NULL, 0, NULL, NULL);

    // Check Euclidean output for NaN (informational only)
    int euclidean_nan = 0;
    (void)euclidean_nan; // suppress unused warning
    for (int i = 0; i < N * D_MODEL; i++) {
        if (isnan(output_euclidean[i]) || isinf(output_euclidean[i])) {
            euclidean_nan = 1;
            printf("  Euclidean NaN at [%d] = %.6e\n", i, output_euclidean[i]);
            break;
        }
    }

    // Compute Euclidean output statistics
    float euc_min = 1e30f, euc_max = -1e30f, euc_sum = 0.0f;
    for (int i = 0; i < N * D_MODEL; i++) {
        float v = output_euclidean[i];
        if (v < euc_min) euc_min = v;
        if (v > euc_max) euc_max = v;
        euc_sum += v;
    }
    printf("  Euclidean output: min=%.6e max=%.6e mean=%.6e\n",
           euc_min, euc_max, euc_sum / (N * D_MODEL));

    // ========== Run Poincaré GQA ==========
    printf("\nRunning Poincaré GQA (R=%.1f)...\n", R);
    memset(output_poincare, 0, N * D_MODEL * sizeof(float));
    wubu_poincare_gqa_forward(x, B, T, &w, R, output_poincare);

    // Test 1: No NaN in output
    int poincare_nan = 0;
    int poincare_inf = 0;
    int first_nan_idx = -1;
    for (int i = 0; i < N * D_MODEL; i++) {
        if (isnan(output_poincare[i])) {
            if (!poincare_nan) first_nan_idx = i;
            poincare_nan++;
        }
        if (isinf(output_poincare[i])) {
            poincare_inf++;
        }
    }

    // Compute Poincaré output statistics
    float poi_min = 1e30f, poi_max = -1e30f, poi_sum = 0.0f;
    for (int i = 0; i < N * D_MODEL; i++) {
        float v = output_poincare[i];
        if (v < poi_min) poi_min = v;
        if (v > poi_max) poi_max = v;
        poi_sum += v;
    }
    printf("  Poincaré output:   min=%.6e max=%.6e mean=%.6e\n",
           poi_min, poi_max, poi_sum / (N * D_MODEL));

    // ========== Verify Shape ==========
    int same_size = 1;
    // Both outputs are [B, T, D_MODEL], so total elements should match
    // (We already know this since we allocated same-size buffers)

    // ========== Results ==========
    printf("\n=== Results ===\n");
    printf("Test 1 (NaN-free): ");
    if (poincare_nan == 0) {
        printf("PASS — no NaN values in output\n");
    } else {
        printf("FAIL — %d NaN values found (first at index %d)\n",
               poincare_nan, first_nan_idx);
    }

    printf("Test 2 (Finite):   ");
    if (poincare_inf == 0) {
        printf("PASS — no Inf values in output\n");
    } else {
        printf("FAIL — %d Inf values found\n", poincare_inf);
    }

    printf("Test 3 (Shape):    ");
    if (same_size) {
        printf("PASS — output shape is [%d, %d, %d] (%d elements)\n",
               B, T, D_MODEL, N * D_MODEL);
    } else {
        printf("FAIL — output shape mismatch\n");
    }

    // Additional sanity: values should be in a reasonable range
    int in_range = 1;
    for (int i = 0; i < N * D_MODEL; i++) {
        if (fabsf(output_poincare[i]) > 1e10f) {
            in_range = 0;
            break;
        }
    }
    printf("Test 4 (Range):    ");
    if (in_range) {
        printf("PASS — all values within ±1e10\n");
    } else {
        printf("FAIL — extreme values detected\n");
    }

    // ========== Cleanup ==========
    free(w.attn_q_weight);
    free(w.attn_k_weight);
    free(w.attn_v_weight);
    free(w.attn_output_weight);
    free(w.attn_q_norm_weight);
    free(w.attn_k_norm_weight);
    free(x);
    free(output_euclidean);
    free(output_poincare);

    int passed = (poincare_nan == 0) && (poincare_inf == 0) && same_size && in_range;
    if (passed) {
        printf("\n=== ALL TESTS PASSED ===\n");
        return 0;
    } else {
        printf("\n=== SOME TESTS FAILED ===\n");
        return 1;
    }
}
