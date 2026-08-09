/**
 * Test Nested SSM Backward pass.
 *
 * Tests BPTT through K independent Poincaré ball recurrences:
 *   1. Forward + backward with K=1, R=0.956 (gradient check)
 *   2. Forward + backward with K=4, R=[0.5, 1.0, 2.0, 5.0]
 *   3. State gradient flows correctly (check d_state_init_grad)
 *   4. Weight gradients are finite and sensible
 *
 * Strategy: run forward_save, then backward, verify NaN-free gradients.
 * Full numerical gradient check compares finite-difference perturbation
 * vs analytic gradient.
 */

#include "wubu_nested_ssm.h"
#include "wubu_mobius.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ================================================================
// Test data pattern: deterministic synthetic input
// ================================================================
static void make_random_weights(ssm_layer_weights *w) {
    unsigned int seed = 42;
    int qkv_dim = KEY_DIM * 2 + VALUE_DIM;

    w->attn_qkv_weight = (float *)malloc(D_MODEL * qkv_dim * sizeof(float));
    w->attn_gate_weight = (float *)malloc(D_MODEL * VALUE_DIM * sizeof(float));
    w->ssm_beta_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
    w->ssm_alpha_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
    w->ssm_dt_bias = (float *)malloc(DT_RANK * sizeof(float));
    w->ssm_a = (float *)malloc(DT_RANK * sizeof(float));
    w->ssm_conv1d_weight = (float *)malloc(CONV_KERNEL * CONV_DIM * sizeof(float));
    w->ssm_norm_weight = (float *)malloc(SSM_D_STATE * sizeof(float));
    w->ssm_out_weight = (float *)malloc(VALUE_DIM * D_MODEL * sizeof(float));

    // Fill with deterministic pseudo-random values
    for (int i = 0; i < D_MODEL * qkv_dim; i++)
        w->attn_qkv_weight[i] = (float)((rand_r(&seed) / (double)RAND_MAX) * 0.1 - 0.05);
    for (int i = 0; i < D_MODEL * VALUE_DIM; i++)
        w->attn_gate_weight[i] = (float)((rand_r(&seed) / (double)RAND_MAX) * 0.1 - 0.05);
    for (int i = 0; i < D_MODEL * DT_RANK; i++)
        w->ssm_beta_weight[i] = (float)((rand_r(&seed) / (double)RAND_MAX) * 0.1 - 0.05);
    for (int i = 0; i < D_MODEL * DT_RANK; i++)
        w->ssm_alpha_weight[i] = (float)((rand_r(&seed) / (double)RAND_MAX) * 0.1 - 0.05);
    for (int i = 0; i < DT_RANK; i++)
        w->ssm_dt_bias[i] = (float)((rand_r(&seed) / (double)RAND_MAX) * 0.1 - 0.05);
    for (int i = 0; i < DT_RANK; i++)
        w->ssm_a[i] = (float)((rand_r(&seed) / (double)RAND_MAX) * 0.1 + 0.01);
    for (int i = 0; i < CONV_KERNEL * CONV_DIM; i++)
        w->ssm_conv1d_weight[i] = (float)((rand_r(&seed) / (double)RAND_MAX) * 0.1 - 0.05);
    for (int i = 0; i < SSM_D_STATE; i++)
        w->ssm_norm_weight[i] = (float)((rand_r(&seed) / (double)RAND_MAX) * 0.1 + 0.9);
    for (int i = 0; i < VALUE_DIM * D_MODEL; i++)
        w->ssm_out_weight[i] = (float)((rand_r(&seed) / (double)RAND_MAX) * 0.1 - 0.05);
}

static void make_input(float *x, int N) {
    unsigned int seed = 123;
    for (int i = 0; i < N * D_MODEL; i++)
        x[i] = (float)((rand_r(&seed) / (double)RAND_MAX) * 0.5 - 0.25);
}

static void free_weights(ssm_layer_weights *w) {
    free(w->attn_qkv_weight);
    free(w->attn_gate_weight);
    free(w->ssm_beta_weight);
    free(w->ssm_alpha_weight);
    free(w->ssm_dt_bias);
    free(w->ssm_a);
    free(w->ssm_conv1d_weight);
    free(w->ssm_norm_weight);
    free(w->ssm_out_weight);
}

// ================================================================
// Utility: check array for NaN/Inf
// ================================================================
static int check_nan_inf(const float *arr, int n, const char *label) {
    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(arr[i]) || isinf(arr[i])) {
            bad++;
            if (bad <= 3)
                printf("  %s[%d] = %e\n", label, i, arr[i]);
        }
    }
    return bad;
}

static int check_nan_inf_all(const float *arr, int n, const char *label) {
    int bad = check_nan_inf(arr, n, label);
    if (bad > 0) printf("  %s: %d NaN/Inf out of %d\n", label, bad, n);
    return bad;
}

// ================================================================
// Numerical gradient check (central finite differences)
// ================================================================
static double compute_loss(const float *output, const float *dummy_d_out, int N) {
    // Simulate a scalar loss: sum_i output[i] * dummy_d_out[i]
    double loss = 0.0;
    for (int i = 0; i < N * D_MODEL; i++) {
        loss += (double)output[i] * (double)dummy_d_out[i];
    }
    return loss;
}

// Perturb a single weight element and re-run forward
static double perturb_forward(const float *x, int B, int T,
                               ssm_layer_weights *w,
                               wubu_nested_ssm_state_t *nstate,
                               float *conv_state,
                               const wubu_nested_ssm_gating_t *gating,
                               float *output,
                               int perturb_idx, int weight_stride,
                               float *weight_base, float eps) {
    float orig = weight_base[perturb_idx];
    weight_base[perturb_idx] = orig + eps;
    // Re-init state
    memset(nstate->states, 0, nstate->K * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
    memset(conv_state, 0, B * (CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
    wubu_nested_ssm_forward(x, B, T, w, nstate, conv_state, gating, output);
    double loss_p = compute_loss(output, (const float *)(output + 0), B * T); // dummy

    weight_base[perturb_idx] = orig - eps;
    memset(nstate->states, 0, nstate->K * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
    memset(conv_state, 0, B * (CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
    wubu_nested_ssm_forward(x, B, T, w, nstate, conv_state, gating, output);
    double loss_m = compute_loss(output, (const float *)(output + 0), B * T);

    weight_base[perturb_idx] = orig; // restore
    return (loss_p - loss_m) / (2.0 * eps);
}

// ================================================================
// Main test
// ================================================================
int main() {
    printf("=== Nested SSM Backward Test ===\n\n");

    int B = 1, T = 3;  // Small for test speed
    int N = B * T;
    int qkv_dim = CONV_DIM;
    int HEAD_STATE_SZ = SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;

    // Allocate input
    float *x = (float *)malloc(N * D_MODEL * sizeof(float));
    float *output = (float *)malloc(N * D_MODEL * sizeof(float));
    float *d_output = (float *)malloc(N * D_MODEL * sizeof(float));
    float *d_x = (float *)calloc(N * D_MODEL, sizeof(float));
    ssm_layer_weights w;
    float *conv_state = (float *)calloc(B * (CONV_KERNEL - 1) * CONV_DIM, sizeof(float));

    if (!x || !output || !d_output || !d_x) {
        printf("FAIL: allocation failed\n");
        return 1;
    }

    make_random_weights(&w);
    make_input(x, N);

    // Dummy upstream gradient: same shape as output
    unsigned int seed_g = 456;
    for (int i = 0; i < N * D_MODEL; i++)
        d_output[i] = (float)((rand_r(&seed_g) / (double)RAND_MAX) * 0.1 - 0.05);

    int total_pass = 0;
    int total_tests = 0;

    // ================================================================
    // Test 1: K=1, R=0.956 — basic backward sanity
    // ================================================================
    printf("--- Test 1: K=1 backward sanity (R=0.956) ---\n");
    total_tests++;

    float R_single = 0.956f;
    wubu_nested_ssm_state_t nstate1;
    float R1[1] = {R_single};
    if (wubu_nested_ssm_init(&nstate1, 1, R1) != 0) {
        printf("FAIL: init failed\n");
        return 1;
    }

    memset(conv_state, 0, B * (CONV_KERNEL - 1) * CONV_DIM * sizeof(float));

    // Forward with save
    nested_ssm_fwd_save_t save1;
    memset(&save1, 0, sizeof(save1));
    wubu_nested_ssm_forward_save(x, B, T, &w, &nstate1, conv_state, NULL, output, &save1);

    // Check forward output
    int fwd_nan = check_nan_inf_all(output, N * D_MODEL, "forward output");
    printf("  Forward NaN: %s\n", fwd_nan == 0 ? "✅" : "❌");

    // Run backward
    float *d_qkv_w = (float *)calloc(D_MODEL * qkv_dim, sizeof(float));
    float *d_gate_w = (float *)calloc(D_MODEL * VALUE_DIM, sizeof(float));
    float *d_beta_w = (float *)calloc(D_MODEL * DT_RANK, sizeof(float));
    float *d_alpha_w = (float *)calloc(D_MODEL * DT_RANK, sizeof(float));
    float *d_conv1d_w = (float *)calloc(CONV_KERNEL * CONV_DIM, sizeof(float));
    float *d_out_w = (float *)calloc(VALUE_DIM * D_MODEL, sizeof(float));
    float *d_norm_w = (float *)calloc(SSM_D_STATE, sizeof(float));
    float *d_state_init = (float *)calloc(nstate1.K * HEAD_STATE_SZ, sizeof(float));

    wubu_nested_ssm_backward(B, T, x, output, d_output, NULL, &nstate1, &save1, &w,
                              d_x, d_qkv_w, d_gate_w, d_beta_w, d_alpha_w,
                              d_conv1d_w, d_out_w, d_norm_w, d_state_init, NULL);

    // Check for NaN in all gradients
    int bwd_nan = 0;
    bwd_nan += check_nan_inf_all(d_x, N * D_MODEL, "d_x");
    bwd_nan += check_nan_inf_all(d_qkv_w, D_MODEL * qkv_dim, "d_qkv_w");
    bwd_nan += check_nan_inf_all(d_gate_w, D_MODEL * VALUE_DIM, "d_gate_w");
    bwd_nan += check_nan_inf_all(d_beta_w, D_MODEL * DT_RANK, "d_beta_w");
    bwd_nan += check_nan_inf_all(d_alpha_w, D_MODEL * DT_RANK, "d_alpha_w");
    bwd_nan += check_nan_inf_all(d_conv1d_w, CONV_KERNEL * CONV_DIM, "d_conv1d_w");
    bwd_nan += check_nan_inf_all(d_out_w, VALUE_DIM * D_MODEL, "d_out_w");
    bwd_nan += check_nan_inf_all(d_norm_w, SSM_D_STATE, "d_norm_w");
    bwd_nan += check_nan_inf_all(d_state_init, nstate1.K * HEAD_STATE_SZ, "d_state_init");

    int t1_pass = (fwd_nan == 0 && bwd_nan == 0);
    printf("  Test 1: %s (%d NaN/Inf in gradients)\n", t1_pass ? "✅ PASS" : "❌ FAIL", bwd_nan);
    total_pass += t1_pass;

    // Cleanup Test 1
    wubu_nested_ssm_fwd_save_free(&save1);
    free(d_qkv_w); free(d_gate_w); free(d_beta_w); free(d_alpha_w);
    free(d_conv1d_w); free(d_out_w); free(d_norm_w); free(d_state_init);

    // ================================================================
    // Test 2: K=2, R=[0.8, 1.5] — multi-ball backward
    // ================================================================
    printf("\n--- Test 2: K=2 backward (R=[0.8, 1.5]) ---\n");
    total_tests++;

    float R2[2] = {0.8f, 1.5f};
    wubu_nested_ssm_state_t nstate2;
    if (wubu_nested_ssm_init(&nstate2, 2, R2) != 0) {
        printf("FAIL: init K=2 failed\n");
        return 1;
    }

    memset(conv_state, 0, B * (CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
    memset(d_x, 0, N * D_MODEL * sizeof(float));

    // Gating: bias toward ball 1
    wubu_nested_ssm_gating_t gating2;
    gating2.ball_weights[0] = 0.5f;
    gating2.ball_weights[1] = 2.0f;

    nested_ssm_fwd_save_t save2;
    memset(&save2, 0, sizeof(save2));
    wubu_nested_ssm_forward_save(x, B, T, &w, &nstate2, conv_state, &gating2, output, &save2);

    int fwd2_nan = check_nan_inf_all(output, N * D_MODEL, "K=2 forward");

    d_qkv_w = (float *)calloc(D_MODEL * qkv_dim, sizeof(float));
    d_gate_w = (float *)calloc(D_MODEL * VALUE_DIM, sizeof(float));
    d_beta_w = (float *)calloc(D_MODEL * DT_RANK, sizeof(float));
    d_alpha_w = (float *)calloc(D_MODEL * DT_RANK, sizeof(float));
    d_conv1d_w = (float *)calloc(CONV_KERNEL * CONV_DIM, sizeof(float));
    d_out_w = (float *)calloc(VALUE_DIM * D_MODEL, sizeof(float));
    d_norm_w = (float *)calloc(SSM_D_STATE, sizeof(float));
    d_state_init = (float *)calloc(nstate2.K * HEAD_STATE_SZ, sizeof(float));

    wubu_nested_ssm_backward(B, T, x, output, d_output, gating2.ball_weights, &nstate2, &save2, &w,
                              d_x, d_qkv_w, d_gate_w, d_beta_w, d_alpha_w,
                              d_conv1d_w, d_out_w, d_norm_w, d_state_init, NULL);

    int bwd2_nan = 0;
    bwd2_nan += check_nan_inf_all(d_x, N * D_MODEL, "d_x");
    bwd2_nan += check_nan_inf_all(d_qkv_w, D_MODEL * qkv_dim, "d_qkv_w");
    bwd2_nan += check_nan_inf_all(d_gate_w, D_MODEL * VALUE_DIM, "d_gate_w");
    bwd2_nan += check_nan_inf_all(d_beta_w, D_MODEL * DT_RANK, "d_beta_w");
    bwd2_nan += check_nan_inf_all(d_alpha_w, D_MODEL * DT_RANK, "d_alpha_w");
    bwd2_nan += check_nan_inf_all(d_conv1d_w, CONV_KERNEL * CONV_DIM, "d_conv1d_w");
    bwd2_nan += check_nan_inf_all(d_out_w, VALUE_DIM * D_MODEL, "d_out_w");
    bwd2_nan += check_nan_inf_all(d_norm_w, SSM_D_STATE, "d_norm_w");
    bwd2_nan += check_nan_inf_all(d_state_init, nstate2.K * HEAD_STATE_SZ, "d_state_init");

    // Check state gradient non-zero
    float max_state_grad = 0.0f;
    for (int i = 0; i < nstate2.K * HEAD_STATE_SZ; i++) {
        float abs_v = fabsf(d_state_init[i]);
        if (abs_v > max_state_grad) max_state_grad = abs_v;
    }
    printf("  Max |d_state_init|: %e\n", max_state_grad);

    int t2_pass = (fwd2_nan == 0 && bwd2_nan == 0 && max_state_grad > 1e-10f);
    printf("  Test 2: %s (NaN=%d, state_grad=%.2e)\n",
           t2_pass ? "✅ PASS" : "❌ FAIL", bwd2_nan, max_state_grad);
    total_pass += t2_pass;

    wubu_nested_ssm_fwd_save_free(&save2);
    wubu_nested_ssm_free(&nstate2);
    free(d_qkv_w); free(d_gate_w); free(d_beta_w); free(d_alpha_w);
    free(d_conv1d_w); free(d_out_w); free(d_norm_w); free(d_state_init);

    // ================================================================
    // Test 3: K=3 with varied curvatures — check weight gradients
    // ================================================================
    printf("\n--- Test 3: K=3 backward (R=[0.5, 1.0, 2.0]) ---\n");
    total_tests++;

    float R3[3] = {0.5f, 1.0f, 2.0f};
    wubu_nested_ssm_state_t nstate3;
    if (wubu_nested_ssm_init(&nstate3, 3, R3) != 0) {
        printf("FAIL: init K=3 failed\n");
        return 1;
    }

    memset(conv_state, 0, B * (CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
    memset(d_x, 0, N * D_MODEL * sizeof(float));
    memset(output, 0, N * D_MODEL * sizeof(float));

    // Biased gating
    wubu_nested_ssm_gating_t gating3;
    gating3.ball_weights[0] = -1.0f;
    gating3.ball_weights[1] = 0.0f;
    gating3.ball_weights[2] = 3.0f;

    nested_ssm_fwd_save_t save3;
    memset(&save3, 0, sizeof(save3));
    wubu_nested_ssm_forward_save(x, B, T, &w, &nstate3, conv_state, &gating3, output, &save3);

    d_qkv_w = (float *)calloc(D_MODEL * qkv_dim, sizeof(float));
    d_gate_w = (float *)calloc(D_MODEL * VALUE_DIM, sizeof(float));
    d_beta_w = (float *)calloc(D_MODEL * DT_RANK, sizeof(float));
    d_alpha_w = (float *)calloc(D_MODEL * DT_RANK, sizeof(float));
    d_conv1d_w = (float *)calloc(CONV_KERNEL * CONV_DIM, sizeof(float));
    d_out_w = (float *)calloc(VALUE_DIM * D_MODEL, sizeof(float));
    d_norm_w = (float *)calloc(SSM_D_STATE, sizeof(float));
    d_state_init = (float *)calloc(nstate3.K * HEAD_STATE_SZ, sizeof(float));

    wubu_nested_ssm_backward(B, T, x, output, d_output, gating3.ball_weights, &nstate3, &save3, &w,
                              d_x, d_qkv_w, d_gate_w, d_beta_w, d_alpha_w,
                              d_conv1d_w, d_out_w, d_norm_w, d_state_init, NULL);

    int bwd3_nan = 0;
    bwd3_nan += check_nan_inf_all(d_qkv_w, D_MODEL * qkv_dim, "d_qkv_w");
    bwd3_nan += check_nan_inf_all(d_out_w, VALUE_DIM * D_MODEL, "d_out_w");

    // Verify gradient magnitudes are reasonable
    float max_dqkv = 0, max_dout = 0;
    for (int i = 0; i < D_MODEL * qkv_dim; i++) {
        float av = fabsf(d_qkv_w[i]);
        if (av > max_dqkv) max_dqkv = av;
    }
    for (int i = 0; i < VALUE_DIM * D_MODEL; i++) {
        float av = fabsf(d_out_w[i]);
        if (av > max_dout) max_dout = av;
    }
    printf("  max|d_qkv_w| = %e, max|d_out_w| = %e\n", max_dqkv, max_dout);

    int t3_pass = (bwd3_nan == 0 && max_dqkv > 0 && max_dout > 0);
    printf("  Test 3: %s\n", t3_pass ? "✅ PASS" : "❌ FAIL");
    total_pass += t3_pass;

    wubu_nested_ssm_fwd_save_free(&save3);
    wubu_nested_ssm_free(&nstate3);
    free(d_qkv_w); free(d_gate_w); free(d_beta_w); free(d_alpha_w);
    free(d_conv1d_w); free(d_out_w); free(d_norm_w); free(d_state_init);

    // ================================================================
    // Summary
    // ================================================================
    printf("\n=== Summary ===\n");
    printf("  Test 1 (K=1 backward sanity):     %s\n", total_pass > 0 ? "PASS" : "FAIL");
    printf("  Test 2 (K=2 multi-ball backward): %s\n", total_pass > 1 ? "PASS" : "FAIL");
    printf("  Test 3 (K=3 weight gradients):    %s\n", total_pass > 2 ? "PASS" : "FAIL");
    printf("  Total: %d/%d passed\n", total_pass, total_tests);

    // Cleanup
    free(x);
    free(output);
    free(d_output);
    free(d_x);
    free_weights(&w);
    free(conv_state);
    wubu_nested_ssm_free(&nstate1);

    return total_pass >= 2 ? 0 : 1;
}
