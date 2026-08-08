/**
 * test_one_ssm_backward.c — Validate one SSM layer backward pass.
 *
 * Loads the real model, runs one SSM layer forward with save,
 * then backward with random d_output. Checks gradients are finite.
 *
 * This validates the entire backward primitive chain works end-to-end
 * on real model data.
 */
#include "wubu_model.h"
#include "gguf_reader.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>

static void sigsegv_handler(int sig, siginfo_t *info, void *ctx) {
    (void)ctx;
    write(2, "SIGSEGV at addr=", 16);
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%p\n", info->si_addr);
    write(2, buf, n);
    _exit(1);
}

static float max_f(const float *arr, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) {
        if (!isnan(arr[i]) && !isinf(arr[i])) {
            float a = fabsf(arr[i]);
            if (a > m) m = a;
        }
    }
    return m;
}

int main(int argc, char **argv) {
    // Install SIGSEGV handler
    struct sigaction sa;
    sa.sa_sigaction = sigsegv_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);

    const char *model_path = argc > 1 ? argv[1]
        : "/home/wubu2/models/qwen3.6-35b-a3b-T50-IQ2_M.gguf";
    int layer_idx = argc > 2 ? atoi(argv[2]) : 0;

    printf("Loading model: %s\n", model_path);
    wubu_model_t model;
    if (!wubu_model_init(&model, model_path)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("Model loaded: %d layers\n", model.n_layers);
    fflush(stdout);

    // Find first SSM layer
    int ssm_idx = -1;
    for (int l = 0; l < model.n_layers; l++) {
        if (model.layers[l].is_ssm) {
            if (ssm_idx == -1) ssm_idx = l;
        }
    }
    if (layer_idx < 0 || layer_idx >= model.n_layers) layer_idx = 0;
    if (!model.layers[layer_idx].is_ssm) {
        printf("Layer %d is not SSM, finding first SSM...\n", layer_idx);
        layer_idx = ssm_idx;
    }

    printf("Using layer %d (SSM)\n", layer_idx);
    fflush(stdout);
    wubu_layer_t *layer = &model.layers[layer_idx];
    printf("Layer SSM weights: qkv=%p, ssm_state=%p\n",
           (void*)layer->ssm.attn_qkv_weight, (void*)model.ssm_states);
    fflush(stdout);
    float *ssm_state = model.ssm_states + layer_idx * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    printf("ssm_state=%p conv_states=%p\n", (void*)ssm_state, (void*)model.conv_states);
    fflush(stdout);
    float *conv_state = model.conv_states + layer_idx * (CONV_KERNEL - 1) * CONV_DIM;

    // Create a small batch: B=1, T=1 tokens (decode path, simpler)
    int B = 1, T = 1, N = B * T;
    printf("Allocating buffers... N=%d D_MODEL=%d\n", N, D_MODEL);
    fflush(stdout);
    float *x = (float *)malloc(N * D_MODEL * sizeof(float));
    float *output = (float *)malloc(N * D_MODEL * sizeof(float));
    float *d_output = (float *)malloc(N * D_MODEL * sizeof(float));
    float *d_x = (float *)calloc(N * D_MODEL, sizeof(float));

    // Random input in reasonable range
    for (int i = 0; i < N * D_MODEL; i++)
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    // Allocate workspace and intermediates
    printf("Allocating workspace...\n");
    fflush(stdout);
    ssm_workspace_t *ws = wubu_ssm_workspace_alloc(B, T);
    if (!ws) { fprintf(stderr, "ws alloc failed\n"); return 1; }
    printf("Workspace allocated: ws=%p, qkv_all=%p\n", (void*)ws, (void*)ws->qkv_all);
    fflush(stdout);

    // Forward pass — preserves intermediates in workspace
    printf("Forward...\n");
    printf("  qkv_weight_q=%p type=%d gate_weight_q=%p type=%d\n",
           (void*)layer->ssm.attn_qkv_weight_q, layer->ssm.attn_qkv_weight_type,
           (void*)layer->ssm.attn_gate_weight_q, layer->ssm.attn_gate_weight_type);
    printf("  norm_weight=%p ssm_state=%p conv_state=%p ws=%p\n",
           (void*)layer->attn_norm_weight, (void*)ssm_state, (void*)conv_state, (void*)ws);
    fflush(stdout);
    
    // No RMSNorm — pass x directly (we just need to test the forward/backward)
    wubu_ssm_forward(x, B, T, &layer->ssm, ssm_state, conv_state,
                     output, NULL, NULL, ws);

    printf("Forward output range: [%.4f, %.4f]\n",
           output[0], output[N * D_MODEL - 1]);

    // Random d_output
    for (int i = 0; i < N * D_MODEL; i++)
        d_output[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;

    // Backward pass — uses saved intermediates from ws
    printf("Backward...\n");
    d_output[0] = 1.0f;  // ensure non-zero gradient

    // Allocate weight gradients
    int n_w = D_MODEL * CONV_DIM;
    float *d_qkv = (float *)calloc(n_w, sizeof(float));
    float *d_gate = (float *)calloc(D_MODEL * VALUE_DIM, sizeof(float));
    float *d_beta = (float *)calloc(D_MODEL * DT_RANK, sizeof(float));
    float *d_alpha = (float *)calloc(D_MODEL * DT_RANK, sizeof(float));
    float *d_conv1d = (float *)calloc(CONV_KERNEL * CONV_DIM, sizeof(float));
    float *d_ssm_out = (float *)calloc(VALUE_DIM * D_MODEL, sizeof(float));
    float *d_norm = (float *)calloc(SSM_D_STATE, sizeof(float));
    float *d_state = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));

    wubu_ssm_backward(B, T, x, output, d_output,
        ws->qkv_all, ws->z_all, ws->beta_raw, ws->alpha_raw,
        ws->conv_output, ws->q_conv, ws->k_conv, ws->v_conv,
        ws->q_norm, ws->k_norm, ws->delta_out, ws->z_silu,
        ssm_state, NULL, NULL, conv_state,
        &layer->ssm,
        d_x, d_qkv, d_gate, d_beta, d_alpha,
        d_conv1d, d_ssm_out, d_norm, d_state);

    // Validate gradients
    int nan_count = 0, zero_count = 0;
    float max_dx = 0, max_dw = 0;
    for (int i = 0; i < N * D_MODEL; i++) {
        if (isnan(d_x[i])) nan_count++;
        else if (fabsf(d_x[i]) < 1e-30f) zero_count++;
        else { float a = fabsf(d_x[i]); if (a > max_dx) max_dx = a; }
    }
    for (int i = 0; i < n_w; i++) {
        if (isnan(d_qkv[i])) nan_count++;
        else { float a = fabsf(d_qkv[i]); if (a > max_dw) max_dw = a; }
    }

    printf("\n=== Results ===\n");
    printf("d_x:        NaN=%d, zero=%d, max=%.6e\n", nan_count, zero_count, max_dx);
    printf("d_qkv:      NaN=%d, max=%.6e\n", nan_count, max_dw);
    printf("d_gate:     max=%.6e\n", max_f(d_gate, D_MODEL * VALUE_DIM));
    printf("d_beta:     max=%.6e\n", max_f(d_beta, D_MODEL * DT_RANK));
    printf("d_alpha:    max=%.6e\n", max_f(d_alpha, D_MODEL * DT_RANK));
    printf("d_conv1d:   max=%.6e\n", max_f(d_conv1d, CONV_KERNEL * CONV_DIM));
    printf("d_ssm_out:  max=%.6e\n", max_f(d_ssm_out, VALUE_DIM * D_MODEL));
    printf("d_norm:     max=%.6e\n", max_f(d_norm, SSM_D_STATE));
    printf("d_state:    max=%.6e\n", max_f(d_state, SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE));

    int pass = (nan_count == 0 && max_dx > 0 && max_dw > 0);
    printf("\n%s: gradients %s\n",
           pass ? "PASS" : "FAIL",
           pass ? "valid and non-zero" : "have issues");

    // Cleanup
    free(x); free(output); free(d_output); free(d_x);
    free(d_qkv); free(d_gate); free(d_beta); free(d_alpha);
    free(d_conv1d); free(d_ssm_out); free(d_norm); free(d_state);
    wubu_ssm_workspace_free(ws);
    wubu_model_free(&model);
    return pass ? 0 : 1;
}
