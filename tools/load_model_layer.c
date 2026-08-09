#include "wubu_ssm.h"
#include "wubu_mobius.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_core_dumps.h"

/**
 * Load Qwen3.6 SSM layer weights from GGUF file.
 * Opens the model, finds all tensors for layer `layer_idx`, dequantizes to float32.
 */
int load_ssm_layer(gguf_ctx *ctx, int layer_idx, ssm_layer_weights *w) {
    char name[256];
    int ok = 1;
    
    // attn_qkv.weight [2048, 8192] Q8_K
    snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", layer_idx);
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    int qkv_dim = KEY_DIM * 2 + VALUE_DIM;
    w->attn_qkv_weight = (float *)malloc(D_MODEL * qkv_dim * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->attn_qkv_weight, D_MODEL * qkv_dim) > 0);
    printf("  Loaded %s [%ld, %ld] type=%d -> f32\n", name, (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    
    // attn_gate.weight [2048, 4096] Q8_K
    snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    w->attn_gate_weight = (float *)malloc(D_MODEL * VALUE_DIM * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->attn_gate_weight, D_MODEL * VALUE_DIM) > 0);
    printf("  Loaded %s [%ld, %ld] type=%d\n", name, (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    
    // ssm_beta.weight [2048, 32] F32
    snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    w->ssm_beta_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->ssm_beta_weight, D_MODEL * DT_RANK) > 0);
    printf("  Loaded %s [%ld, %ld] type=%d\n", name, (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    
    // ssm_alpha.weight [2048, 32] F32
    snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    w->ssm_alpha_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->ssm_alpha_weight, D_MODEL * DT_RANK) > 0);
    printf("  Loaded %s [%ld, %ld] type=%d\n", name, (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    
    // ssm_dt.bias [32] F32
    snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    w->ssm_dt_bias = (float *)malloc(DT_RANK * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->ssm_dt_bias, DT_RANK) > 0);
    printf("  Loaded %s [%ld] type=%d\n", name, (long)t->dims[0], t->ggml_type);
    
    // ssm_a [32] F32
    snprintf(name, sizeof(name), "blk.%d.ssm_a", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    w->ssm_a = (float *)malloc(DT_RANK * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->ssm_a, DT_RANK) > 0);
    printf("  Loaded %s [%ld] type=%d\n", name, (long)t->dims[0], t->ggml_type);
    
    // ssm_conv1d.weight [4, 8192] F32
    snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    w->ssm_conv1d_weight = (float *)malloc(CONV_KERNEL * CONV_DIM * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->ssm_conv1d_weight, CONV_KERNEL * CONV_DIM) > 0);
    printf("  Loaded %s [%ld, %ld] type=%d\n", name, (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    
    // ssm_norm.weight [128] F32
    snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    w->ssm_norm_weight = (float *)malloc(SSM_D_STATE * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->ssm_norm_weight, SSM_D_STATE) > 0);
    printf("  Loaded %s [%ld] type=%d\n", name, (long)t->dims[0], t->ggml_type);
    
    // ssm_out.weight [4096, 2048] Q6_K
    snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    w->ssm_out_weight = (float *)malloc(VALUE_DIM * D_MODEL * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->ssm_out_weight, VALUE_DIM * D_MODEL) > 0);
    printf("  Loaded %s [%ld, %ld] type=%d\n", name, (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    
    return ok;
}

/**
 * Load Qwen3.6 GQA layer weights from GGUF file.
 */
int load_gqa_layer(gguf_ctx *ctx, int layer_idx, gqa_layer_weights *w) {
    char name[256];
    int ok = 1;
    int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;    // 4096
    int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;   // 512
    
    const char *SSM_NAMES[] = {"attn_qkv.weight"};  // just for layer type check, not used
    // wq.weight -> attn_q.weight (Qwen3.6 naming)
    snprintf(name, sizeof(name), "blk.%d.attn_q.weight", layer_idx);
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    w->attn_q_weight = (float *)malloc(D_MODEL * q_dim * 2 * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->attn_q_weight, D_MODEL * q_dim * 2) > 0);
    printf("  Loaded %s [%ld, %ld] type=%d\n", name, (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    
    // wk.weight -> attn_k.weight
    snprintf(name, sizeof(name), "blk.%d.attn_k.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    w->attn_k_weight = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->attn_k_weight, D_MODEL * kv_dim) > 0);
    printf("  Loaded %s [%ld, %ld] type=%d\n", name, (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    
    // wv.weight -> attn_v.weight
    snprintf(name, sizeof(name), "blk.%d.attn_v.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    w->attn_v_weight = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->attn_v_weight, D_MODEL * kv_dim) > 0);
    printf("  Loaded %s [%ld, %ld] type=%d\n", name, (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    
    // attn_output.weight [4096, 2048]
    snprintf(name, sizeof(name), "blk.%d.attn_output.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    w->attn_output_weight = (float *)malloc(q_dim * D_MODEL * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->attn_output_weight, q_dim * D_MODEL) > 0);
    printf("  Loaded %s [%ld, %ld] type=%d\n", name, (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    
    // attn_q_norm.weight [256]
    snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    w->attn_q_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->attn_q_norm_weight, GQA_HEAD_DIM) > 0);
    printf("  Loaded %s [%ld] type=%d\n", name, (long)t->dims[0], t->ggml_type);
    
    // attn_k_norm.weight [256]
    snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 0; }
    w->attn_k_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
    ok = ok && (gguf_read_tensor_f32(ctx, t, w->attn_k_norm_weight, GQA_HEAD_DIM) > 0);
    printf("  Loaded %s [%ld] type=%d\n", name, (long)t->dims[0], t->ggml_type);
    
    return ok;
}

void free_ssm_weights(ssm_layer_weights *w) {
    free(w->attn_qkv_weight); free(w->attn_gate_weight);
    free(w->ssm_beta_weight); free(w->ssm_alpha_weight);
    free(w->ssm_dt_bias); free(w->ssm_a);
    free(w->ssm_conv1d_weight); free(w->ssm_norm_weight); free(w->ssm_out_weight);
}

void free_gqa_weights(gqa_layer_weights *w) {
    free(w->attn_q_weight); free(w->attn_k_weight); free(w->attn_v_weight);
    free(w->attn_output_weight); free(w->attn_q_norm_weight); free(w->attn_k_norm_weight);
}

int main(int argc, char **argv) {
    wubu_disable_core_dumps();
    if (argc < 2) {
        printf("Usage: %s <gguf_model_path> [layer_idx]\n", argv[0]);
        printf("  Runs SSM forward on layer `layer_idx` (default: 0) from the model.\n");
        return 1;
    }
    
    const char *model_path = argv[1];
    int layer_idx = argc > 2 ? atoi(argv[2]) : 0;
    
    // Open model
    gguf_ctx *ctx = gguf_open(model_path);
    if (!ctx) {
        fprintf(stderr, "ERROR: Cannot open %s\n", model_path);
        return 1;
    }
    printf("Model: %s (%d tensors)\n", model_path, (int)ctx->n_tensors);
    
    // Check if SSM or GQA
    int is_ssm = wubu_is_ssm_layer(layer_idx);
    printf("Layer %d: %s\n\n", layer_idx, is_ssm ? "SSM" : "GQA");
    
    // Generate dummy input (single token)
    int B = 1, T = 1;
    float *x = (float *)malloc(B * T * D_MODEL * sizeof(float));
    srand(42);
    for (int i = 0; i < B * T * D_MODEL; i++) x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // non-uniform
    
    if (is_ssm) {
        // Load SSM weights
        ssm_layer_weights w;
        if (!load_ssm_layer(ctx, layer_idx, &w)) {
            fprintf(stderr, "Failed to load SSM layer %d\n", layer_idx);
            goto cleanup;
        }
        
        // Allocate state
        float *ssm_state = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
        float *conv_state = (float *)calloc(B * (CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
        float *output = (float *)malloc(B * T * D_MODEL * sizeof(float));
        
        // Run Euclidean SSM
        printf("\nRunning Euclidean SSM forward...\n");
        wubu_ssm_forward(x, B, T, &w, ssm_state, conv_state, output, NULL, NULL, NULL);
        
        float min_v = 1e30f, max_v = -1e30f;
        for (int i = 0; i < B * T * D_MODEL; i++) {
            if (output[i] < min_v) min_v = output[i];
            if (output[i] > max_v) max_v = output[i];
        }
        printf("  Output range: [%e, %e]\n", min_v, max_v);
        printf("  Output[0:8]:");
        for (int i = 0; i < 8 && i < B*T*D_MODEL; i++) printf(" %+.6f", output[i]);
        printf("\n");
        
        // Run Poincare SSM
        float R = 0.956f;
        memset(ssm_state, 0, SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
        memset(conv_state, 0, B * (CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
        printf("\nRunning Poincare SSM forward (R=%.3f)...\n", R);
        wubu_poincare_ssm_forward(x, B, T, &w, ssm_state, conv_state, R, output);
        
        min_v = 1e30f; max_v = -1e30f;
        int nan = 0, inf = 0;
        for (int i = 0; i < B * T * D_MODEL; i++) {
            if (isnan(output[i])) { nan++; continue; }
            if (isinf(output[i])) { inf++; continue; }
            if (output[i] < min_v) min_v = output[i];
            if (output[i] > max_v) max_v = output[i];
        }
        printf("  Output range: [%e, %e]\n", min_v, max_v);
        printf("  Output[0:8]:");
        for (int i = 0; i < 8 && i < B*T*D_MODEL; i++) printf(" %+.6f", output[i]);
        printf("\n");
        printf("  NaN=%d Inf=%d\n", nan, inf);
        
        free(ssm_state); free(conv_state); free(output);
        free_ssm_weights(&w);
    } else {
        gqa_layer_weights w;
        if (!load_gqa_layer(ctx, layer_idx, &w)) {
            fprintf(stderr, "Failed to load GQA layer %d\n", layer_idx);
            goto cleanup;
        }
        
        float *output = (float *)malloc(B * T * D_MODEL * sizeof(float));
        
        printf("\nRunning GQA forward...\n");
        wubu_gqa_forward(x, B, T, &w, output, NULL, NULL, 0, NULL, NULL);
        
        float min_v = 1e30f, max_v = -1e30f;
        for (int i = 0; i < B * T * D_MODEL; i++) {
            if (output[i] < min_v) min_v = output[i];
            if (output[i] > max_v) max_v = output[i];
        }
        printf("  Output range: [%e, %e]\n", min_v, max_v);
        printf("  Output[0:8]:");
        for (int i = 0; i < 8 && i < B*T*D_MODEL; i++) printf(" %+.6f", output[i]);
        printf("\n");
        
        free(output);
        free_gqa_weights(&w);
    }
    
cleanup:
    free(x);
    gguf_close(ctx);
    return 0;
}
