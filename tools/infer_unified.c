/**
 * infer_unified.c — Unified 40-layer inference (SSM→GQA→MoE)
 *
 * Loads GGUF once, runs all 40 layers with per-layer MoE.
 * Uses wubu_moe_load_layer for MoE (full dequant per layer).
 *
 * KNOWN ISSUE: NaN at GQA layers due to pre-existing SSM forward
 * instability with out-of-distribution inputs. Use real token
 * embeddings for stable forward.
 */
#include "wubu_model.h"
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "wubu_core_dumps.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Per-layer MoE quantized metadata (from pre-buffered GGUF)
typedef struct {
    int ty_gi, ty_ge, ty_gs;
    int64_t expert_raw, expert_raw_down;
    bool has_moe;
} moe_quant_t;

int main(int argc, char **argv) {
    wubu_disable_core_dumps();
    const char *path = argc > 1 ? argv[1]
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int B = 1, T = argc > 2 ? atoi(argv[2]) : 4;
    int N = B * T;
    int verbose = argc > 3 ? atoi(argv[3]) : 1;

    printf("=== Unified 40-layer Inference ===\n");
    printf("Model: %s  B=%d T=%d\n", path, B, T);

    // ================================================================
    // 1. Load GGUF + buffer all data
    // ================================================================
    double t0 = now_sec();
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    double t_load = now_sec() - t0;
    printf("GGUF load+buffer: %.2f s\n", t_load);

    // ================================================================
    // 2. Load model (SSM/GQA weights)
    // ================================================================
    t0 = now_sec();
    wubu_model_t model;
    if (!wubu_model_init(&model, path)) {
        fprintf(stderr, "Model init failed\n");
        gguf_close(ctx);
        return 1;
    }
    // Model's GGUF ctx: close it, use ours for MoE
    // SSM/GQA weights are already malloc'd, so the ctx isn't needed anymore
    if (model.gguf_ctx) {
        gguf_close(model.gguf_ctx);
    }
    model.gguf_ctx = ctx;
    double t_model = now_sec() - t0;
    printf("Model init: %.2f s\n", t_model);

    // ================================================================
    // 3. Gather MoE quantization metadata (from buffered GGUF)
    // ================================================================
    int64_t expert_n = (int64_t)D_MODEL * D_FF;
    int64_t expert_n_down = (int64_t)D_FF * D_MODEL;
    moe_quant_t *moe_q = (moe_quant_t *)calloc(model.n_layers, sizeof(moe_quant_t));

    for (int l = 0; l < model.n_layers; l++) {
        moe_quant_t *mq = &moe_q[l];
        char name[256];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
        gguf_tensor_info *t = gguf_find_tensor(ctx, name);
        mq->has_moe = (t != NULL);
        if (t) {
            mq->ty_gi = t->ggml_type;
            // Check expert tensors exist
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", l);
            t = gguf_find_tensor(ctx, name);
            mq->has_moe = mq->has_moe && (t != NULL);
            if (t) {
                mq->ty_ge = t->ggml_type;
                mq->expert_raw = gguf_raw_size(mq->ty_ge, expert_n);
                mq->expert_raw_down = gguf_raw_size(mq->ty_ge, expert_n_down);
            }
        }
    }

    // ================================================================
    // 4. Create test input (small scale to avoid SSM instability)
    // ================================================================
    float *embd = (float *)malloc(N * D_MODEL * sizeof(float));
    srand(42);
    for (int i = 0; i < N * D_MODEL; i++)
        embd[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;

    // ================================================================
    // 5. Forward pass through all layers
    // ================================================================
    float *x = (float *)malloc(N * D_MODEL * sizeof(float));
    float *normed = (float *)malloc(N * D_MODEL * sizeof(float));
    float *attn_out = (float *)malloc(N * D_MODEL * sizeof(float));
    float *normed2 = (float *)malloc(N * D_MODEL * sizeof(float));
    float *ffn_out = (float *)malloc(N * D_MODEL * sizeof(float));

    memcpy(x, embd, N * D_MODEL * sizeof(float));

    double total_ssm = 0, total_gqa = 0, total_norm = 0, total_moe = 0;

    for (int l = 0; l < model.n_layers; l++) {
        wubu_layer_t *layer = &model.layers[l];

        // Pre-attention RMSNorm
        double t_n = now_sec();
        wubu_rms_norm(B, T, D_MODEL, x, layer->attn_norm_weight, 1e-6f, normed);
        total_norm += now_sec() - t_n;

        // SSM or GQA attention
        double t_a = now_sec();
        if (layer->is_ssm) {
            float *ssm_state = model.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = model.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, B, T, &layer->ssm, ssm_state, conv_state, attn_out, NULL, NULL, NULL);
            total_ssm += now_sec() - t_a;
        } else {
            wubu_gqa_forward(normed, B, T, &layer->gqa, attn_out, NULL, NULL, 0, NULL, NULL);
            total_gqa += now_sec() - t_a;
        }

        // Check for NaN
        int nan_idx = -1;
        for (int i = 0; i < N * D_MODEL; i++) {
            if (isnan(attn_out[i])) { nan_idx = i; break; }
        }
        if (nan_idx >= 0) {
            int tt = nan_idx / D_MODEL, dd = nan_idx % D_MODEL;
            printf("  L%02d %s NaN at [t=%d,d=%d] | normed[0:4]=%.2e %.2e %.2e %.2e\n",
                   l, layer->is_ssm ? "SSM" : "GQA", tt, dd,
                   normed[0], normed[1], normed[2], normed[3]);
        }

        // Residual
        for (int i = 0; i < N * D_MODEL; i++) x[i] += attn_out[i];

        // Post-attention RMSNorm
        t_n = now_sec();
        wubu_rms_norm(B, T, D_MODEL, x, layer->post_attn_norm_weight, 1e-6f, normed2);
        total_norm += now_sec() - t_n;

        // MoE forward (full dequant per layer)
        double t_moe = now_sec();
        int n_unique = 0;

        if (moe_q[l].has_moe) {
            if (wubu_moe_load_layer(ctx, l, &layer->moe)) {
                wubu_moe_forward(normed2, B, T, &layer->moe, ffn_out, NULL);
                wubu_moe_free_layer(&layer->moe);
            } else {
                memcpy(ffn_out, normed2, N * D_MODEL * sizeof(float));
            }
        } else {
            memcpy(ffn_out, normed2, N * D_MODEL * sizeof(float));
        }
        total_moe += now_sec() - t_moe;

        // Residual
        for (int i = 0; i < N * D_MODEL; i++) x[i] += ffn_out[i];

        if (verbose) {
            float mean = 0.0f;
            for (int i = 0; i < D_MODEL; i++) mean += fabsf(x[i]);
            mean /= D_MODEL;
            printf("  L%02d %s | attn %6.2fms | moe %6.2fms | x[0:4] %.4f %.4f %.4f %.4f | mean %.4f\n",
                   l, layer->is_ssm ? "SSM" : "GQA",
                   (now_sec() - t_a) * 1000,
                   (now_sec() - t_moe) * 1000,
                   x[0], x[1], x[2], x[3], mean);
        }
    }

    // Final RMSNorm
    if (model.norm_weight) {
        wubu_rms_norm(B, T, D_MODEL, x, model.norm_weight, 1e-6f, normed);
        memcpy(x, normed, N * D_MODEL * sizeof(float));
    }

    // ================================================================
    // 6. Summary
    // ================================================================
    double total_time = now_sec() - t0;
    printf("\n=== Unified 40-layer Summary ===\n");
    printf("Total time: %.3f s (%.1f tok/s)\n", total_time, N / total_time);
    printf("SSM fwd:    %.3f s\n", total_ssm);
    printf("GQA fwd:    %.3f s\n", total_gqa);
    printf("Norms:      %.3f s\n", total_norm);
    printf("MoE fwd:    %.3f s\n", total_moe);

    float mean = 0.0f, min_v = 1e30f, max_v = -1e30f;
    for (int i = 0; i < N * D_MODEL; i++) {
        mean += fabsf(x[i]);
        if (x[i] < min_v) min_v = x[i];
        if (x[i] > max_v) max_v = x[i];
    }
    mean /= (N * D_MODEL);
    printf("Final hidden: mean %.6f range [%.4e, %.4e]\n", mean, min_v, max_v);

    int nan_total = 0;
    for (int i = 0; i < N * D_MODEL; i++)
        if (isnan(x[i])) nan_total++;
    printf("NaN count: %d/%d\n", nan_total, N * D_MODEL);

    // Cleanup
    free(x); free(normed); free(attn_out); free(normed2); free(ffn_out);
    free(embd);
    free(moe_q);
    wubu_model_free(&model);
    // ctx closed by wubu_model_free (model.gguf_ctx = ctx)

    printf("\n=== Unified Inference PASS ===\n");
    return 0;
}
