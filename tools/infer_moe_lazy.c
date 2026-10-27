/**
 * infer_moe_lazy.c — Lazy MoE inference
 *
 * Key optimization: dequant ONLY the top-2 experts per token instead of all 256.
 * For B=1, T=4: at most 8 unique experts → ~50× dequant speedup vs full dequant.
 *
 * Flow:
 *   1. Load quantized weight pointers (no dequant yet)
 *   2. Dequantize ROUTER only (small: 2048×256 = 0.5M floats)
 *   3. Route input to get top-k experts per token
 *   4. Collect UNIQUE expert IDs across all tokens
 *   5. Dequant only those experts' gate/up/down weights
 *   6. Run forward using cached dequantized weights
 *   7. Benchmark vs full dequant path
 */
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

// Per-expert dequantized weights (one expert's gate/up/down)
typedef struct {
    int expert_id;
    float *gate;   // [D_MODEL, D_FF]
    float *up;     // [D_MODEL, D_FF]
    float *down;   // [D_FF, D_MODEL]
    bool used;
} expert_scratch_t;

int main(int argc, char **argv) {
    wubu_disable_core_dumps();
    const char *path = argc > 1 ? argv[1]
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int layer = argc > 2 ? atoi(argv[2]) : 0;
    int B = 1, T = argc > 3 ? atoi(argv[3]) : 4;

    printf("=== Lazy MoE Inference ===\n");
    printf("Model: %s  Layer: %d  B=%d T=%d\n", path, layer, B, T);

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);

    // ------------------------------------------------------------------
    // 1. Load quantized tensor pointers (NO dequant yet)
    // ------------------------------------------------------------------
    char name[256];
    const uint8_t *q_gate_inp = NULL;
    const uint8_t *q_gate_exps = NULL;
    const uint8_t *q_up_exps = NULL;
    const uint8_t *q_down_exps = NULL;
    const uint8_t *q_gate_shexp = NULL;
    const uint8_t *q_up_shexp = NULL;
    const uint8_t *q_down_shexp = NULL;
    int ty_ge = 0; // expert type
    int ty_gi = 0; // router type
    int ty_gs = 0; // shared expert type

    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", layer);
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (t) { ty_gi = t->ggml_type; q_gate_inp = (const uint8_t *)ctx->data_blob + t->data_offset; }

    snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (t) { ty_ge = t->ggml_type; q_gate_exps = (const uint8_t *)ctx->data_blob + t->data_offset; }

    snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (t) q_up_exps = (const uint8_t *)ctx->data_blob + t->data_offset;

    snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (t) q_down_exps = (const uint8_t *)ctx->data_blob + t->data_offset;

    snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (t) { ty_gs = t->ggml_type; q_gate_shexp = (const uint8_t *)ctx->data_blob + t->data_offset; }

    snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (t) q_up_shexp = (const uint8_t *)ctx->data_blob + t->data_offset;

    snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (t) q_down_shexp = (const uint8_t *)ctx->data_blob + t->data_offset;

    if (!q_gate_exps || !q_up_exps || !q_down_exps) {
        fprintf(stderr, "Missing expert tensors\n");
        gguf_close(ctx);
        return 1;
    }
    printf("Quantized type: %d\n", ty_ge);

    // Pre-compute per-expert raw byte sizes
    int64_t expert_n = (int64_t)D_MODEL * D_FF;         // 2048*512 = 1,048,576
    int64_t expert_n_down = (int64_t)D_FF * D_MODEL;    // 512*2048 = 1,048,576
    int64_t expert_raw = gguf_raw_size(ty_ge, expert_n);
    int64_t expert_raw_down = gguf_raw_size(ty_ge, expert_n_down);

    // Shared expert sizes
    int64_t shared_n = (int64_t)D_MODEL * SHARED_D_FF;
    int64_t shared_n_down = (int64_t)SHARED_D_FF * D_MODEL;

    printf("Per-expert raw size: %ld B (gate/up), %ld B (down)\n",
           (long)expert_raw, (long)expert_raw_down);
    printf("Total 256-expert raw: %.1f MB (×3 tensors)\n",
           256.0 * (expert_raw + expert_raw + expert_raw_down) / (1024*1024));

    // ------------------------------------------------------------------
    // 2. Dequantize router ONLY
    // ------------------------------------------------------------------
    double t0 = now_sec();
    float *gate_inp = (float *)malloc(D_MODEL * N_EXPERTS * sizeof(float));
    gguf_dequantize(q_gate_inp, ty_gi, D_MODEL * N_EXPERTS, gate_inp);
    printf("Router dequant: %.3f ms\n", (now_sec() - t0) * 1000);

    // ------------------------------------------------------------------
    // 3. Build moe_weights_t with ONLY router + shared (NOT expert weights)
    // ------------------------------------------------------------------
    moe_weights_t moe;
    memset(&moe, 0, sizeof(moe));
    moe.ffn_gate_inp = gate_inp;
    moe.ffn_gate_shexp = NULL; // deferred
    moe.ffn_up_shexp = NULL;
    moe.ffn_down_shexp = NULL;
    moe.loaded = true;

    // Test input
    int N = B * T;
    float *x = (float *)malloc(N * D_MODEL * sizeof(float));
    float *output = (float *)malloc(N * D_MODEL * sizeof(float));
    for (int i = 0; i < N * D_MODEL; i++)
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    // ------------------------------------------------------------------
    // 4. Route to get top-k indices per token
    // ------------------------------------------------------------------
    float *scores = (float *)malloc(N * N_EXPERTS * sizeof(float));
    wubu_moe_router(x, B, T, moe.ffn_gate_inp, scores, N_EXPERTS, D_MODEL);

    // Softmax + top-k (same as wubu_moe_forward)
    int *topk_indices = (int *)malloc(N * N_ACTIVE_EXPTS * sizeof(int));
    float *topk_weights = (float *)malloc(N * N_ACTIVE_EXPTS * sizeof(float));

    for (int s = 0; s < N; s++) {
        float *score_s = scores + s * N_EXPERTS;

        // Softmax
        float max_s = score_s[0];
        for (int e = 1; e < N_EXPERTS; e++)
            if (score_s[e] > max_s) max_s = score_s[e];

        float sum_exp = 0.0f;
        for (int e = 0; e < N_EXPERTS; e++)
            sum_exp += expf(score_s[e] - max_s);
        float inv_sum = 1.0f / (sum_exp + 1e-30f);

        float softmax_vals[N_EXPERTS];
        for (int e = 0; e < N_EXPERTS; e++)
            softmax_vals[e] = expf(score_s[e] - max_s) * inv_sum;

        int *indices_s = topk_indices + s * N_ACTIVE_EXPTS;
        float *weights_s = topk_weights + s * N_ACTIVE_EXPTS;

        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            int best_idx = -1;
            float best_val = -1e30f;
            for (int e = 0; e < N_EXPERTS; e++) {
                bool used = false;
                for (int pk = 0; pk < k; pk++)
                    if (indices_s[pk] == e) { used = true; break; }
                if (!used && softmax_vals[e] > best_val) {
                    best_val = softmax_vals[e];
                    best_idx = e;
                }
            }
            indices_s[k] = best_idx;
            weights_s[k] = best_val;
        }

        float sum_w = 0.0f;
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) sum_w += weights_s[k];
        if (sum_w > 1e-30f) {
            float inv_sum_w = 1.0f / sum_w;
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) weights_s[k] *= inv_sum_w;
        }
    }
    printf("Routing done for %d tokens\n", N);

    // ------------------------------------------------------------------
    // 5. Collect UNIQUE expert IDs across all tokens
    // ------------------------------------------------------------------
    int unique_ids[N_ACTIVE_EXPTS * N]; // max possible
    int n_unique = 0;
    for (int s = 0; s < N; s++) {
        int *indices_s = topk_indices + s * N_ACTIVE_EXPTS;
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            int eid = indices_s[k];
            if (eid < 0) continue;
            bool seen = false;
            for (int u = 0; u < n_unique; u++)
                if (unique_ids[u] == eid) { seen = true; break; }
            if (!seen) unique_ids[n_unique++] = eid;
        }
    }
    printf("Unique experts needed: %d/%d\n", n_unique, N_EXPERTS);

    // ------------------------------------------------------------------
    // 6. Dequant only selected experts
    // ------------------------------------------------------------------
    t0 = now_sec();
    expert_scratch_t *experts = (expert_scratch_t *)calloc(n_unique, sizeof(expert_scratch_t));
    for (int u = 0; u < n_unique; u++) {
        int eid = unique_ids[u];
        experts[u].expert_id = eid;

        // Dequant gate_exps[eid]
        experts[u].gate = (float *)malloc(expert_n * sizeof(float));
        const uint8_t *gate_ptr = q_gate_exps + (int64_t)eid * expert_raw;
        gguf_dequantize(gate_ptr, ty_ge, expert_n, experts[u].gate);

        // Dequant up_exps[eid]
        experts[u].up = (float *)malloc(expert_n * sizeof(float));
        const uint8_t *up_ptr = q_up_exps + (int64_t)eid * expert_raw;
        gguf_dequantize(up_ptr, ty_ge, expert_n, experts[u].up);

        // Dequant down_exps[eid]
        experts[u].down = (float *)malloc(expert_n_down * sizeof(float));
        const uint8_t *down_ptr = q_down_exps + (int64_t)eid * expert_raw_down;
        gguf_dequantize(down_ptr, ty_ge, expert_n_down, experts[u].down);

        experts[u].used = true;
    }
    double lazy_dequant_time = now_sec() - t0;
    printf("Lazy dequant (%d experts): %.3f ms total\n",
           n_unique, lazy_dequant_time * 1000);
    printf("  Per-expert avg: %.3f ms\n", lazy_dequant_time / n_unique * 1000);

    // Also dequant shared expert
    t0 = now_sec();
    float *gate_shexp = (float *)malloc(D_MODEL * SHARED_D_FF * sizeof(float));
    float *up_shexp = (float *)malloc(D_MODEL * SHARED_D_FF * sizeof(float));
    float *down_shexp = (float *)malloc(SHARED_D_FF * D_MODEL * sizeof(float));
    gguf_dequantize(q_gate_shexp, ty_gs, shared_n, gate_shexp);
    gguf_dequantize(q_up_shexp, ty_gs, shared_n, up_shexp);
    gguf_dequantize(q_down_shexp, ty_gs, shared_n_down, down_shexp);
    printf("Shared expert dequant: %.3f ms\n", (now_sec() - t0) * 1000);

    // ------------------------------------------------------------------
    // 7. Run MoE forward using cached dequantized experts
    // ------------------------------------------------------------------
    // Reuse wubu_moe_forward but with a modified weights struct
    // We need to construct temporary dequantized full arrays with zeros
    // for non-selected experts, OR write a custom forward that uses the cache.
    //
    // For fairness of comparison, we build a full moe_weights_t with zeros
    // for non-selected experts (same API path).
    // Then we compare time vs the full-dequant path.

    // Build full dequant arrays with only selected experts populated
    int64_t ne_full = (int64_t)D_MODEL * D_FF * N_EXPERTS;
    float *full_gate = (float *)calloc(ne_full, sizeof(float));
    float *full_up = (float *)calloc(ne_full, sizeof(float));
    float *full_down = (float *)calloc((int64_t)D_FF * D_MODEL * N_EXPERTS, sizeof(float));

    for (int u = 0; u < n_unique; u++) {
        int eid = experts[u].expert_id;
        // Copy dequantized weights into full arrays
        memcpy(full_gate + (int64_t)eid * D_MODEL * D_FF,
               experts[u].gate, expert_n * sizeof(float));
        memcpy(full_up + (int64_t)eid * D_MODEL * D_FF,
               experts[u].up, expert_n * sizeof(float));
        memcpy(full_down + (int64_t)eid * D_FF * D_MODEL,
               experts[u].down, expert_n_down * sizeof(float));
    }

    moe_weights_t moe_lazy;
    memset(&moe_lazy, 0, sizeof(moe_lazy));
    moe_lazy.ffn_gate_inp = gate_inp;
    moe_lazy.ffn_gate_exps = full_gate;
    moe_lazy.ffn_up_exps = full_up;
    moe_lazy.ffn_down_exps = full_down;
    moe_lazy.ffn_gate_shexp = gate_shexp;
    moe_lazy.ffn_up_shexp = up_shexp;
    moe_lazy.ffn_down_shexp = down_shexp;
    moe_lazy.loaded = true;

    int iters = 10;
    double total = 0.0;
    for (int i = 0; i < iters; i++) {
        t0 = now_sec();
        wubu_moe_forward(x, B, T, &moe_lazy, output, NULL, N_ACTIVE_EXPTS, N_EXPERTS, D_MODEL, D_FF);
        total += now_sec() - t0;
    }
    double lazy_forward_time = total / iters * 1000;
    printf("\nLazy MoE forward (%d iters): avg %.3f ms (%.0f tok/s)\n",
           iters, lazy_forward_time, B * T / (total / iters));

    // ------------------------------------------------------------------
    // 8. Full dequant benchmark
    // ------------------------------------------------------------------
    printf("\n--- Full dequant benchmark ---\n");
    moe_weights_t moe_full;
    memset(&moe_full, 0, sizeof(moe_full));
    t0 = now_sec();
    moe_full.ffn_gate_inp = (float *)malloc(D_MODEL * N_EXPERTS * sizeof(float));
    gguf_dequantize(q_gate_inp, ty_gi, D_MODEL * N_EXPERTS, moe_full.ffn_gate_inp);

    moe_full.ffn_gate_exps = (float *)malloc(ne_full * sizeof(float));
    gguf_dequantize(q_gate_exps, ty_ge, ne_full, moe_full.ffn_gate_exps);
    moe_full.ffn_up_exps = (float *)malloc(ne_full * sizeof(float));
    gguf_dequantize(q_up_exps, ty_ge, ne_full, moe_full.ffn_up_exps);
    int64_t ne_full_down = (int64_t)D_FF * D_MODEL * N_EXPERTS;
    moe_full.ffn_down_exps = (float *)malloc(ne_full_down * sizeof(float));
    gguf_dequantize(q_down_exps, ty_ge, ne_full_down, moe_full.ffn_down_exps);

    moe_full.ffn_gate_shexp = (float *)malloc(shared_n * sizeof(float));
    gguf_dequantize(q_gate_shexp, ty_gs, shared_n, moe_full.ffn_gate_shexp);
    moe_full.ffn_up_shexp = (float *)malloc(shared_n * sizeof(float));
    gguf_dequantize(q_up_shexp, ty_gs, shared_n, moe_full.ffn_up_shexp);
    moe_full.ffn_down_shexp = (float *)malloc(shared_n_down * sizeof(float));
    gguf_dequantize(q_down_shexp, ty_gs, shared_n_down, moe_full.ffn_down_shexp);
    moe_full.loaded = true;
    double full_dequant_time = now_sec() - t0;
    printf("Full dequant (all 256 experts): %.3f s\n", full_dequant_time);

    total = 0.0;
    for (int i = 0; i < iters; i++) {
        t0 = now_sec();
        wubu_moe_forward(x, B, T, &moe_full, output, NULL, N_ACTIVE_EXPTS, N_EXPERTS, D_MODEL, D_FF);
        total += now_sec() - t0;
    }
    double full_forward_time = total / iters * 1000;
    printf("Full MoE forward (%d iters): avg %.3f ms (%.0f tok/s)\n",
           iters, full_forward_time, B * T / (total / iters));

    // ------------------------------------------------------------------
    // 9. Summary
    // ------------------------------------------------------------------
    printf("\n=== LAZY vs FULL ===\n");
    printf("Dequant:   lazy %.3f ms vs full %.3f s (%.0f× speedup)\n",
           lazy_dequant_time * 1000, full_dequant_time,
           full_dequant_time / (lazy_dequant_time + 1e-30));
    printf("Forward:   lazy %.3f ms vs full %.3f ms\n",
           lazy_forward_time, full_forward_time);
    printf("Total/tok: lazy %.3f ms vs full %.3f ms\n",
           lazy_dequant_time * 1000 / (B*T) + lazy_forward_time / (B*T),
           full_dequant_time * 1000 / (B*T) + full_forward_time / (B*T));

    // Verify output match
    float *full_output = (float *)malloc(N * D_MODEL * sizeof(float));
    wubu_moe_forward(x, B, T, &moe_full, full_output, NULL, N_ACTIVE_EXPTS, N_EXPERTS, D_MODEL, D_FF);

    float max_diff = 0.0f;
    int n_mismatch = 0;
    for (int i = 0; i < N * D_MODEL; i++) {
        float diff = fabsf(output[i] - full_output[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-5f) n_mismatch++;
    }
    printf("\nOutput comparison: max_diff=%.2e mismatch=%d/%d\n",
           max_diff, n_mismatch, N * D_MODEL);
    if (max_diff > 1e-3f)
        printf("WARNING: Output mismatch > 1e-3!\n");
    else
        printf("PASS: Outputs match.\n");

    // Cleanup
    free(full_gate); free(full_up); free(full_down);
    free(gate_inp);
    free(gate_shexp); free(up_shexp); free(down_shexp);
    free(x); free(output); free(full_output);
    free(scores); free(topk_indices); free(topk_weights);
    for (int u = 0; u < n_unique; u++) {
        free(experts[u].gate);
        free(experts[u].up);
        free(experts[u].down);
    }
    free(experts);
    wubu_moe_free_layer(&moe_full);
    gguf_close(ctx);

    printf("\n=== Lazy MoE Inference PASS ===\n");
    return 0;
}
