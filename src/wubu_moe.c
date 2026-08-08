#include "wubu_moe.h"
#include "wubu_ssm.h"
#include "mtp_q8_cache.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

// GPU MoE expert forward (declared in wubu_model_gpu.cu, C linkage)
#ifdef GPU_SUPPORT
void wubu_model_gpu_moe_experts(const moe_weights_t *w,
    const float *x_s,
    const int *indices_s, const float *weights_s,
    float expert_contribs[8][D_MODEL],
    void *model_ptr);
#endif

#ifdef GPU_SUPPORT
// GPU SSM recurrence forward
void wubu_gpu_ssm_recurrence(
    float *ssm_state,
    const float *q, const float *k, const float *v,
    const float *beta, const float *gate,
    float *delta_out,
    void *stream);
#endif

int wubu_moe_load_layer(gguf_ctx *ctx, int layer, moe_weights_t *moe) {
    char name[256];
    memset(moe, 0, sizeof(*moe));
    
    // Router: ffn_gate_inp.weight [D_MODEL, N_EXPERTS]
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", layer);
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_gate_inp = (float *)malloc(D_MODEL * N_EXPERTS * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_gate_inp, D_MODEL * N_EXPERTS))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
    
    // Expert gate
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    int64_t n = (int64_t)D_MODEL * D_FF * N_EXPERTS;
    moe->ffn_gate_exps = (float *)malloc(n * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_gate_exps, n))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
    
    // Expert up
    snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_up_exps = (float *)malloc(n * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_up_exps, n))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
    
    // Expert down
    snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    n = (int64_t)D_FF * D_MODEL * N_EXPERTS;
    moe->ffn_down_exps = (float *)malloc(n * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_down_exps, n))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
    
    // Shared expert gate
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_gate_shexp = (float *)malloc(D_MODEL * SHARED_D_FF * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_gate_shexp, D_MODEL * SHARED_D_FF))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
    
    // Shared expert up
    snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_up_shexp = (float *)malloc(D_MODEL * SHARED_D_FF * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_up_shexp, D_MODEL * SHARED_D_FF))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
    
    // Shared expert down
    snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE load: missing %s\n", name); return 0; }
    moe->ffn_down_shexp = (float *)malloc(SHARED_D_FF * D_MODEL * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, moe->ffn_down_shexp, SHARED_D_FF * D_MODEL))
        { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
    
    // ffn_gate_inp_shexp — shared expert output gate: sigmoid(x_s @ this) scales output
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) {
        // Some models don't have this tensor; gate defaults to 1.0
        moe->ffn_gate_inp_shexp = NULL;
        printf("  Layer %d: no shared expert gate (ffn_gate_inp_shexp)\n", layer);
    } else {
        moe->ffn_gate_inp_shexp = (float *)malloc(D_MODEL * sizeof(float));
        if (!gguf_read_tensor_f32(ctx, t, moe->ffn_gate_inp_shexp, D_MODEL))
            { fprintf(stderr, "MoE load: failed %s\n", name); return 0; }
        printf("  Layer %d: shared expert gate loaded (f32, %d elems)\n", layer, D_MODEL);
    }
    
    moe->loaded = true;
    return 1;
}

void wubu_moe_free_layer(moe_weights_t *moe) {
    if (!moe || !moe->loaded) return;
    free(moe->ffn_gate_inp);
    free(moe->ffn_gate_exps);
    free(moe->ffn_up_exps);
    free(moe->ffn_down_exps);
    free(moe->ffn_gate_shexp);
    free(moe->ffn_up_shexp);
    free(moe->ffn_down_shexp);
    free(moe->ffn_gate_inp_shexp);
    memset(moe, 0, sizeof(*moe));
}

// ============================================================
// MoE Router: x @ gate_inp → softmax → top-k selection
// ============================================================

void wubu_moe_router(const float *x, int B, int T,
                     const float *gate_inp,
                     float *scores) {
    // x: [B, T, D_MODEL]
    // gate_inp: [D_MODEL, N_EXPERTS]
    // scores: [B*T, N_EXPERTS] (router logits, pre-softmax)
    int N = B * T;
    
    #pragma omp parallel for if(N > 1)
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *score_s = scores + s * N_EXPERTS;
        
        // x_s @ gate_inp [2048] @ [2048, 256] -> [256]
        for (int e = 0; e < N_EXPERTS; e++) {
            float sum = 0.0f;
            for (int k = 0; k < D_MODEL; k++) {
                sum += x_s[k] * gate_inp[k + e * D_MODEL];
            }
            score_s[e] = sum;
        }
    }
}

// ============================================================
// N64 RDRAM Pre-Cache Fill: Router-Only on Pre-Attention Normed
// ============================================================
// Computes router scores + softmax + top-8 on normed (BEFORE SSM/GQA).
// Gives THIS layer's actual expert indices for prefetch during SSM.
// The attn_out is typically 10-20% of residual, so expert selection
// on normed ≈ normed2 — good enough for speculative L3 fill.
void wubu_moe_router_only(const float *x, int B, int T,
                          const moe_weights_t *w,
                          int *indices_out) {
    int N = B * T;
    float *scores = (float *)malloc(N * N_EXPERTS * sizeof(float));
    if (!scores) return;

    // Step 1: Router scores: x_s @ ffn_gate_inp [2048] @ [2048, 256] -> [256]
    #pragma omp parallel for if(N > 1)
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *score_s = scores + s * N_EXPERTS;
        for (int e = 0; e < N_EXPERTS; e++) {
            float sum = 0.0f;
            for (int k = 0; k < D_MODEL; k++)
                sum += x_s[k] * w->ffn_gate_inp[k + e * D_MODEL];
            score_s[e] = sum;
        }
    }

    // Step 2: Softmax + top-8 per token
    for (int s = 0; s < N; s++) {
        float *score_s = scores + s * N_EXPERTS;
        int *idx_s = indices_out + s * N_ACTIVE_EXPTS;

        // Softmax with numerical stability
        float max_s = score_s[0];
        for (int e = 1; e < N_EXPERTS; e++)
            if (score_s[e] > max_s) max_s = score_s[e];

        float sum_exp = 0.0f;
        for (int e = 0; e < N_EXPERTS; e++)
            sum_exp += expf(score_s[e] - max_s);
        float inv_sum = 1.0f / (sum_exp + 1e-30f);

        // Top-8 single pass
        float temp_w[N_ACTIVE_EXPTS];
        int   temp_i[N_ACTIVE_EXPTS];

        // Initialize with first 8
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            temp_i[k] = k;
            temp_w[k] = expf(score_s[k] - max_s) * inv_sum;
        }
        // Sort ascending (worst first)
        for (int i = 0; i < N_ACTIVE_EXPTS-1; i++)
            for (int j = i+1; j < N_ACTIVE_EXPTS; j++)
                if (temp_w[i] > temp_w[j]) {
                    float tw = temp_w[i]; temp_w[i] = temp_w[j]; temp_w[j] = tw;
                    int ti = temp_i[i]; temp_i[i] = temp_i[j]; temp_i[j] = ti;
                }

        // Remaining 248 experts: replace worst if better
        for (int e = N_ACTIVE_EXPTS; e < N_EXPERTS; e++) {
            float val = expf(score_s[e] - max_s) * inv_sum;
            if (val > temp_w[0]) {
                temp_w[0] = val;
                temp_i[0] = e;
                int pos = 0;
                while (pos + 1 < N_ACTIVE_EXPTS && temp_w[pos] > temp_w[pos+1]) {
                    float tw = temp_w[pos]; temp_w[pos] = temp_w[pos+1]; temp_w[pos+1] = tw;
                    int ti = temp_i[pos]; temp_i[pos] = temp_i[pos+1]; temp_i[pos+1] = ti;
                    pos++;
                }
            }
        }

        // Write output
        for (int k = 0; k < N_ACTIVE_EXPTS; k++)
            idx_s[k] = temp_i[k];
    }

    free(scores);
}

// ============================================================
// Quantized MoE Expert Computation (on-the-fly IQ2_XXS dequant)
// ============================================================

// Load one expert's quantized weights from the GGUF data blob
// Returns raw_size on success, 0 on failure
int wubu_moe_load_layer_quant(gguf_ctx *ctx, int layer,
                              uint8_t *gate_q, uint8_t *up_q, uint8_t *down_q,
                              int64_t *gate_raw_size, int64_t *up_raw_size, int64_t *down_raw_size) {
    char name[256];

    // Expert gate — quantized [D_MODEL, D_FF, N_EXPERTS]
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer);
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE quant load: missing %s\n", name); return 0; }
    int64_t n_elems = (int64_t)D_MODEL * D_FF;
    int64_t raw_sz = gguf_raw_size(t->ggml_type, n_elems);
    if (raw_sz <= 0) { fprintf(stderr, "MoE quant load: unsupported type %d\n", t->ggml_type); return 0; }
    if (!ctx->data_blob) {
        fprintf(stderr, "MoE quant load: data blob not buffered; call gguf_buffer_data first\n");
        return 0;
    }
    const uint8_t *src = (const uint8_t *)ctx->data_blob + t->data_offset;
    memcpy(gate_q, src, raw_sz);  // first expert only (expert 0)
    if (gate_raw_size) *gate_raw_size = raw_sz;

    // Expert up — quantized [D_MODEL, D_FF, N_EXPERTS]
    snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE quant load: missing %s\n", name); return 0; }
    raw_sz = gguf_raw_size(t->ggml_type, n_elems);
    src = (const uint8_t *)ctx->data_blob + t->data_offset;
    memcpy(up_q, src, raw_sz);
    if (up_raw_size) *up_raw_size = raw_sz;

    // Expert down — quantized [D_FF, D_MODEL, N_EXPERTS]
    snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "MoE quant load: missing %s\n", name); return 0; }
    n_elems = (int64_t)D_FF * D_MODEL;
    raw_sz = gguf_raw_size(t->ggml_type, n_elems);
    src = (const uint8_t *)ctx->data_blob + t->data_offset;
    memcpy(down_q, src, raw_sz);
    if (down_raw_size) *down_raw_size = raw_sz;

    return 1;
}

// Compute one expert with on-the-fly IQ2_XXS dequant dot product
// gate_q/up_q: quantized weight for one expert [D_MODEL, D_FF], column-major
// down_q: quantized weight for one expert [D_FF, D_MODEL], column-major
// temp: [D_FF * 3] scratch
// output: [D_MODEL]
//
// Memory layout of quantized gate/up (D_MODEL=2048, D_FF=512):
//   Each column of 2048 elements = 8 blocks × 66 bytes = 528 bytes
//   Total: 512 columns × 528 bytes = 270,336 bytes
//
// Memory layout of quantized down (D_FF=512, D_MODEL=2048):
//   Each column of 512 elements = 2 blocks × 66 bytes = 132 bytes
//   Total: 2048 columns × 132 bytes = 270,336 bytes
//
void moe_expert_forward_dequant(const float *x,
                                const uint8_t *gate_q, const uint8_t *up_q, const uint8_t *down_q,
                                float *temp, float *output) {
    // temp layout: [gate_out(D_FF) | up_out(D_FF) | act(D_FF)]
    float *gate_out = temp;
    float *up_out = temp + D_FF;
    float *act = temp + 2 * D_FF;

    const int blocks_per_col = D_MODEL / 256;  // 8
    const int gate_col_bytes = blocks_per_col * 66;  // 528 bytes per column

    // gate = x @ gate_q  [D_MODEL] @ [D_MODEL, D_FF] -> [D_FF]
    // For each output column j: dot over 8 IQ2_XXS blocks
    for (int j = 0; j < D_FF; j++) {
        const uint8_t *qcol = gate_q + j * gate_col_bytes;
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_col; b++) {
            sum += iq2_xxs_dot_block(qcol + b * 66, x + b * 256);
        }
        gate_out[j] = sum;
    }

    // up = x @ up_q  [D_MODEL] @ [D_MODEL, D_FF] -> [D_FF]
    for (int j = 0; j < D_FF; j++) {
        const uint8_t *qcol = up_q + j * gate_col_bytes;
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_col; b++) {
            sum += iq2_xxs_dot_block(qcol + b * 66, x + b * 256);
        }
        up_out[j] = sum;
    }

    // act = silu(gate) * up
    for (int j = 0; j < D_FF; j++) {
        float g = gate_out[j];
        float silu_g;
        if (g < -80.0f) silu_g = 0.0f;
        else silu_g = g / (1.0f + expf(-g));
        act[j] = silu_g * up_out[j];
    }

    // output = act @ down_q  [D_FF] @ [D_FF, D_MODEL] -> [D_MODEL]
    const int blocks_per_down_col = D_FF / 256;  // 2
    const int down_col_bytes = blocks_per_down_col * 66;  // 132 bytes per column
    for (int j = 0; j < D_MODEL; j++) {
        const uint8_t *qcol = down_q + j * down_col_bytes;
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_down_col; b++) {
            sum += iq2_xxs_dot_block(qcol + b * 66, act + b * 256);
        }
        output[j] = sum;
    }
}

// ============================================================
// MoE Expert Computation (single expert for one token)
// ============================================================

static void moe_expert_forward(
    const float *x,          // [D_MODEL]
    const float *gate_weight, // [D_MODEL, D_FF]
    const float *up_weight,   // [D_MODEL, D_FF]
    const float *down_weight, // [D_FF, D_MODEL]
    float *temp,              // [D_FF * 3] scratch
    float *output)           // [D_MODEL]
{
    // temp layout: [gate_out(D_FF) | up_out(D_FF) | act(D_FF)]
    float *gate_out = temp;
    float *up_out = temp + D_FF;
    float *act = temp + 2 * D_FF;
    
    // gate = x @ gate_weight  [D_MODEL] @ [D_MODEL, D_FF] -> [D_FF]
    for (int j = 0; j < D_FF; j++) {
        float sum = 0.0f;
        for (int k = 0; k < D_MODEL; k++)
            sum += x[k] * gate_weight[k + j * D_MODEL];
        gate_out[j] = sum;
    }
    
    // up = x @ up_weight  [D_MODEL] @ [D_MODEL, D_FF] -> [D_FF]
    for (int j = 0; j < D_FF; j++) {
        float sum = 0.0f;
        for (int k = 0; k < D_MODEL; k++)
            sum += x[k] * up_weight[k + j * D_MODEL];
        up_out[j] = sum;
    }
    
    // act = silu(gate) * up
    for (int j = 0; j < D_FF; j++) {
        float g = gate_out[j];
        // silu: g * sigmoid(g)
        float silu_g;
        if (g < -80.0f) silu_g = 0.0f;
        else silu_g = g / (1.0f + expf(-g));
        act[j] = silu_g * up_out[j];
    }
    
    // output = act @ down_weight  [D_FF] @ [D_FF, D_MODEL] -> [D_MODEL]
    for (int j = 0; j < D_MODEL; j++) {
        float sum = 0.0f;
        for (int k = 0; k < D_FF; k++)
            sum += act[k] * down_weight[k + j * D_FF];
        output[j] = sum;
    }
}

// ============================================================
// MoE Layer Forward Pass
// ============================================================

void wubu_moe_forward(const float *x, int B, int T,
                      const moe_weights_t *w,
                      float *output,
                      int *selected_experts) {
    if (!w->loaded) {
        // No MoE weights loaded: pass through
        memcpy(output, x, B * T * D_MODEL * sizeof(float));
        return;
    }
    
    int N = B * T;
    
    // Allocate scratch
    float *scores = (float *)malloc(N * N_EXPERTS * sizeof(float));
    int *topk_indices = (int *)malloc(N * N_ACTIVE_EXPTS * sizeof(int));
    float *topk_weights = (float *)malloc(N * N_ACTIVE_EXPTS * sizeof(float));
    float *shared_gate_all = (float *)malloc(N * SHARED_D_FF * sizeof(float));
    float *shared_up_all = (float *)malloc(N * SHARED_D_FF * sizeof(float));
    float *shared_act_all = (float *)malloc(N * SHARED_D_FF * sizeof(float));
    float *expert_temp_all = (float *)malloc(N * D_FF * 3 * sizeof(float));
    
    if (!scores || !topk_indices || !topk_weights || !shared_gate_all || !shared_up_all || !shared_act_all || !expert_temp_all) {
        fprintf(stderr, "MoE forward: allocation failed\n");
        free(scores); free(topk_indices); free(topk_weights);
        free(shared_gate_all); free(shared_up_all); free(shared_act_all); free(expert_temp_all);
        memcpy(output, x, N * D_MODEL * sizeof(float));
        return;
    }
    
    // Step 1-2: Router scores → softmax → top-k (or use pre-computed indices)
    if (w->precomputed_indices) {
        // N64 pre-cache fill path: router already computed on pre-attention normed.
        // Compute softmax weights for the 8 pre-selected experts on this input.
        // Softmax over subset is mathematically identical to full softmax for the
        // selected scores (normalization cancels to same relative weights).
        for (int s = 0; s < N; s++) {
            const float *x_s = x + s * D_MODEL;
            const int *idx_s = w->precomputed_indices + s * N_ACTIVE_EXPTS;
            int *indices_s = topk_indices + s * N_ACTIVE_EXPTS;
            float *weights_s = topk_weights + s * N_ACTIVE_EXPTS;

            // Compute scores for selected 8 experts: x_s @ ffn_gate_inp[:, e]
            float scores8[N_ACTIVE_EXPTS];
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                int e = idx_s[k];
                float sum = 0.0f;
                for (int i = 0; i < D_MODEL; i++)
                    sum += x_s[i] * w->ffn_gate_inp[i + e * D_MODEL];
                scores8[k] = sum;
            }

            // Softmax over selected 8
            float max_s = scores8[0];
            for (int k = 1; k < N_ACTIVE_EXPTS; k++)
                if (scores8[k] > max_s) max_s = scores8[k];
            float sum_exp = 0.0f;
            for (int k = 0; k < N_ACTIVE_EXPTS; k++)
                sum_exp += expf(scores8[k] - max_s);
            float inv_sum = 1.0f / (sum_exp + 1e-30f);
            float tw = 0.0f;
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                float w_val = expf(scores8[k] - max_s) * inv_sum;
                weights_s[k] = w_val;
                indices_s[k] = idx_s[k];
                tw += w_val;
            }
            // Normalize top-k weights to sum to 1
            if (tw > 1e-30f) {
                float inv_tw = 1.0f / tw;
                for (int k = 0; k < N_ACTIVE_EXPTS; k++)
                    weights_s[k] *= inv_tw;
            }
        }
    } else {
        // Full router path: all 256 scores, softmax, top-k selection
        // Step 1: Compute router scores
        wubu_moe_router(x, B, T, w->ffn_gate_inp, scores);

        // Step 2: Softmax and top-k for each token
        // (Using softmax — the model was trained with softmax routing.
        //  Sigmoid gating is a training-time optimization for load balancing.)
        for (int s = 0; s < N; s++) {
            float *score_s = scores + s * N_EXPERTS;

            // Find max for numerical stability
            float max_s = score_s[0];
            for (int e = 1; e < N_EXPERTS; e++)
                if (score_s[e] > max_s) max_s = score_s[e];

            // Compute softmax denominator
            float sum_exp = 0.0f;
            for (int e = 0; e < N_EXPERTS; e++)
                sum_exp += expf(score_s[e] - max_s);
            float inv_sum = 1.0f / (sum_exp + 1e-30f);

            // Compute softmax values
            float softmax_vals[N_EXPERTS];
            for (int e = 0; e < N_EXPERTS; e++)
                softmax_vals[e] = expf(score_s[e] - max_s) * inv_sum;

            // Find top-k indices — single pass O(E·K)
            int *indices_s = topk_indices + s * N_ACTIVE_EXPTS;
            float *weights_s = topk_weights + s * N_ACTIVE_EXPTS;

            // Initialize with first k elements
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                indices_s[k] = k;
                weights_s[k] = softmax_vals[k];
            }
            // Sort first k ascending (worst first)
            for (int i = 0; i < N_ACTIVE_EXPTS-1; i++)
                for (int j = i+1; j < N_ACTIVE_EXPTS; j++)
                    if (weights_s[i] > weights_s[j]) {
                        float tmp_w = weights_s[i]; weights_s[i] = weights_s[j]; weights_s[j] = tmp_w;
                        int tmp_i = indices_s[i]; indices_s[i] = indices_s[j]; indices_s[j] = tmp_i;
                    }

            // Single pass: for each remaining expert, replace worst if better
            for (int e = N_ACTIVE_EXPTS; e < N_EXPERTS; e++) {
                float val = softmax_vals[e];
                if (val > weights_s[0]) {  // better than worst in top-k
                    weights_s[0] = val;
                    indices_s[0] = e;
                    // Bubble down to maintain sorted order (worst first)
                    int pos = 0;
                    while (pos + 1 < N_ACTIVE_EXPTS && weights_s[pos] > weights_s[pos+1]) {
                        float tw = weights_s[pos]; weights_s[pos] = weights_s[pos+1]; weights_s[pos+1] = tw;
                        int ti = indices_s[pos]; indices_s[pos] = indices_s[pos+1]; indices_s[pos+1] = ti;
                        pos++;
                    }
                }
            }

            // Normalize top-k weights to sum to 1
            float sum_w = 0.0f;
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) sum_w += weights_s[k];
            if (sum_w > 1e-30f) {
                float inv_sum_w = 1.0f / sum_w;
                for (int k = 0; k < N_ACTIVE_EXPTS; k++) weights_s[k] *= inv_sum_w;
            }
        }
    }
    
    // Save selected expert indices for prefetch (if caller requested)
    if (selected_experts) {
        memcpy(selected_experts, topk_indices, N * N_ACTIVE_EXPTS * sizeof(int));
    }
    
    // Step 3: Process each token through selected experts + shared expert
    // Single parallel region: uses tasks for expert dispatch (no nested teams)
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int s = 0; s < N; s++) {
            const float *x_s = x + s * D_MODEL;
            float *out_s = output + s * D_MODEL;
            int *indices_s = topk_indices + s * N_ACTIVE_EXPTS;
            float *weights_s = topk_weights + s * N_ACTIVE_EXPTS;
            
            // Thread-local buffers (each thread gets its own)
            float shared_gate[SHARED_D_FF];
            float shared_up[SHARED_D_FF];
            float shared_act[SHARED_D_FF];
            
            // ---- Shared expert (sequential per token) ----
            // Quantize x_s once, reuse Q8 for gate+up projections
            if (w->ffn_gate_shexp_q) {
                uint8_t shexp_q8_buf[4096];
                quantize_row_q8_K(x_s, (block_q8_K *)shexp_q8_buf, D_MODEL);
                quantized_matmul_from_q8(shexp_q8_buf,
                    w->ffn_gate_shexp_q, w->ffn_gate_shexp_q_type,
                    D_MODEL, SHARED_D_FF, 0, shared_gate);
                quantized_matmul_from_q8(shexp_q8_buf,
                    w->ffn_up_shexp_q, w->ffn_up_shexp_q_type,
                    D_MODEL, SHARED_D_FF, 0, shared_up);
                for (int j = 0; j < SHARED_D_FF; j++) {
                    float g = shared_gate[j];
                    shared_act[j] = (g < -80.0f ? 0.0f : g / (1.0f + expf(-g))) * shared_up[j];
                }
                quantized_matmul(shared_act, w->ffn_down_shexp_q, w->ffn_down_shexp_q_type,
                                SHARED_D_FF, D_MODEL, 0, out_s);
            } else {
                for (int j = 0; j < SHARED_D_FF; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < D_MODEL; k++)
                        sum += x_s[k] * w->ffn_gate_shexp[k + j * D_MODEL];
                    shared_gate[j] = sum;
                }
                for (int j = 0; j < SHARED_D_FF; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < D_MODEL; k++)
                        sum += x_s[k] * w->ffn_up_shexp[k + j * D_MODEL];
                    shared_up[j] = sum;
                }
                for (int j = 0; j < SHARED_D_FF; j++) {
                    float g = shared_gate[j];
                    shared_act[j] = (g < -80.0f ? 0.0f : g / (1.0f + expf(-g))) * shared_up[j];
                }
                for (int j = 0; j < D_MODEL; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < SHARED_D_FF; k++)
                        sum += shared_act[k] * w->ffn_down_shexp[k + j * SHARED_D_FF];
                    out_s[j] = sum;
                }
            }
            
            // Apply shared expert output gate: sigmoid(x_s @ ffn_gate_inp_shexp)
            if (w->ffn_gate_inp_shexp) {
                float gate_val = 0.0f;
                for (int k = 0; k < D_MODEL; k++)
                    gate_val += x_s[k] * w->ffn_gate_inp_shexp[k];
                float gate_sig = 1.0f / (1.0f + expf(-gate_val));
                for (int j = 0; j < D_MODEL; j++)
                    out_s[j] *= gate_sig;
            }
            
            // ---- Routed expert contributions via OpenMP tasks ----
            // Tasks avoid nested team creation overhead (~10-50μs)
            float expert_contribs[N_ACTIVE_EXPTS][D_MODEL];
            memset(expert_contribs, 0, sizeof(expert_contribs));
            
            if (w->gpu_ctx) {
#ifdef GPU_SUPPORT
                // GPU accelerated: all 8 experts in one kernel call
                wubu_model_gpu_moe_experts(w, x_s, indices_s, weights_s,
                    expert_contribs, w->gpu_ctx);
#else
                (void)expert_contribs;
#endif
            } else {
            #pragma omp taskgroup
            {
                for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                    #pragma omp task firstprivate(k) shared(x_s, w, indices_s, weights_s, expert_contribs)
                    {
                        int e = indices_s[k];
                        float wgt = weights_s[k];
                        float *exp_out = expert_contribs[k];
                        
                        // Clip expert index for pruned models (not all 256 experts loaded)
                        if ((w->n_experts_loaded > 0 && e >= w->n_experts_loaded) ||
                            e < 0 || wgt < 1e-30f) {
                            memset(exp_out, 0, D_MODEL * sizeof(float));
                        } else {
                            float gate_out[D_FF];
                            float up_out[D_FF];
                            float act[D_FF];
                        
                        if (w->ffn_gate_exps_q) {
                            // Q8_0 lazy dequant cache path (MTP draft head only)
                            if (w->q8_cache) {
                                mtp_iq_cache_t *cache = (mtp_iq_cache_t *)w->q8_cache;
                                bool was_miss;
                                mtp_iq_cache_slot_t *slot = mtp_iq_cache_get_slot(cache, e, &was_miss);
                                if (was_miss) {
                                    mtp_iq_cache_fill(slot, w, e);
                                }
                                // Use original vec_dot path against cached raw quantized bytes
                                uint8_t exp_q8_buf[4096];
                                quantize_row_q8_K(x_s, (block_q8_K *)exp_q8_buf, D_MODEL);
                                quantized_matmul_from_q8(exp_q8_buf,
                                    slot->gate_q, slot->gate_type,
                                    D_MODEL, D_FF, 0, gate_out);
                                quantized_matmul_from_q8(exp_q8_buf,
                                    slot->up_q, slot->up_type,
                                    D_MODEL, D_FF, 0, up_out);
                                for (int j = 0; j < D_FF; j++) {
                                    float g = gate_out[j];
                                    act[j] = (g < -80.0f ? 0.0f : g / (1.0f + expf(-g))) * up_out[j];
                                }
                                quantized_matmul(act, slot->down_q, slot->down_type,
                                    D_FF, D_MODEL, 0, exp_out);
                                for (int j = 0; j < D_MODEL; j++)
                                    exp_out[j] *= wgt;
                            } else {
                            int64_t gate_bytes = gguf_raw_size(w->ffn_gate_exps_q_type, (int64_t)D_MODEL * D_FF);
                            int64_t up_bytes   = gguf_raw_size(w->ffn_up_exps_q_type,   (int64_t)D_MODEL * D_FF);
                            int64_t down_bytes = gguf_raw_size(w->ffn_down_exps_q_type, (int64_t)D_FF * D_MODEL);
                            
                            const uint8_t *gate_q = w->ffn_gate_exps_q + (int64_t)e * gate_bytes;
                            const uint8_t *up_q   = w->ffn_up_exps_q   + (int64_t)e * up_bytes;
                            const uint8_t *down_q = w->ffn_down_exps_q + (int64_t)e * down_bytes;
                            
                            // Quantize x_s once, reuse for gate+up
                            uint8_t exp_q8_buf[4096];
                            quantize_row_q8_K(x_s, (block_q8_K *)exp_q8_buf, D_MODEL);
                            quantized_matmul_from_q8(exp_q8_buf,
                                gate_q, w->ffn_gate_exps_q_type,
                                D_MODEL, D_FF, 0, gate_out);
                            quantized_matmul_from_q8(exp_q8_buf,
                                up_q, w->ffn_up_exps_q_type,
                                D_MODEL, D_FF, 0, up_out);
                            
                            for (int j = 0; j < D_FF; j++) {
                                float g = gate_out[j];
                                act[j] = (g < -80.0f ? 0.0f : g / (1.0f + expf(-g))) * up_out[j];
                            }
                            
                            quantized_matmul(act, down_q, w->ffn_down_exps_q_type, D_FF, D_MODEL, 0, exp_out);
                            
                            for (int j = 0; j < D_MODEL; j++)
                                exp_out[j] *= wgt;
                            }
                        } else {
                            const float *gate_w = w->ffn_gate_exps + (int64_t)e * D_MODEL * D_FF;
                            const float *up_w   = w->ffn_up_exps   + (int64_t)e * D_MODEL * D_FF;
                            const float *down_w = w->ffn_down_exps + (int64_t)e * D_FF * D_MODEL;
                            
                            for (int j = 0; j < D_FF; j++) {
                                float sum = 0.0f;
                                for (int ii = 0; ii < D_MODEL; ii++)
                                    sum += x_s[ii] * gate_w[ii + j * D_MODEL];
                                gate_out[j] = sum;
                            }
                            for (int j = 0; j < D_FF; j++) {
                                float sum = 0.0f;
                                for (int ii = 0; ii < D_MODEL; ii++)
                                    sum += x_s[ii] * up_w[ii + j * D_MODEL];
                                up_out[j] = sum;
                            }
                            for (int j = 0; j < D_FF; j++) {
                                float g = gate_out[j];
                                act[j] = (g < -80.0f ? 0.0f : g / (1.0f + expf(-g))) * up_out[j];
                            }
                            for (int j = 0; j < D_MODEL; j++) {
                                float sum = 0.0f;
                                for (int ii = 0; ii < D_FF; ii++)
                                    sum += act[ii] * down_w[ii + j * D_FF];
                                exp_out[j] = sum * wgt;
                            }
                        }
                    }
                }
            }
            }
            }
            // Accumulate expert contributions (sequential, no atomics needed)
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                for (int j = 0; j < D_MODEL; j++)
                    out_s[j] += expert_contribs[k][j];
            }
        }
    }
    
    free(scores);
    free(topk_indices);
    free(topk_weights);
    free(shared_gate_all);
    free(shared_up_all);
    free(shared_act_all);
    free(expert_temp_all);
}
