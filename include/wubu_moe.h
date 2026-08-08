#ifndef WUBU_MOE_H
#define WUBU_MOE_H

#include <stdbool.h>

#include "gguf_reader.h"

#ifdef __cplusplus
extern "C" {
#endif

// MoE hyperparameters for Qwen3.6-35B-A3B
#define N_EXPERTS       256   // total routed experts
#define N_ACTIVE_EXPTS  8     // top-k experts per token
#define D_FF            512   // expert intermediate dimension
#define SHARED_D_FF     512   // shared expert intermediate dimension

// MoE weights for one layer
typedef struct {
    // Router
    float *ffn_gate_inp;      // [D_MODEL, N_EXPERTS] = [2048, 256] — router weight
    
    // Routed experts (3D tensors, expert index is slowest dim)
    float *ffn_gate_exps;     // [D_MODEL, D_FF, N_EXPERTS] = [2048, 512, 256]
    float *ffn_up_exps;       // [D_MODEL, D_FF, N_EXPERTS] = [2048, 512, 256]
    float *ffn_down_exps;     // [D_FF, D_MODEL, N_EXPERTS] = [512, 2048, 256]
    
    // Shared expert (always active)
    float *ffn_gate_shexp;    // [D_MODEL, SHARED_D_FF] = [2048, 512]
    float *ffn_up_shexp;      // [D_MODEL, SHARED_D_FF] = [2048, 512]
    float *ffn_down_shexp;    // [SHARED_D_FF, D_MODEL] = [512, 2048]
    
    // Router bias for shared expert
    float *ffn_gate_inp_shexp; // [D_MODEL] — shared expert output gate (per-token scalar via sigmoid)
    
    // Quantized weight pointers (raw GGUF blob, no dequant)
    const uint8_t *ffn_gate_exps_q;   // gate_exps quantized blob ptr
    int   ffn_gate_exps_q_type;       // e.g. GGML_TYPE_IQ2_XXS
    const uint8_t *ffn_up_exps_q;     // up_exps quantized blob ptr
    int   ffn_up_exps_q_type;
    const uint8_t *ffn_down_exps_q;   // down_exps quantized blob ptr
    int   ffn_down_exps_q_type;
    const uint8_t *ffn_gate_shexp_q;  // shared gate quantized blob ptr
    int   ffn_gate_shexp_q_type;
    const uint8_t *ffn_up_shexp_q;    // shared up quantized blob ptr
    int   ffn_up_shexp_q_type;
    const uint8_t *ffn_down_shexp_q;  // shared down quantized blob ptr
    int   ffn_down_shexp_q_type;
    
    // Whether weights are loaded (F32 heap-allocated or via quantized blob pointers)
    bool loaded;
    bool load_from_blob; // true: F32 pointers point into mmap'd blob, don't free

    // GPU context (set by wubu_model.c when GPU is active). If non-NULL,
    // wubu_moe_forward uses GPU for the 8 expert quantized matmuls.
    void *gpu_ctx;

    // Number of loaded routed experts (for pruned models where not all 256 are loaded)
    // 0 means all 256 are loaded (default). When < 256, expert indices from the router 
    // are clipped to [0, n_experts_loaded-1] in wubu_moe_forward.
    int n_experts_loaded;

    // Q8_0 lazy dequant cache for MTP draft head (blk.40 only).
    // When non-NULL, MoE forward uses Q8_0-cached version of IQ2 experts.
    // Cache struct defined in mtp_q8_cache.h, cast to mtp_q8_cache_t*.
    void *q8_cache;

    // Pre-computed expert indices from n64 pre-cache fill (wubu_model.c).
    // When non-NULL, wubu_moe_forward skips the full router + top-k and
    // computes softmax weights for these 8 indices directly.
    // Saves ~0.5ms/layer (2048×256 router matmul) on decode path.
    const int *precomputed_indices;
} moe_weights_t;

// MoE forward pass for one layer
// x: [B, T, D_MODEL] — input (post-attention normalized)
// output: [B, T, D_MODEL] — MoE output
// selected_experts: if non-NULL, filled with [N*N_ACTIVE_EXPTS] expert indices for prefetch
void wubu_moe_forward(const float *x, int B, int T,
                      const moe_weights_t *w,
                      float *output,
                      int *selected_experts);

// Load one layer's MoE weights from an open GGUF context
// Allocates and dequantizes all 3 expert tensors (O(3 GB))
// Caller must free with wubu_moe_free_layer after use
// Returns 1 on success, 0 on failure
int wubu_moe_load_layer(gguf_ctx *ctx, int layer, moe_weights_t *moe);

// Alternative: load MoE weights in quantized form (keep IQ2_XXS raw data)
// Memory-efficient: keeps ~10GB of quantized data instead of ~35GB of f32
// Writes raw_size bytes into gate_q, up_q, down_q
// Returns raw_size on success, 0 on failure
int wubu_moe_load_layer_quant(gguf_ctx *ctx, int layer,
                              uint8_t *gate_q, uint8_t *up_q, uint8_t *down_q,
                              int64_t *gate_raw_size, int64_t *up_raw_size, int64_t *down_raw_size);

// Compute one expert with on-the-fly IQ2_XXS dequant
// x: [D_MODEL] input
// gate_q/up_q/down_q: quantized weight data for one expert
// temp: [D_FF * 3] scratch
// output: [D_MODEL]
void moe_expert_forward_dequant(const float *x,
                                const uint8_t *gate_q, const uint8_t *up_q, const uint8_t *down_q,
                                float *temp, float *output);

// Free one layer's MoE weights
void wubu_moe_free_layer(moe_weights_t *moe);

// Helper: compute router logits and select top-k experts
// scores: [B*T, N_EXPERTS] — output router logits
void wubu_moe_router(const float *x, int B, int T,
                     const float *gate_inp,
                     float *scores);

// N64 pre-cache fill: compute router + top-k on pre-attention normed
// Returns THIS layer's expert indices (NOT prev layer's)
// indices_out: [B*T, N_ACTIVE_EXPTS] — top-k expert indices per token
void wubu_moe_router_only(const float *x, int B, int T,
                          const moe_weights_t *w,
                          int *indices_out);

// MoE backward pass
// d_output: [B*T, D_MODEL] — gradient at MoE output (FFN path)
// normed2: [B*T, D_MODEL] — saved MoE input
// d_normed2: [B*T, D_MODEL] — output gradient w.r.t. MoE input
// weight_grad_bufs: pre-allocated zero-initialized gradient buffers (or NULL to skip)
void wubu_moe_backward(const float *d_output, int B, int T,
                       const float *normed2,
                       const moe_weights_t *w,
                       float *d_normed2,
                       float *d_gate_inp,
                       float *d_gate_exps,
                       float *d_up_exps,
                       float *d_down_exps,
                       float *d_gate_shexp,
                       float *d_up_shexp,
                       float *d_down_shexp);

#ifdef __cplusplus
}
#endif

#endif // WUBU_MOE_H
