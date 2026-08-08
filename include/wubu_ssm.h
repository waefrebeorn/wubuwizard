#ifndef WUBU_SSM_H
#define WUBU_SSM_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Qwen3.6-35B-A3B Gated Delta Net (SSM) Module
// ============================================================

// Hyperparameters (fixed for Qwen3.6-35B-A3B qwen35moe architecture)
#define D_MODEL     2048   // hidden dimension
#define D_INNER     4096   // SSM inner dimension (value_dim)
#define SSM_K_HEADS 16     // SSM num_k_heads (ssm_n_group)
#define SSM_V_HEADS 32     // SSM num_v_heads (ssm_dt_rank)
#define SSM_D_STATE 128    // SSM state dimension (head_k_dim = head_v_dim)
#define KEY_DIM     (SSM_D_STATE * SSM_K_HEADS)   // 2048
#define VALUE_DIM   (SSM_D_STATE * SSM_V_HEADS)   // 4096
#define CONV_DIM    (KEY_DIM * 2 + VALUE_DIM)     // 8192 = Q(2048)+K(2048)+V(4096)
#define CONV_KERNEL 4      // conv1d kernel size
#define DT_RANK     32     // ssm_time_step_rank

// GQA hyperparameters
#define GQA_Q_HEADS    16
#define GQA_KV_HEADS   2
#define GQA_HEAD_DIM   256

// RoPE parameters (from Qwen3.6-35B config.json)
#define ROPE_THETA          10000000.0f  // rope_theta
#define PARTIAL_ROTARY_FACTOR 0.25f     // partial_rotary_factor
#define ROTARY_DIM          ((int)(GQA_HEAD_DIM * PARTIAL_ROTARY_FACTOR))  // 64

// MRoPE sections (from config: rope.dimension_sections = [11, 11, 10, 0])
// These define how the 32 frequency pairs are split across text/height/width.
// For text-only, all positions are equal but frequencies restart per section.
#define MRoPE_SECTIONS      3
#define MRoPE_SEC0_PAIRS    11
#define MRoPE_SEC1_PAIRS    11
#define MRoPE_SEC2_PAIRS    10
// Total: 11+11+10 = 32 pairs = 64 dims

// All weights for one SSM layer
typedef struct {
    // Fused QKV projection: x @ attn_qkv -> [Q(2048), K(2048), V(4096)]
    float *attn_qkv_weight;  // [D_MODEL, KEY_DIM*2+VALUE_DIM] = [2048, 8192]
    
    // Gate (z) projection: x @ attn_gate -> [4096]
    float *attn_gate_weight;  // [D_MODEL, VALUE_DIM] = [2048, 4096]
    
    // SSM projections
    float *ssm_beta_weight;   // [D_MODEL, DT_RANK] = [2048, 32]
    float *ssm_alpha_weight;  // [D_MODEL, DT_RANK] = [2048, 32]
    float *ssm_dt_bias;       // [DT_RANK] = [32]
    float *ssm_a;             // [DT_RANK] = [32]  (-A_log)
    
    // Convolution
    float *ssm_conv1d_weight; // [CONV_KERNEL, CONV_DIM] = [4, 8192]
    
    // Gated normalization
    float *ssm_norm_weight;   // [SSM_D_STATE] = [128]
    
    // Output projection
    float *ssm_out_weight;    // [VALUE_DIM, D_MODEL] = [4096, 2048]
    
    // Quantized weight pointers (into GGUF data_blob, don't free)
    const uint8_t *attn_qkv_weight_q;   // raw Q5_K
    int attn_qkv_weight_type;
    const uint8_t *attn_gate_weight_q;  // raw Q5_K
    int attn_gate_weight_type;
    const uint8_t *ssm_out_weight_q;    // raw Q6_K
    int ssm_out_weight_type;
    
    // Pre-attention and post-attention norms
    float *attn_norm_weight;          // [D_MODEL] = [2048]
    float *post_attention_norm_weight; // [D_MODEL] = [2048]
    
    // GPU recurrence state (optional, set by wubu_model.c when GPU active)
    void *gpu_ssm_state;     // device: [V_HEADS][D_STATE][D_STATE]
    void *gpu_q_buf;         // device: [V_HEADS][D_STATE]
    void *gpu_k_buf;         // device: [V_HEADS][D_STATE]
    void *gpu_v_buf;         // device: [V_HEADS][D_STATE]
    void *gpu_beta_buf;      // device: [V_HEADS]
    void *gpu_gate_buf;      // device: [V_HEADS]
    void *gpu_delta_buf;     // device: [V_HEADS][D_STATE]
    void *gpu_stream;        // CUDA stream (void* to avoid CUDA dependency in header)
} ssm_layer_weights;

// All weights for one GQA layer
typedef struct {
    // Q + gate fused: wq [D_MODEL, GQA_Q_HEADS*GQA_HEAD_DIM*2] = [2048, 8192]
    float *attn_q_weight;      // [2048, 8192]
    // K projection
    float *attn_k_weight;      // [D_MODEL, GQA_KV_HEADS*GQA_HEAD_DIM] = [2048, 512]
    // V projection
    float *attn_v_weight;      // [D_MODEL, GQA_KV_HEADS*GQA_HEAD_DIM] = [2048, 512]
    // Output projection
    float *attn_output_weight; // [GQA_Q_HEADS*GQA_HEAD_DIM, D_MODEL] = [4096, 2048]
    
    // Quantized weight pointers (into GGUF data_blob, don't free)
    const uint8_t *attn_q_weight_q;        // raw Q5_K
    int attn_q_weight_type;
    const uint8_t *attn_k_weight_q;        // raw Q5_K
    int attn_k_weight_type;
    const uint8_t *attn_v_weight_q;        // raw Q5_K
    int attn_v_weight_type;
    const uint8_t *attn_output_weight_q;   // raw Q5_K
    int attn_output_weight_type;
    
    // Q/K norms
    float *attn_q_norm_weight;  // [GQA_HEAD_DIM] = [256]
    float *attn_k_norm_weight;  // [GQA_HEAD_DIM] = [256]
    
    // Pre/post norms
    float *attn_norm_weight;          // [D_MODEL] = [2048]
    float *post_attention_norm_weight; // [D_MODEL] = [2048]
} gqa_layer_weights;

// Full model state (for SSM recurrent state)
typedef struct {
    int n_layers;
    bool *is_ssm;             // layer_types[40]: which layers are SSM
    
    // Per-layer weights (union of SSM and GQA)
    ssm_layer_weights *ssm_layers;   // 30 layers
    gqa_layer_weights *gqa_layers;   // 10 layers
    
    // SSM recurrent states [layer][head][128][128]
    float ***ssm_states;  // [n_layers][SSM_V_HEADS][SSM_D_STATE][SSM_D_STATE]
    
    // Conv states [layer][conv_kernel-1][conv_dim]
    float **conv_states;  // [n_layers][CONV_KERNEL-1][CONV_DIM]
} wubu_model;

// ============================================================
// SSM Workspace: pre-allocate all intermediate buffers once
// and reuse across layers to eliminate 17 malloc/free per layer.
// ============================================================
typedef struct {
    float *qkv_all;       // [N, KEY_DIM*2+VALUE_DIM]
    float *z_all;         // [N, VALUE_DIM]
    float *beta_raw;      // [N, DT_RANK]
    float *alpha_raw;     // [N, DT_RANK]
    float *conv_input;    // [B, T+CONV_KERNEL-1, CONV_DIM]
    float *conv_output;   // [N, CONV_DIM]
    float *q_conv;        // [N, KEY_DIM]
    float *k_conv;        // [N, KEY_DIM]
    float *v_conv;        // [N, VALUE_DIM]
    float *q_norm;        // [N, KEY_DIM]
    float *k_norm;        // [N, KEY_DIM]
    float *delta_out;     // [N, VALUE_DIM]
    float *z_silu;        // [N, VALUE_DIM]
    float *beta_flat;     // [N, DT_RANK]
    float *gate_flat;     // [N, DT_RANK]
    float *alpha_biased;  // [N, DT_RANK]
    float *alpha_softplus;// [N, DT_RANK]
    int B, T, N;          // dimensions used for allocation
} ssm_workspace_t;

// Allocate SSM workspace for given batch/token dimensions.
// Returns NULL on allocation failure. All pointers are 64-byte aligned.
ssm_workspace_t *wubu_ssm_workspace_alloc(int B, int T);

// Free SSM workspace (NULL-safe).
void wubu_ssm_workspace_free(ssm_workspace_t *ws);

// ============================================================
// Forward pass functions
// ============================================================

// SSM L2 norm epsilon (global, set from GGUF config)
extern float g_ssm_l2_eps;

// Single SSM layer forward pass
// x: [B, T, D_MODEL]
// weights: SSM layer weights
// ssm_state: [SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE] (mutable)
// conv_state: [CONV_KERNEL-1, CONV_DIM] (mutable)
// output: [B, T, D_MODEL]
// ws: pre-allocated workspace (NULL = per-call malloc, backward compat)
void wubu_ssm_forward(const float *x, int B, int T,
                      const ssm_layer_weights *w,
                      float *ssm_state,
                      float *conv_state,
                      float *output,
                      const float *gpu_qkv, const float *gpu_z,
                      ssm_workspace_t *ws);

// Saved SSM forward intermediates (for backward pass)
// All arrays [B*T x dim] unless noted
typedef struct {
    float *qkv_all;      // [N, CONV_DIM]
    float *z_all;        // [N, VALUE_DIM]
    float *beta_raw;     // [N, DT_RANK]
    float *alpha_raw;    // [N, DT_RANK]
    float *conv_post_silu; // [N, CONV_DIM] (post-SiLU conv output)
    float *q_conv;       // [N, KEY_DIM]
    float *k_conv;       // [N, KEY_DIM]
    float *v_conv;       // [N, VALUE_DIM]
    float *q_norm;       // [N, KEY_DIM]
    float *k_norm;       // [N, KEY_DIM]
    float *delta_out;    // [N, VALUE_DIM] (pre-gated-norm)
    float *z_silu;       // [N, VALUE_DIM]
    float *beta_flat;    // [N, DT_RANK] sigmoid(beta_raw)
    float *gate_flat;    // [N, DT_RANK] alpha_softplus * ssm_a
    float *states_t;     // [(T+1), SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE] per-timestep states
    float *conv_state_copy; // [B, CONV_KERNEL-1, CONV_DIM] copy of conv_state
} ssm_fwd_save_t;

// Single SSM + save forward (pass save=NULL for standard forward)
// Does NOT use workspace (training-only path, called once per training step)
void wubu_ssm_forward_save(const float *x, int B, int T,
                           const ssm_layer_weights *w,
                           float *ssm_state,
                           float *conv_state,
                           float *output,
                           ssm_fwd_save_t *save);

// Single GQA layer forward pass
// x: [B, T, D_MODEL]
// GQA forward with KV cache support
// k_cache/v_cache: cached K_norm and V from previous decode steps (or NULL for first call)
// cache_len: number of cached positions (0 for first call)
// k_out/v_out: output buffers for the NEW K_norm and V (caller can cache these)
void wubu_gqa_forward(const float *x, int B, int T,
                      const gqa_layer_weights *weights,
                      float *output,
                      const void *k_cache, const void *v_cache, int cache_len,
                      void *k_out, void *v_out);

// Saved GQA forward intermediates (for backward pass)
typedef struct {
    float *Q_norm;    // [N, GQA_Q_HEADS * GQA_HEAD_DIM]
    float *Q_raw;     // [N, GQA_Q_HEADS * GQA_HEAD_DIM] (pre-RMSNorm)
    float *K_norm;    // [N, GQA_KV_HEADS * GQA_HEAD_DIM]
    float *K_raw;     // [N, GQA_KV_HEADS * GQA_HEAD_DIM] (pre-RMSNorm)
    float *V;         // [N, GQA_KV_HEADS * GQA_HEAD_DIM]
    float *gate;      // [N, GQA_Q_HEADS * GQA_HEAD_DIM] (pre-sigmoid)
    float *gate_sig;  // [N, GQA_Q_HEADS * GQA_HEAD_DIM] (sigmoid output)
    float *attn_out_pre_gate; // [N, GQA_Q_HEADS * GQA_HEAD_DIM]
} gqa_fwd_save_t;

// Single GQA + save forward (pass save=NULL for standard forward)
void wubu_gqa_forward_save(const float *x, int B, int T,
                           const gqa_layer_weights *weights,
                           float *output,
                           gqa_fwd_save_t *save);

// Single Poincaré SSM layer forward pass (hyperbolic recurrence)
// Same interface as wubu_ssm_forward but uses Möbius operations
// for the recurrence step.
void wubu_poincare_ssm_forward(const float *x, int B, int T,
                               const ssm_layer_weights *w,
                               float *ssm_state,
                               float *conv_state,
                               float R,
                               float *output);

// Single Poincaré GQA forward pass (hyperbolic attention)
// Same interface as wubu_gqa_forward but uses Poincaré distance
// instead of dot-product attention.
void wubu_poincare_gqa_forward(const float *x, int B, int T,
                               const gqa_layer_weights *weights,
                               float R,
                               float *output);

// Poincaré SSM backward pass (gyration chain rule)
// Uses saved state trajectory from gpu_poincare_ssm_forward_save
void wubu_poincare_ssm_backward(int B, int T, float R,
    const float *normed, const float *attn_out, const float *d_attn_out,
    const ssm_layer_weights *w,
    const float *d_qkv, const float *d_z, const float *d_beta_r,
    const float *d_alpha_r, const float *d_conv, const float *d_q_c,
    const float *d_k_c, const float *d_v_c, const float *d_q_n,
    const float *d_k_n, const float *d_delta, const float *d_z_s,
    const float *d_states_t, const float *d_beta_s, const float *d_gate,
    const float *d_conv_s,
    float *d_normed,
    float *d_qkv_weight, float *d_gate_weight,
    float *d_beta_weight, float *d_alpha_weight,
    float *d_conv1d_weight, float *d_ssm_out_weight,
    float *d_ssm_norm_weight, float *d_state_init_grad);

// Chunked DeltaNet SSM recurrence (3x prefill speedup)
// Only supports B=1 currently. Uses chunked algorithm for T >= 64.
void wubu_ssm_chunked_recurrence(int B, int T,
                                  const float *q_norm,
                                  const float *k_norm,
                                  const float *v_conv,
                                  const float *beta_flat,
                                  const float *gate_flat,
                                  float *ssm_state,
                                  float *delta_out);

// Sequential SSM recurrence (exact match to original code, extracted for verification)
void wubu_ssm_sequential_recurrence(int B, int T,
                                     const float *q_norm,
                                     const float *k_norm,
                                     const float *v_conv,
                                     const float *beta_flat,
                                     const float *gate_flat,
                                     float *ssm_state,
                                     float *delta_out);

// Utility functions
int wubu_is_ssm_layer(int layer_idx);
void wubu_softplus(int n, const float *x, float *out);
void wubu_silu(int n, const float *x, float *out);
void wubu_sigmoid(int n, const float *x, float *out);
void wubu_l2_norm(int B, int T, int n_heads, int d,
                 const float *x, float eps, float *out);
void wubu_rms_norm(int B, int T, int d,
                   const float *x, const float *weight, float eps, float *out);
void wubu_conv1d(int B, int T, int C, int k,
                 const float *input, const float *kernel,
                 float *output);

// Qwen3.6 MRoPE
void wubu_rope(int B, int T, int n_heads, int head_dim,
               const float *x, const int *positions,
               int n_rot, const int *sections,
               float base, float *output);

// ============================================================
// Backward Pass Functions (Phase 4)
void wubu_ssm_backward_output_proj(
    const float *delta_out, const float *d_output,
    const float *ssm_out_weight,
    const uint8_t *ssm_out_weight_q, int ssm_out_weight_type,
    float *d_delta_out, float *d_ssm_out_weight, int N);

// Backward through gated normalization (Step 10)
void wubu_ssm_backward_gated_norm(
    const float *x, const float *z_silu,
    const float *d_out, const float *norm_w,
    float *d_x, float *d_z_silu, int B, int T);

// Backward through SiLU activation
void wubu_silu_backward(int n, const float *x, const float *y,
                        const float *dy, float *dx);

// Backward through L2 normalization
void wubu_l2_norm_backward(int B, int T, int n_heads, int d,
                           const float *x, float eps,
                           const float *d_out, float *d_x);

// Backward through SSM delta net recurrence (Step 9) — BPTT
void wubu_ssm_backward_recurrence(
    int B, int T,
    const float *saved_states,
    const float *q_norm, const float *k_norm,
    const float *v_conv,
    const float *beta_flat, const float *gate_flat,
    const float *d_output,
    float *d_q_norm, float *d_k_norm,
    float *d_v_conv,
    float *d_beta_flat, float *d_gate_flat,
    float *d_state_init);

// Full SSM layer backward (chains steps 11 through 0)
void wubu_ssm_backward(
    int B, int T,
    const float *x, const float *output, const float *d_output,
    const float *qkv_all, const float *z_all,
    const float *beta_raw, const float *alpha_raw,
    const float *conv_output,
    const float *q_conv, const float *k_conv, const float *v_conv,
    const float *q_norm, const float *k_norm,
    const float *delta_out, const float *z_silu,
    const float *ssm_states,
    const float *beta_flat, const float *gate_flat,
    const float *conv_state,   // [B, CONV_KERNEL-1, CONV_DIM] — for conv1d wgrad
    const ssm_layer_weights *w,
    float *d_x,
    float *d_qkv_weight, float *d_gate_weight,
    float *d_beta_weight, float *d_alpha_weight,
    float *d_conv1d_weight, float *d_ssm_out_weight,
    float *d_ssm_norm_weight,
    float *d_ssm_state_init);

// GQA attention backward (Step 5)
void wubu_gqa_backward_attention(
    int B, int T,
    const float *Q_norm, const float *K_norm, const float *V,
    const float *d_attn_out,
    float *d_Q, float *d_K, float *d_V);

// Full GQA layer backward (chains steps 7 through 1)
void wubu_gqa_backward(
    int B, int T,
    const float *x, const float *Q_norm, const float *Q_raw,
    const float *K_norm, const float *K_raw,
    const float *V,
    const float *gate, const float *gate_sig,
    const float *attn_out, const float *output,
    const float *d_output,
    const gqa_layer_weights *w,
    float *d_x,
    float *d_q_weight, float *d_k_weight, float *d_v_weight,
    float *d_q_norm_weight, float *d_k_norm_weight,
    float *d_out_weight);

// RMSNorm backward helper
void wubu_rms_norm_backward(int B, int T, int d,
                            const float *x, const float *weight, float eps,
                            const float *d_out, float *d_x);

#ifdef __cplusplus
}
#endif

#endif // WUBU_SSM_H
