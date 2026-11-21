#ifndef WUBU_BENCH_H
#define WUBU_BENCH_H

#include "wubu_ssm.h"
#include "gguf_reader.h"
#include "cpu_timing.h"   // cycle-accurate timing: rdtsc, clflush, cache probe
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* CUDA is optional: x86/CUDA hosts keep the GPU helpers, ARM/CPU hosts do not. */
#if defined(__has_include)
#  if __has_include(<cuda_runtime.h>) && __has_include(<cublas_v2.h>)
#    define WUBU_HAVE_CUDA 1
#  endif
#endif
#ifdef WUBU_HAVE_CUDA
#  include "cuda_kernels.h"
#  include <cuda_runtime.h>
#  include <cublas_v2.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef WUBU_HAVE_CUDA

// ================================================================
// Timing helper
// ================================================================
static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ================================================================
// Comparison helpers
// ================================================================
static inline float max_abs_diff(const float *a, const float *b, int n) {
    float md = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > md) md = d;
    }
    return md;
}

static inline float max_abs_val(const float *a, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabsf(a[i]);
        if (v > m) m = v;
    }
    return m;
}

// ================================================================
// GPU SSM forward pass (all steps, one layer on GPU)
// Same interface as CPU wubu_ssm_forward but runs on GPU.
// All d_* arrays must be pre-allocated GPU memory.
// ================================================================
void gpu_ssm_forward(cublasHandle_t cublas_h, cudaStream_t stream,
                     const float *d_x, int B, int T,
                     // GPU weight pointers
                     const float *d_attn_qkv,
                     const float *d_attn_gate,
                     const float *d_ssm_beta,
                     const float *d_ssm_alpha,
                     const float *d_ssm_dt_bias,
                     const float *d_ssm_a,
                     const float *d_ssm_conv1d,
                     const float *d_ssm_norm,
                     const float *d_ssm_out,
                     // Mutable state (GPU)
                     float *d_ssm_state,
                     float *d_conv_state,
                     // Output (GPU)
                     float *d_output,
                     // Scratch buffers (GPU, pre-allocated by caller)
                     float *d_qkv,
                     float *d_z,
                     float *d_beta,
                     float *d_alpha,
                     float *d_beta_sig,
                     float *d_alpha_bi,
                     float *d_gate,
                     float *d_conv_input,
                     float *d_conv_out,
                     float *d_q_conv,
                     float *d_k_conv,
                     float *d_v_conv,
                     float *d_q_norm,
                     float *d_k_norm,
                     float *d_delta_out,
                     float *d_z_silu);

// GPU SSM forward with per-timestep state trajectory saving (for BPTT)
// d_states_t: [(T+1), SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE] or NULL
void gpu_ssm_forward_save(cublasHandle_t cublas_h, cudaStream_t stream,
                          const float *d_x, int B, int T,
                          const float *d_attn_qkv,
                          const float *d_attn_gate,
                          const float *d_ssm_beta,
                          const float *d_ssm_alpha,
                          const float *d_ssm_dt_bias,
                          const float *d_ssm_a,
                          const float *d_ssm_conv1d,
                          const float *d_ssm_norm,
                          const float *d_ssm_out,
                          float *d_ssm_state,
                          float *d_conv_state,
                          float *d_states_t,
                          float *d_output,
                          float *d_qkv,
                          float *d_z,
                          float *d_beta,
                          float *d_alpha,
                          float *d_beta_sig,
                          float *d_alpha_bi,
                          float *d_gate,
                          float *d_conv_input,
                          float *d_conv_out,
                          float *d_q_conv,
                          float *d_k_conv,
                          float *d_v_conv,
                          float *d_q_norm,
                          float *d_k_norm,
                          float *d_delta_out,
                          float *d_z_silu);

// ================================================================
// GPU GQA forward pass
// ================================================================
void gpu_gqa_forward(cublasHandle_t cublas_h, cudaStream_t stream,
                     const float *d_x, int B, int T,
                     const float *d_attn_q,
                     const float *d_attn_k,
                     const float *d_attn_v,
                     const float *d_attn_out_w,
                     const float *d_q_norm_w,
                     const float *d_k_norm_w,
                     // Output (GPU)
                     float *d_output,
                     // Scratch (pre-allocated)
                     float *d_Q_full,
                     float *d_K,
                     float *d_V,
                     float *d_scratch,
                     const float *d_sincos);

// GPU GQA forward with intermediate saves for backward
void gpu_gqa_forward_save(cublasHandle_t cublas_h, cudaStream_t stream,
                          const float *d_x, int B, int T,
                          const float *d_attn_q,
                          const float *d_attn_k,
                          const float *d_attn_v,
                          const float *d_attn_out_w,
                          const float *d_q_norm_w,
                          const float *d_k_norm_w,
                          float *d_output,
                          float *d_Q_full, float *d_K, float *d_V, float *d_scratch,
                          float *d_Q_norm_save, float *d_K_raw_save,
                          float *d_K_norm_save, float *d_attn_out_save);

// ================================================================
// Load SSM layer weights from GGUF (f32), allocate + upload to GPU
// ================================================================
typedef struct {
    float *d_attn_qkv;
    float *d_attn_gate;
    float *d_ssm_beta;
    float *d_ssm_alpha;
    float *d_ssm_dt_bias;
    float *d_ssm_a;
    float *d_ssm_conv1d;
    float *d_ssm_norm;
    float *d_ssm_out;
} gpu_ssm_weights;

int gpu_load_ssm_layer(gguf_ctx *ctx, int layer_idx,
                       gpu_ssm_weights *w, cudaStream_t stream);

void gpu_free_ssm_weights(gpu_ssm_weights *w);

// ================================================================
// Load GQA layer weights from GGUF (f32), allocate + upload to GPU
// ================================================================
typedef struct {
    float *d_attn_q;
    float *d_attn_k;
    float *d_attn_v;
    float *d_attn_out_w;
    float *d_q_norm_w;
    float *d_k_norm_w;
} gpu_gqa_weights;

int gpu_load_gqa_layer(gguf_ctx *ctx, int layer_idx,
                       gpu_gqa_weights *w, cudaStream_t stream);

void gpu_free_gqa_weights(gpu_gqa_weights *w);

// ================================================================
// GPU Output Projection — hiddens → logits via cuBLAS SGEMM
// output_weight_host: [vocab_size, D_MODEL] row-major (from GGUF)
// d_output_logits: [B*T, vocab_size] GPU output buffer
// ================================================================
float* gpu_upload_output_weight(cublasHandle_t handle, const float *host_weight,
                                 int vocab_size, cudaStream_t stream);
void gpu_output_projection(cublasHandle_t handle, cudaStream_t stream,
                           const float *d_hidden, int B, int T,
                           const float *d_output_weight, int vocab_size,
                           float *d_logits);
void gpu_free_output_weight(float *d_weight);

// CUDA kernel for output projection (replaces cuBLAS path)
#ifdef __cplusplus
extern "C" {
#endif
void launch_output_proj_kernel(cudaStream_t stream,
                               const float *d_hidden, int D,
                               const float *d_output_weight, int64_t V,
                               float *d_logits);
#ifdef __cplusplus
}
#endif

// ================================================================
// GPU Poincaré SSM forward pass (same interface + float R)
// ================================================================
void gpu_poincare_ssm_forward(cublasHandle_t cublas_h, cudaStream_t stream,
                     const float *d_x, int B, int T,
                     const float *d_attn_qkv,
                     const float *d_attn_gate,
                     const float *d_ssm_beta,
                     const float *d_ssm_alpha,
                     const float *d_ssm_dt_bias,
                     const float *d_ssm_a,
                     const float *d_ssm_conv1d,
                     const float *d_ssm_norm,
                     const float *d_ssm_out,
                     float *d_ssm_state,
                     float *d_conv_state,
                     float *d_output,
                     float *d_qkv,
                     float *d_z,
                     float *d_beta,
                     float *d_alpha,
                     float *d_beta_sig,
                     float *d_alpha_bi,
                     float *d_gate,
                     float *d_conv_input,
                     float *d_conv_out,
                     float *d_q_conv,
                     float *d_k_conv,
                     float *d_v_conv,
                     float *d_q_norm,
                     float *d_k_norm,
                     float *d_delta_out,
                     float *d_z_silu,
                     float R);

// GPU Poincaré SSM forward — save variant (captures state trajectory)
void gpu_poincare_ssm_forward_save(cublasHandle_t cublas_h, cudaStream_t stream,
                     const float *d_x, int B, int T,
                     const float *d_attn_qkv,
                     const float *d_attn_gate,
                     const float *d_ssm_beta,
                     const float *d_ssm_alpha,
                     const float *d_ssm_dt_bias,
                     const float *d_ssm_a,
                     const float *d_ssm_conv1d,
                     const float *d_ssm_norm,
                     const float *d_ssm_out,
                     float *d_ssm_state,
                     float *d_conv_state,
                     float *d_output,
                     float *d_qkv,
                     float *d_z,
                     float *d_beta,
                     float *d_alpha,
                     float *d_beta_sig,
                     float *d_alpha_bi,
                     float *d_gate,
                     float *d_conv_input,
                     float *d_conv_out,
                     float *d_q_conv,
                     float *d_k_conv,
                     float *d_v_conv,
                     float *d_q_norm,
                     float *d_k_norm,
                     float *d_delta_out,
                     float *d_z_silu,
                     float R,
                     float *d_states_t);

// ================================================================
// GPU SSM decode step (1 token, state carry)
// All d_* buffers must be GPU pointers. State persists on GPU.
// ================================================================
void gpu_ssm_decode_step(cublasHandle_t cublas_h, cudaStream_t stream,
    const float *d_x_step,
    gpu_ssm_weights *gw,
    float *d_ssm_state, float *d_conv_state,
    float *d_output,
    float *d_scratch_buf);

// ================================================================
// GPU GQA decode step (1 token, CPU KV cache)
// Uploads x_step, projects on GPU, downloads K/V for CPU KV cache,
// downloads Q_norm for CPU attention, then output proj on GPU.
// ================================================================
void gpu_gqa_decode_step(cublasHandle_t cublas_h, cudaStream_t stream,
    const float *d_x_step,
    gpu_gqa_weights *gw,
    // Scratch (pre-allocated on GPU)
    float *d_Q_full,
    float *d_K, float *d_V,
    float *d_attn_out,
    // Output from GPU projections (host buffers, caller-allocated)
    float *h_Q_out,    // [GQA_Q_HEADS * GQA_HEAD_DIM * 2] = [8192]
    float *h_K_out,    // [GQA_KV_DIM] = [512]
    float *h_V_out);   // [GQA_KV_DIM] = [512]

// GPU output projection for decode step (1 token)
void gpu_gqa_output_proj(cublasHandle_t cublas_h, cudaStream_t stream,
    const float *d_attn_out,    // [q_dim] on GPU
    gpu_gqa_weights *gw,
    float *d_output);           // [D_MODEL] on GPU

// ================================================================
// Allocate GPU scratch buffers for SSM forward (B*T tokens)
// Returns scratch size in floats.
// ================================================================
size_t gpu_ssm_scratch_needed(int B, int T);

// Pre-allocate GPU SSM prefill scratch buffer and return d_scratch
void gpu_ssm_prefill_alloc(int B, int T,
    float **d_scratch, cudaStream_t stream);

// GPU SSM prefill wrapper: upload host x, run GPU forward, download output
// State (d_ssm_state, d_conv_state) must already be on GPU.
// d_scratch must be large enough (use gpu_ssm_scratch_needed).
void gpu_ssm_prefill(cublasHandle_t cublas_h, cudaStream_t stream,
    const float *h_x, int B, int T,
    gpu_ssm_weights *gw,
    float *d_ssm_state, float *d_conv_state,
    float *d_scratch,
    float *h_output);

// ================================================================
// GPU GQA prefill wrapper: upload host x, run GPU forward, download output + KV
// ================================================================
void gpu_gqa_prefill(cublasHandle_t cublas_h, cudaStream_t stream,
    const float *h_x, int B, int T,
    gpu_gqa_weights *gw,
    const float *d_sincos,
    float *d_Q_full, float *d_K, float *d_V, float *d_scratch,
    float *h_output,
    float *h_K_out,   // [B*T, GQA_KV_DIM] host output for KV cache
    float *h_V_out);  // [B*T, GQA_KV_DIM] host output for KV cache

#ifdef __cplusplus
}
#endif /* WUBU_HAVE_CUDA */

#ifdef __cplusplus
}
#endif

#endif // WUBU_BENCH_H
