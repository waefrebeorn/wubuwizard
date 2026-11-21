#ifdef WUBU_HAVE_CUDA
#include "bench.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ================================================================
// GPU Output Projection — hidden @ output_weight^T via cuBLAS
// ================================================================
float* gpu_upload_output_weight(cublasHandle_t handle, const float *host_weight,
                                 int vocab_size, cudaStream_t stream) {
    int64_t n = (int64_t)D_MODEL * vocab_size;
    float *d_w;
    cudaMalloc((void**)&d_w, n * sizeof(float));
    cudaMemcpyAsync(d_w, host_weight, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return d_w;
}

void gpu_output_projection(cublasHandle_t handle, cudaStream_t stream,
                           const float *d_hidden, int B, int T,
                           const float *d_output_weight, int vocab_size,
                           float *d_logits) {
    // Use custom CUDA kernel — cuBLAS SGEMM fails on large vocab sizes
    launch_output_proj_kernel(stream, d_hidden, D_MODEL,
                              d_output_weight, (int64_t)vocab_size, d_logits);
}

void gpu_free_output_weight(float *d_weight) {
    if (d_weight) cudaFree(d_weight);
}

// ================================================================
// GPU SSM Layer Forward Pass
// ================================================================
void gpu_ssm_forward(cublasHandle_t cublas_h, cudaStream_t stream,
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
                     float *d_z_silu) {
    const int N = B * T;
    const int qkv_dim = KEY_DIM * 2 + VALUE_DIM; // 8192

    // DEBUG: dump GPU input just before matmul
    if (N == 1) {
        float *host_x = (float*)malloc(N * D_MODEL * sizeof(float));
        cudaMemcpy(host_x, d_x, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);
        FILE *f = fopen("/tmp/debug_gpu_input.bin", "wb"); if(f){fwrite(host_x, sizeof(float), N * D_MODEL, f); fclose(f);}
        printf("GPU input RMS=%.6f\\n", sqrtf(host_x[0]*host_x[0]/D_MODEL));
        free(host_x);
    }

    // ===== Step 1: QKV projection =====
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_qkv, qkv_dim, d_qkv, 1.0f, 0.0f);

    // DEBUG: dump qkv for L0
    if (N == 1) {
        float *host_qkv = (float*)malloc(N * qkv_dim * sizeof(float));
        cudaMemcpy(host_qkv, d_qkv, N * qkv_dim * sizeof(float), cudaMemcpyDeviceToHost);
        FILE *f = fopen("/tmp/debug_gpu_qkv.bin", "wb"); if(f){fwrite(host_qkv, sizeof(float), N * qkv_dim, f); fclose(f);}
        free(host_qkv);
    }

    // ===== Step 2: z gate projection =====
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_gate, VALUE_DIM, d_z, 1.0f, 0.0f);

    // ===== Step 3: beta/alpha projections =====
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_ssm_beta, DT_RANK, d_beta, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_ssm_alpha, DT_RANK, d_alpha, 1.0f, 0.0f);

    // ===== Step 4: beta = sigmoid(beta_raw), gate = softplus(alpha + dt_bias) * ssm_a =====
    wubu_cuda_sigmoid(N * DT_RANK, d_beta, d_beta_sig, stream);
    wubu_cuda_add_bias(N, DT_RANK, d_alpha, d_ssm_dt_bias, d_alpha_bi, stream);
    wubu_cuda_softplus(N * DT_RANK, d_alpha_bi, d_gate, stream);
    wubu_cuda_mul_by_scalar(N, DT_RANK, d_gate, d_ssm_a, d_gate, stream);

    // ===== Step 5: Convolution =====
    // First CONV_KERNEL-1 positions are conv_state, then qkv data
    for (int b = 0; b < B; b++) {
        cudaMemcpyAsync(d_conv_input + b * (T + CONV_KERNEL - 1) * CONV_DIM,
                        d_conv_state + b * (CONV_KERNEL - 1) * CONV_DIM,
                        (CONV_KERNEL - 1) * CONV_DIM * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_conv_input + b * (T + CONV_KERNEL - 1) * CONV_DIM + (CONV_KERNEL - 1) * CONV_DIM,
                        d_qkv + b * T * CONV_DIM,
                        T * CONV_DIM * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // Convolution + SiLU
    wubu_cuda_conv1d(B, T, CONV_DIM, CONV_KERNEL, d_conv_input, d_ssm_conv1d, d_conv_out, stream);
    wubu_cuda_silu(N * CONV_DIM, d_conv_out, d_conv_out, stream);

    // Update conv_state: last CONV_KERNEL-1 elements
    for (int b = 0; b < B; b++) {
        cudaMemcpyAsync(d_conv_state + b * (CONV_KERNEL - 1) * CONV_DIM,
                        d_conv_input + (b * (T + CONV_KERNEL - 1) + T) * CONV_DIM,
                        (CONV_KERNEL - 1) * CONV_DIM * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // ===== Step 6: Split Q, K, V =====
    wubu_cuda_split_qkv(N, KEY_DIM, VALUE_DIM, d_conv_out,
                        d_q_conv, d_k_conv, d_v_conv, stream);

    // ===== Step 7: L2 Normalize Q and K =====
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_q_conv, 1e-12f, d_q_norm, stream);
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_k_conv, 1e-12f, d_k_norm, stream);

    // DEBUG: dump L0 Q, K, V, beta, gate
    if (B == 1 && T == 1) {
        float *host_q = (float*)malloc(KEY_DIM * sizeof(float));
        float *host_k = (float*)malloc(KEY_DIM * sizeof(float));
        float *host_v = (float*)malloc(VALUE_DIM * sizeof(float));
        float *host_beta = (float*)malloc(DT_RANK * sizeof(float));
        float *host_gate = (float*)malloc(DT_RANK * sizeof(float));
        cudaMemcpy(host_q, d_q_norm, KEY_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_k, d_k_norm, KEY_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_v, d_v_conv, VALUE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_beta, d_beta_sig, DT_RANK * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_gate, d_gate, DT_RANK * sizeof(float), cudaMemcpyDeviceToHost);
        FILE *f = fopen("/tmp/debug_gpu_q.bin", "wb"); if(f){fwrite(host_q, sizeof(float), KEY_DIM, f); fclose(f);}
        f = fopen("/tmp/debug_gpu_k.bin", "wb"); if(f){fwrite(host_k, sizeof(float), KEY_DIM, f); fclose(f);}
        f = fopen("/tmp/debug_gpu_v.bin", "wb"); if(f){fwrite(host_v, sizeof(float), VALUE_DIM, f); fclose(f);}
        f = fopen("/tmp/debug_gpu_beta.bin", "wb"); if(f){fwrite(host_beta, sizeof(float), DT_RANK, f); fclose(f);}
        f = fopen("/tmp/debug_gpu_gate.bin", "wb"); if(f){fwrite(host_gate, sizeof(float), DT_RANK, f); fclose(f);}
        free(host_q); free(host_k); free(host_v); free(host_beta); free(host_gate);
    }

    // ===== Steps 8-9: Gated Delta Net recurrence =====
    int repeat_factor = SSM_V_HEADS / SSM_K_HEADS;
    
    // Allocate persistent channel for the step kernel (done once per layer)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int s = b * T + t;

            // Read beta and gate per-token (small, DT_RANK=32)
            float beta_host[DT_RANK], gate_host[DT_RANK];
            cudaMemcpyAsync(beta_host, d_beta_sig + s * DT_RANK,
                           DT_RANK * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(gate_host, d_gate + s * DT_RANK,
                           DT_RANK * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            for (int vh = 0; vh < SSM_V_HEADS; vh++) {
                int kh = vh % SSM_K_HEADS;
                float bg = beta_host[vh];
                float gg = gate_host[vh];

                float *d_q_vh = d_q_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                float *d_k_vh = d_k_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                float *d_v_vh = d_v_conv + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                float *d_h = d_ssm_state + (vh * SSM_D_STATE * SSM_D_STATE);
                float *d_out_vh = d_delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;

                wubu_cuda_delta_net_step(d_h, d_k_vh, d_v_vh, d_q_vh,
                                         gg, bg, d_out_vh, stream);
            }
        }
    }

    // ===== Step 10: Gated normalization =====
    wubu_cuda_silu(N * VALUE_DIM, d_z, d_z_silu, stream);
    wubu_cuda_gated_norm(B, T, SSM_V_HEADS, SSM_D_STATE,
                         d_delta_out, d_ssm_norm, d_z_silu, stream);

    // ===== Step 11: Output projection =====
    wubu_cuda_matmul(cublas_h, d_delta_out, N, VALUE_DIM, d_ssm_out, D_MODEL, d_output, 1.0f, 0.0f);

    cudaStreamSynchronize(stream);
}

// GPU SSM forward with per-timestep state trajectory saving
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
                          float *d_z_silu) {
    // Steps 1-7: IDENTICAL to gpu_ssm_forward
    const int N = B * T;
    const int qkv_dim = KEY_DIM * 2 + VALUE_DIM;
    int state_sz = SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_qkv, qkv_dim, d_qkv, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_gate, VALUE_DIM, d_z, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_ssm_beta, DT_RANK, d_beta, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_ssm_alpha, DT_RANK, d_alpha, 1.0f, 0.0f);
    
    wubu_cuda_sigmoid(N * DT_RANK, d_beta, d_beta_sig, stream);
    wubu_cuda_add_bias(N, DT_RANK, d_alpha, d_ssm_dt_bias, d_alpha_bi, stream);
    wubu_cuda_softplus(N * DT_RANK, d_alpha_bi, d_gate, stream);
    wubu_cuda_mul_by_scalar(N, DT_RANK, d_gate, d_ssm_a, d_gate, stream);
    
    for (int b = 0; b < B; b++) {
        cudaMemcpyAsync(d_conv_input + b * (T + CONV_KERNEL - 1) * CONV_DIM,
                        d_conv_state + b * (CONV_KERNEL - 1) * CONV_DIM,
                        (CONV_KERNEL - 1) * CONV_DIM * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_conv_input + b * (T + CONV_KERNEL - 1) * CONV_DIM + (CONV_KERNEL - 1) * CONV_DIM,
                        d_qkv + b * T * CONV_DIM,
                        T * CONV_DIM * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
    wubu_cuda_conv1d(B, T, CONV_DIM, CONV_KERNEL, d_conv_input, d_ssm_conv1d, d_conv_out, stream);
    wubu_cuda_silu(N * CONV_DIM, d_conv_out, d_conv_out, stream);
    
    for (int b = 0; b < B; b++) {
        cudaMemcpyAsync(d_conv_state + b * (CONV_KERNEL - 1) * CONV_DIM,
                        d_conv_input + (b * (T + CONV_KERNEL - 1) + T) * CONV_DIM,
                        (CONV_KERNEL - 1) * CONV_DIM * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
    
    wubu_cuda_split_qkv(N, KEY_DIM, VALUE_DIM, d_conv_out, d_q_conv, d_k_conv, d_v_conv, stream);
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_q_conv, 1e-12f, d_q_norm, stream);
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_k_conv, 1e-12f, d_k_norm, stream);
    
    // Step 8-9: Recurrence with trajectory saving
    int repeat_factor = SSM_V_HEADS / SSM_K_HEADS;
    
    // Save initial state if trajectory buffer provided
    if (d_states_t)
        cudaMemcpyAsync(d_states_t, d_ssm_state, state_sz * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int s = b * T + t;
            
            float beta_host[DT_RANK], gate_host[DT_RANK];
            cudaMemcpyAsync(beta_host, d_beta_sig + s * DT_RANK,
                           DT_RANK * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(gate_host, d_gate + s * DT_RANK,
                           DT_RANK * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            
            for (int vh = 0; vh < SSM_V_HEADS; vh++) {
                int kh = vh % SSM_K_HEADS;
                float bg = beta_host[vh];
                float gg = gate_host[vh];
                
                float *d_q_vh = d_q_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                float *d_k_vh = d_k_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                float *d_v_vh = d_v_conv + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                float *d_h = d_ssm_state + (vh * SSM_D_STATE * SSM_D_STATE);
                float *d_out_vh = d_delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                
                wubu_cuda_delta_net_step(d_h, d_k_vh, d_v_vh, d_q_vh,
                                         gg, bg, d_out_vh, stream);
            }
            
            // Save state after this timestep
            if (d_states_t)
                cudaMemcpyAsync(d_states_t + (b * (T+1) + t + 1) * state_sz,
                                d_ssm_state, state_sz * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream);
        }
    }
    
    // Steps 10-11: IDENTICAL to gpu_ssm_forward
    wubu_cuda_silu(N * VALUE_DIM, d_z, d_z_silu, stream);
    wubu_cuda_gated_norm(B, T, SSM_V_HEADS, SSM_D_STATE,
                         d_delta_out, d_ssm_norm, d_z_silu, stream);
    wubu_cuda_matmul(cublas_h, d_delta_out, N, VALUE_DIM, d_ssm_out, D_MODEL, d_output, 1.0f, 0.0f);
    
    cudaStreamSynchronize(stream);
}

// ================================================================
// GPU GQA Forward Pass
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
                     const float *d_sincos) {
    const int N = B * T;
    int q_dim_x2 = GQA_Q_HEADS * GQA_HEAD_DIM * 2; // 8192
    int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;       // 512

    // ===== Step 1: Q + gate fused projection =====
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_q, q_dim_x2, d_Q_full, 1.0f, 0.0f);
    // ===== Step 2: K projection =====
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_k, kv_dim, d_K, 1.0f, 0.0f);
    // ===== Step 2b: V projection =====
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_v, kv_dim, d_V, 1.0f, 0.0f);

    // ===== Steps 3-7: Fused GQA attention =====
    wubu_cuda_gqa_forward(cublas_h, stream,
        B, T,
        d_Q_full, d_K, d_V,
        d_q_norm_w, d_k_norm_w,
        d_attn_out_w,
        d_output, d_scratch, d_sincos);

    cudaStreamSynchronize(stream);
}

// ================================================================
// GPU GQA Forward Pass — Save Variant (for backward)
// ================================================================
// Runs identical forward but saves intermediates needed for wubu_gqa_backward.
// Extra save buffers (GPU, pre-allocated by caller, or NULL to skip):
//   d_Q_norm_save  [N, q_dim] — post-RMSNorm Q (before attention overwrites scratch)
//   d_K_raw_save   [N, kv_dim] — pre-RMSNorm K (before RMSNorm overwrites d_K)
//   d_K_norm_save  [N, kv_dim] — post-RMSNorm K
//   d_attn_out_save [N, q_dim] — post-gate attention output (before output proj)
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
                          float *d_K_norm_save, float *d_attn_out_save) {
    const int N = B * T;
    int q_dim_x2 = GQA_Q_HEADS * GQA_HEAD_DIM * 2;
    int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
    int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
    int head_dim = GQA_HEAD_DIM;

    // Step 1: Q + gate fused projection
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_q, q_dim_x2, d_Q_full, 1.0f, 0.0f);
    // Step 2: K projection
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_k, kv_dim, d_K, 1.0f, 0.0f);
    // Step 2b: V projection
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_v, kv_dim, d_V, 1.0f, 0.0f);
    cudaStreamSynchronize(stream);

    // Save K_raw before RMSNorm overwrites d_K (if requested)
    if (d_K_raw_save)
        cudaMemcpyAsync(d_K_raw_save, d_K, N * kv_dim * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);

    // Step 3: Copy Q from d_Q_full first half to d_scratch
    int block = 256;
    int grid_copy = (N * q_dim + block - 1) / block;
    // Use a simple cudaMemcpy for the Q portion: first q_dim floats of d_Q_full
    cudaMemcpyAsync(d_scratch, d_Q_full, N * q_dim * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream);

    // Step 4: RMSNorm Q (in-place on d_scratch)
    int grid_norm_q = (B * T * GQA_Q_HEADS + block - 1) / block;
    wubu_cuda_rms_norm(B, T, GQA_Q_HEADS * head_dim, d_scratch, d_q_norm_w, 1e-6f, d_scratch, stream);
    cudaStreamSynchronize(stream);

    // Save Q_norm before attention overwrites d_scratch (if requested)
    if (d_Q_norm_save)
        cudaMemcpyAsync(d_Q_norm_save, d_scratch, N * q_dim * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);

    // Step 5: RMSNorm K (in-place on d_K)
    int grid_norm_k = (B * T * GQA_KV_HEADS + block - 1) / block;
    wubu_cuda_rms_norm(B, T, GQA_KV_HEADS * head_dim, d_K, d_k_norm_w, 1e-6f, d_K, stream);
    cudaStreamSynchronize(stream);

    // Save K_norm (pre-RoPE) if requested — used by backward
    if (d_K_norm_save)
        cudaMemcpyAsync(d_K_norm_save, d_K, N * kv_dim * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);

    // Step 5.5: Apply RoPE to RMSNorm'd Q and K (in-place)
    float *d_sincos = NULL;
    cudaMalloc((void**)&d_sincos, T * ROTARY_DIM * sizeof(float));
    if (d_sincos) {
        wubu_cuda_precompute_rotary(T, d_sincos, stream);
        cudaStreamSynchronize(stream);
        wubu_cuda_apply_rotary_to_qk((float*)d_scratch, (float*)d_K,
            B, T, GQA_Q_HEADS, GQA_KV_HEADS, head_dim, d_sincos, stream);
        cudaFree(d_sincos);
    }

    // Step 6: Causal attention
    // d_scratch (Q_norm) is input, d_scratch output (attn_out pre-gate)
    wubu_cuda_gqa_attention_only(cublas_h, stream, B, T, d_scratch, d_K, d_V,
                                  d_scratch, GQA_Q_HEADS, GQA_KV_HEADS, head_dim);

    // Save pre-gate attention output (if requested)
    if (d_attn_out_save)
        cudaMemcpyAsync(d_attn_out_save, d_scratch, N * q_dim * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);

    // Step 7: Gate — sigmoid then multiply (in-place on d_scratch)
    wubu_cuda_gqa_gate(d_scratch, d_Q_full, N, q_dim, stream);

    // Step 8: Output projection
    wubu_cuda_matmul(cublas_h, d_scratch, N, q_dim, d_attn_out_w, D_MODEL, d_output, 1.0f, 0.0f);

    cudaStreamSynchronize(stream);
}

// ================================================================
// GPU SSM Weight Loader
// ================================================================
int gpu_load_ssm_layer(gguf_ctx *ctx, int layer_idx,
                       gpu_ssm_weights *w, cudaStream_t stream) {
    char name[256];
    int qkv_dim = KEY_DIM * 2 + VALUE_DIM; // 8192
    gguf_tensor_info *t;

    memset(w, 0, sizeof(*w));

    // Allocate GPU memory for all weights
    w->d_attn_qkv    = wubu_cuda_alloc(D_MODEL * qkv_dim * sizeof(float));
    w->d_attn_gate   = wubu_cuda_alloc(D_MODEL * VALUE_DIM * sizeof(float));
    w->d_ssm_beta    = wubu_cuda_alloc(D_MODEL * DT_RANK * sizeof(float));
    w->d_ssm_alpha   = wubu_cuda_alloc(D_MODEL * DT_RANK * sizeof(float));
    w->d_ssm_dt_bias = wubu_cuda_alloc(DT_RANK * sizeof(float));
    w->d_ssm_a       = wubu_cuda_alloc(DT_RANK * sizeof(float));
    w->d_ssm_conv1d  = wubu_cuda_alloc(CONV_KERNEL * CONV_DIM * sizeof(float));
    w->d_ssm_norm    = wubu_cuda_alloc(SSM_D_STATE * sizeof(float));
    w->d_ssm_out     = wubu_cuda_alloc(VALUE_DIM * D_MODEL * sizeof(float));

    if (!w->d_attn_qkv || !w->d_attn_gate || !w->d_ssm_beta || !w->d_ssm_alpha ||
        !w->d_ssm_dt_bias || !w->d_ssm_a || !w->d_ssm_conv1d || !w->d_ssm_norm || !w->d_ssm_out) {
        fprintf(stderr, "GPU weight allocation failed (layer %d)\n", layer_idx);
        gpu_free_ssm_weights(w);
        return 0;
    }

    // Load + upload each tensor
    float *host_buf;

    snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_ssm_weights(w); return 0; }
    host_buf = (float *)malloc(D_MODEL * qkv_dim * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, D_MODEL * qkv_dim);
    wubu_cuda_to_device(host_buf, w->d_attn_qkv, D_MODEL * qkv_dim * sizeof(float), stream);
    free(host_buf);

    snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_ssm_weights(w); return 0; }
    host_buf = (float *)malloc(D_MODEL * VALUE_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, D_MODEL * VALUE_DIM);
    wubu_cuda_to_device(host_buf, w->d_attn_gate, D_MODEL * VALUE_DIM * sizeof(float), stream);
    free(host_buf);

    snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_ssm_weights(w); return 0; }
    host_buf = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, D_MODEL * DT_RANK);
    wubu_cuda_to_device(host_buf, w->d_ssm_beta, D_MODEL * DT_RANK * sizeof(float), stream);
    free(host_buf);

    snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_ssm_weights(w); return 0; }
    host_buf = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, D_MODEL * DT_RANK);
    wubu_cuda_to_device(host_buf, w->d_ssm_alpha, D_MODEL * DT_RANK * sizeof(float), stream);
    free(host_buf);

    snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_ssm_weights(w); return 0; }
    host_buf = (float *)malloc(DT_RANK * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, DT_RANK);
    wubu_cuda_to_device(host_buf, w->d_ssm_dt_bias, DT_RANK * sizeof(float), stream);
    free(host_buf);

    snprintf(name, sizeof(name), "blk.%d.ssm_a", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_ssm_weights(w); return 0; }
    host_buf = (float *)malloc(DT_RANK * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, DT_RANK);
    wubu_cuda_to_device(host_buf, w->d_ssm_a, DT_RANK * sizeof(float), stream);
    free(host_buf);

    snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_ssm_weights(w); return 0; }
    host_buf = (float *)malloc(CONV_KERNEL * CONV_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, CONV_KERNEL * CONV_DIM);
    wubu_cuda_to_device(host_buf, w->d_ssm_conv1d, CONV_KERNEL * CONV_DIM * sizeof(float), stream);
    free(host_buf);

    snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_ssm_weights(w); return 0; }
    host_buf = (float *)malloc(SSM_D_STATE * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, SSM_D_STATE);
    wubu_cuda_to_device(host_buf, w->d_ssm_norm, SSM_D_STATE * sizeof(float), stream);
    free(host_buf);

    snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_ssm_weights(w); return 0; }
    host_buf = (float *)malloc(VALUE_DIM * D_MODEL * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, VALUE_DIM * D_MODEL);
    wubu_cuda_to_device(host_buf, w->d_ssm_out, VALUE_DIM * D_MODEL * sizeof(float), stream);
    free(host_buf);

    cudaStreamSynchronize(stream);
    return 1;
}

void gpu_free_ssm_weights(gpu_ssm_weights *w) {
    wubu_cuda_free(w->d_attn_qkv);
    wubu_cuda_free(w->d_attn_gate);
    wubu_cuda_free(w->d_ssm_beta);
    wubu_cuda_free(w->d_ssm_alpha);
    wubu_cuda_free(w->d_ssm_dt_bias);
    wubu_cuda_free(w->d_ssm_a);
    wubu_cuda_free(w->d_ssm_conv1d);
    wubu_cuda_free(w->d_ssm_norm);
    wubu_cuda_free(w->d_ssm_out);
}

// ================================================================
// GPU GQA Weight Loader
// ================================================================
int gpu_load_gqa_layer(gguf_ctx *ctx, int layer_idx,
                       gpu_gqa_weights *w, cudaStream_t stream) {
    char name[256];
    int q_dim_x2 = GQA_Q_HEADS * GQA_HEAD_DIM * 2; // 8192
    int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;       // 512
    int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;          // 4096
    gguf_tensor_info *t;

    memset(w, 0, sizeof(*w));

    w->d_attn_q     = wubu_cuda_alloc(D_MODEL * q_dim_x2 * sizeof(float));
    w->d_attn_k     = wubu_cuda_alloc(D_MODEL * kv_dim * sizeof(float));
    w->d_attn_v     = wubu_cuda_alloc(D_MODEL * kv_dim * sizeof(float));
    w->d_attn_out_w = wubu_cuda_alloc(q_dim * D_MODEL * sizeof(float));
    w->d_q_norm_w   = wubu_cuda_alloc(GQA_HEAD_DIM * sizeof(float));
    w->d_k_norm_w   = wubu_cuda_alloc(GQA_HEAD_DIM * sizeof(float));

    if (!w->d_attn_q || !w->d_attn_k || !w->d_attn_v || !w->d_attn_out_w || !w->d_q_norm_w || !w->d_k_norm_w) {
        fprintf(stderr, "GQA GPU weight allocation failed (layer %d)\n", layer_idx);
        gpu_free_gqa_weights(w);
        return 0;
    }

    float *host_buf;

    snprintf(name, sizeof(name), "blk.%d.attn_q.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_gqa_weights(w); return 0; }
    host_buf = (float *)malloc(D_MODEL * q_dim_x2 * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, D_MODEL * q_dim_x2);
    wubu_cuda_to_device(host_buf, w->d_attn_q, D_MODEL * q_dim_x2 * sizeof(float), stream);
    free(host_buf);

    snprintf(name, sizeof(name), "blk.%d.attn_k.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_gqa_weights(w); return 0; }
    host_buf = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, D_MODEL * kv_dim);
    wubu_cuda_to_device(host_buf, w->d_attn_k, D_MODEL * kv_dim * sizeof(float), stream);
    free(host_buf);

    snprintf(name, sizeof(name), "blk.%d.attn_v.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_gqa_weights(w); return 0; }
    host_buf = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, D_MODEL * kv_dim);
    wubu_cuda_to_device(host_buf, w->d_attn_v, D_MODEL * kv_dim * sizeof(float), stream);
    free(host_buf);

    snprintf(name, sizeof(name), "blk.%d.attn_output.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_gqa_weights(w); return 0; }
    host_buf = (float *)malloc(q_dim * D_MODEL * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, q_dim * D_MODEL);
    wubu_cuda_to_device(host_buf, w->d_attn_out_w, q_dim * D_MODEL * sizeof(float), stream);
    free(host_buf);

    snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_gqa_weights(w); return 0; }
    host_buf = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, GQA_HEAD_DIM);
    wubu_cuda_to_device(host_buf, w->d_q_norm_w, GQA_HEAD_DIM * sizeof(float), stream);
    free(host_buf);

    snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Missing %s\n", name); gpu_free_gqa_weights(w); return 0; }
    host_buf = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, t, host_buf, GQA_HEAD_DIM);
    wubu_cuda_to_device(host_buf, w->d_k_norm_w, GQA_HEAD_DIM * sizeof(float), stream);
    free(host_buf);

    cudaStreamSynchronize(stream);
    return 1;
}

void gpu_free_gqa_weights(gpu_gqa_weights *w) {
    wubu_cuda_free(w->d_attn_q);
    wubu_cuda_free(w->d_attn_k);
    wubu_cuda_free(w->d_attn_v);
    wubu_cuda_free(w->d_attn_out_w);
    wubu_cuda_free(w->d_q_norm_w);
    wubu_cuda_free(w->d_k_norm_w);
}

// ================================================================
// GPU Poincaré SSM Layer Forward Pass
// Identical to gpu_ssm_forward except Step 9 uses Poincaré recurrence.
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
                     float R) {
    const int N = B * T;
    const int qkv_dim = KEY_DIM * 2 + VALUE_DIM; // 8192

    // ===== Steps 1-4: IDENTICAL to Euclidean =====
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_qkv, qkv_dim, d_qkv, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_gate, VALUE_DIM, d_z, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_ssm_beta, DT_RANK, d_beta, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_ssm_alpha, DT_RANK, d_alpha, 1.0f, 0.0f);

    wubu_cuda_sigmoid(N * DT_RANK, d_beta, d_beta_sig, stream);
    wubu_cuda_add_bias(N, DT_RANK, d_alpha, d_ssm_dt_bias, d_alpha_bi, stream);
    wubu_cuda_softplus(N * DT_RANK, d_alpha_bi, d_gate, stream);
    wubu_cuda_mul_by_scalar(N, DT_RANK, d_gate, d_ssm_a, d_gate, stream);

    // ===== Steps 5-8: IDENTICAL to Euclidean =====
    for (int b = 0; b < B; b++) {
        cudaMemcpyAsync(d_conv_input + b * (T + CONV_KERNEL - 1) * CONV_DIM,
                        d_conv_state + b * (CONV_KERNEL - 1) * CONV_DIM,
                        (CONV_KERNEL - 1) * CONV_DIM * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_conv_input + b * (T + CONV_KERNEL - 1) * CONV_DIM + (CONV_KERNEL - 1) * CONV_DIM,
                        d_qkv + b * T * CONV_DIM,
                        T * CONV_DIM * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
    wubu_cuda_conv1d(B, T, CONV_DIM, CONV_KERNEL, d_conv_input, d_ssm_conv1d, d_conv_out, stream);
    wubu_cuda_silu(N * CONV_DIM, d_conv_out, d_conv_out, stream);

    for (int b = 0; b < B; b++) {
        cudaMemcpyAsync(d_conv_state + b * (CONV_KERNEL - 1) * CONV_DIM,
                        d_conv_input + (b * (T + CONV_KERNEL - 1) + T) * CONV_DIM,
                        (CONV_KERNEL - 1) * CONV_DIM * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    wubu_cuda_split_qkv(N, KEY_DIM, VALUE_DIM, d_conv_out,
                        d_q_conv, d_k_conv, d_v_conv, stream);
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_q_conv, 1e-12f, d_q_norm, stream);
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_k_conv, 1e-12f, d_k_norm, stream);

    // ===== Step 9: POINCARÉ RECURRENCE (replaces Euclidean) =====
    wubu_cuda_poincare_recurrence(cublas_h, stream, B, T, R,
        d_ssm_state, d_q_norm, d_k_norm, d_v_conv,
        d_gate, d_beta_sig, d_delta_out, NULL);

    // ===== Steps 10-11: IDENTICAL to Euclidean =====
    wubu_cuda_silu(N * VALUE_DIM, d_z, d_z_silu, stream);
    wubu_cuda_gated_norm(B, T, SSM_V_HEADS, SSM_D_STATE,
                         d_delta_out, d_ssm_norm, d_z_silu, stream);
    wubu_cuda_matmul(cublas_h, d_delta_out, N, VALUE_DIM, d_ssm_out, D_MODEL, d_output, 1.0f, 0.0f);

    cudaStreamSynchronize(stream);
}

// ================================================================
// GPU Poincaré SSM forward — save variant for backward
// Saves state trajectory h_t for all timesteps into d_states_t
// d_states_t: [B, T+1, SSM_V_HEADS, D_STATE, D_STATE]
// ================================================================
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
                     float *d_states_t) {
    const int N = B * T;
    const int qkv_dim = KEY_DIM * 2 + VALUE_DIM;

    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_qkv, qkv_dim, d_qkv, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_gate, VALUE_DIM, d_z, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_ssm_beta, DT_RANK, d_beta, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_ssm_alpha, DT_RANK, d_alpha, 1.0f, 0.0f);
    wubu_cuda_sigmoid(N * DT_RANK, d_beta, d_beta_sig, stream);
    wubu_cuda_add_bias(N, DT_RANK, d_alpha, d_ssm_dt_bias, d_alpha_bi, stream);
    wubu_cuda_softplus(N * DT_RANK, d_alpha_bi, d_gate, stream);
    wubu_cuda_mul_by_scalar(N, DT_RANK, d_gate, d_ssm_a, d_gate, stream);
    for (int b = 0; b < B; b++) {
        cudaMemcpyAsync(d_conv_input + b * (T + CONV_KERNEL - 1) * CONV_DIM,
                        d_conv_state + b * (CONV_KERNEL - 1) * CONV_DIM,
                        (CONV_KERNEL - 1) * CONV_DIM * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_conv_input + b * (T + CONV_KERNEL - 1) * CONV_DIM + (CONV_KERNEL - 1) * CONV_DIM,
                        d_qkv + b * T * CONV_DIM,
                        T * CONV_DIM * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
    wubu_cuda_conv1d(B, T, CONV_DIM, CONV_KERNEL, d_conv_input, d_ssm_conv1d, d_conv_out, stream);
    wubu_cuda_silu(N * CONV_DIM, d_conv_out, d_conv_out, stream);
    for (int b = 0; b < B; b++) {
        cudaMemcpyAsync(d_conv_state + b * (CONV_KERNEL - 1) * CONV_DIM,
                        d_conv_input + (b * (T + CONV_KERNEL - 1) + T) * CONV_DIM,
                        (CONV_KERNEL - 1) * CONV_DIM * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
    wubu_cuda_split_qkv(N, KEY_DIM, VALUE_DIM, d_conv_out,
                        d_q_conv, d_k_conv, d_v_conv, stream);
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_q_conv, 1e-12f, d_q_norm, stream);
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_k_conv, 1e-12f, d_k_norm, stream);

    // Save initial state to trajectory[0]
    int state_sz = SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    cudaMemcpyAsync(d_states_t, d_ssm_state, state_sz * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // Step 9: Poincaré recurrence WITH trajectory save
    wubu_cuda_poincare_recurrence(cublas_h, stream, B, T, R,
        d_ssm_state, d_q_norm, d_k_norm, d_v_conv,
        d_gate, d_beta_sig, d_delta_out, d_states_t);

    // Steps 10-11
    wubu_cuda_silu(N * VALUE_DIM, d_z, d_z_silu, stream);
    wubu_cuda_gated_norm(B, T, SSM_V_HEADS, SSM_D_STATE,
                         d_delta_out, d_ssm_norm, d_z_silu, stream);
    wubu_cuda_matmul(cublas_h, d_delta_out, N, VALUE_DIM, d_ssm_out, D_MODEL, d_output, 1.0f, 0.0f);
    cudaStreamSynchronize(stream);
}

// ================================================================
// GPU SSM scratch buffer sizing and allocation
// ================================================================
size_t gpu_ssm_scratch_needed(int B, int T) {
    int N = B * T;
    const int qkv_dim = KEY_DIM * 2 + VALUE_DIM;  // 8192
    size_t total = 0;
    total += (size_t)N * D_MODEL;                 // d_x (input)
    total += (size_t)N * qkv_dim;                 // d_qkv
    total += (size_t)N * VALUE_DIM;          // d_z
    total += (size_t)N * DT_RANK;            // d_beta
    total += (size_t)N * DT_RANK;            // d_alpha
    total += (size_t)N * DT_RANK;            // d_beta_sig
    total += (size_t)N * DT_RANK;            // d_alpha_bi
    total += (size_t)N * DT_RANK;            // d_gate
    total += (size_t)B * (T + CONV_KERNEL - 1) * CONV_DIM; // d_conv_input
    total += (size_t)N * CONV_DIM;           // d_conv_out
    total += (size_t)N * KEY_DIM;            // d_q_conv
    total += (size_t)N * KEY_DIM;            // d_k_conv
    total += (size_t)N * VALUE_DIM;          // d_v_conv
    total += (size_t)N * KEY_DIM;            // d_q_norm
    total += (size_t)N * KEY_DIM;            // d_k_norm
    total += (size_t)N * VALUE_DIM;          // d_delta_out
    total += (size_t)N * VALUE_DIM;          // d_z_silu
    return total;
}

void gpu_ssm_prefill_alloc(int B, int T,
    float **d_scratch, cudaStream_t stream) {
    size_t nf = gpu_ssm_scratch_needed(B, T);
    cudaMalloc((void**)d_scratch, nf * sizeof(float));
}

// ================================================================
// GPU SSM prefill wrapper — upload, run, download
// ================================================================
void gpu_ssm_prefill(cublasHandle_t cublas_h, cudaStream_t stream,
    const float *h_x, int B, int T,
    gpu_ssm_weights *gw,
    float *d_ssm_state, float *d_conv_state,
    float *d_scratch,
    float *h_output)
{
    int N = B * T;
    const int qkv_dim = KEY_DIM * 2 + VALUE_DIM;  // 8192

    // Pointers into scratch buffer
    float *d_x          = d_scratch;                          // input [N, D_MODEL]
    float *d_qkv        = d_scratch + (size_t)N * D_MODEL;    // matmul out [N, qkv_dim]
    float *d_z          = d_qkv     + (size_t)N * qkv_dim;    // z gate [N, vdim]
    float *d_beta      = d_z       + (size_t)N * VALUE_DIM;
    float *d_alpha     = d_beta    + (size_t)N * DT_RANK;
    float *d_beta_sig  = d_alpha   + (size_t)N * DT_RANK;
    float *d_alpha_bi  = d_beta_sig + (size_t)N * DT_RANK;
    float *d_gate      = d_alpha_bi + (size_t)N * DT_RANK;
    float *d_conv_input = d_gate    + (size_t)N * DT_RANK;
    int conv_input_stride = T + CONV_KERNEL - 1;
    float *d_conv_out  = d_conv_input + (size_t)B * conv_input_stride * CONV_DIM;
    float *d_q_conv    = d_conv_out   + (size_t)N * CONV_DIM;
    float *d_k_conv    = d_q_conv     + (size_t)N * KEY_DIM;
    float *d_v_conv    = d_k_conv     + (size_t)N * KEY_DIM;
    float *d_q_norm    = d_v_conv     + (size_t)N * VALUE_DIM;
    float *d_k_norm    = d_q_norm     + (size_t)N * KEY_DIM;
    float *d_delta_out = d_k_norm     + (size_t)N * KEY_DIM;
    float *d_z_silu    = d_delta_out  + (size_t)N * VALUE_DIM;

    // Upload input to dedicated input buffer (separate from d_qkv to avoid aliasing)
    cudaMemcpyAsync(d_x, h_x, (size_t)N * D_MODEL * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Call GPU SSM forward
    gpu_ssm_forward(cublas_h, stream,
                    d_x, B, T,
                    gw->d_attn_qkv, gw->d_attn_gate,
                    gw->d_ssm_beta, gw->d_ssm_alpha,
                    gw->d_ssm_dt_bias, gw->d_ssm_a,
                    gw->d_ssm_conv1d, gw->d_ssm_norm, gw->d_ssm_out,
                    d_ssm_state, d_conv_state,
                    d_scratch,  // reuse d_scratch as output (last step overwrites)
                    d_qkv, d_z, d_beta, d_alpha,
                    d_beta_sig, d_alpha_bi, d_gate,
                    d_conv_input, d_conv_out,
                    d_q_conv, d_k_conv, d_v_conv,
                    d_q_norm, d_k_norm,
                    d_delta_out, d_z_silu);

    // Download output (result is in d_scratch after gpu_ssm_forward writes d_output there)
    cudaMemcpyAsync(h_output, d_scratch, (size_t)N * D_MODEL * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

// ================================================================
// GPU SSM decode step (1 token, state carry on GPU)
// ================================================================
void gpu_ssm_decode_step(cublasHandle_t cublas_h, cudaStream_t stream,
    const float *d_x_step,
    gpu_ssm_weights *gw,
    float *d_ssm_state, float *d_conv_state,
    float *d_output,
    float *d_scratch_buf)
{
    const int B = 1, T = 1;
    const int N = 1;

    // All scratch pointers computed from d_scratch_buf (must be large enough for B=1,T=1)
    const int qkv_dim = KEY_DIM * 2 + VALUE_DIM;
    float *d_qkv       = d_scratch_buf;
    float *d_z         = d_scratch_buf + (size_t)N * qkv_dim;
    float *d_beta      = d_z           + (size_t)N * VALUE_DIM;
    float *d_alpha     = d_beta        + (size_t)N * DT_RANK;
    float *d_beta_sig  = d_alpha       + (size_t)N * DT_RANK;
    float *d_alpha_bi  = d_beta_sig    + (size_t)N * DT_RANK;
    float *d_gate      = d_alpha_bi    + (size_t)N * DT_RANK;
    float *d_conv_input = d_gate       + (size_t)N * DT_RANK;
    int conv_input_stride = T + CONV_KERNEL - 1;
    float *d_conv_out  = d_conv_input  + (size_t)B * conv_input_stride * CONV_DIM;
    float *d_q_conv    = d_conv_out    + (size_t)N * CONV_DIM;
    float *d_k_conv    = d_q_conv      + (size_t)N * KEY_DIM;
    float *d_v_conv    = d_k_conv      + (size_t)N * KEY_DIM;
    float *d_q_norm    = d_v_conv      + (size_t)N * VALUE_DIM;
    float *d_k_norm    = d_q_norm      + (size_t)N * KEY_DIM;
    float *d_delta_out = d_k_norm      + (size_t)N * KEY_DIM;
    float *d_z_silu    = d_delta_out   + (size_t)N * VALUE_DIM;

    // Run full SSM forward on GPU (1 token)
    gpu_ssm_forward(cublas_h, stream,
                    d_x_step, B, T,
                    gw->d_attn_qkv, gw->d_attn_gate,
                    gw->d_ssm_beta, gw->d_ssm_alpha,
                    gw->d_ssm_dt_bias, gw->d_ssm_a,
                    gw->d_ssm_conv1d, gw->d_ssm_norm, gw->d_ssm_out,
                    d_ssm_state, d_conv_state,
                    d_output,  // output goes here
                    d_qkv, d_z, d_beta, d_alpha,
                    d_beta_sig, d_alpha_bi, d_gate,
                    d_conv_input, d_conv_out,
                    d_q_conv, d_k_conv, d_v_conv,
                    d_q_norm, d_k_norm,
                    d_delta_out, d_z_silu);
    // d_output now contains the 1-token result on GPU
}

// ================================================================
// GPU GQA decode step — projections on GPU, downloads for CPU KV cache + attention
// ================================================================
void gpu_gqa_decode_step(cublasHandle_t cublas_h, cudaStream_t stream,
    const float *d_x_step,
    gpu_gqa_weights *gw,
    float *d_Q_full,
    float *d_K, float *d_V,
    float *d_attn_out,
    float *h_Q_out,
    float *h_K_out,
    float *h_V_out)
{
    const int N = 1;
    int q_dim_x2 = GQA_Q_HEADS * GQA_HEAD_DIM * 2; // 8192
    int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;       // 512

    // Step 1: Q + gate fused projection
    wubu_cuda_matmul(cublas_h, d_x_step, N, D_MODEL,
                     gw->d_attn_q, q_dim_x2, d_Q_full, 1.0f, 0.0f);
    // Step 2: K projection
    wubu_cuda_matmul(cublas_h, d_x_step, N, D_MODEL,
                     gw->d_attn_k, kv_dim, d_K, 1.0f, 0.0f);
    // Step 2b: V projection
    wubu_cuda_matmul(cublas_h, d_x_step, N, D_MODEL,
                     gw->d_attn_v, kv_dim, d_V, 1.0f, 0.0f);
    cudaStreamSynchronize(stream);

    // Download all results to host
    cudaMemcpyAsync(h_Q_out, d_Q_full, q_dim_x2 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_K_out, d_K, kv_dim * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_V_out, d_V, kv_dim * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

// ================================================================
// GPU GQA output projection for decode step (1 token)
// ================================================================
void gpu_gqa_output_proj(cublasHandle_t cublas_h, cudaStream_t stream,
    const float *d_attn_out,
    gpu_gqa_weights *gw,
    float *d_output)
{
    const int N = 1;
    int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM; // 4096
    wubu_cuda_matmul(cublas_h, d_attn_out, N, q_dim,
                     gw->d_attn_out_w, D_MODEL, d_output, 1.0f, 0.0f);
    cudaStreamSynchronize(stream);
}

// ================================================================
// GPU GQA prefill — full forward + KV cache extraction
// ================================================================
void gpu_gqa_prefill(cublasHandle_t cublas_h, cudaStream_t stream,
    const float *h_x, int B, int T,
    gpu_gqa_weights *gw,
    const float *d_sincos,
    float *d_Q_full, float *d_K, float *d_V, float *d_scratch,
    float *h_output,
    float *h_K_out,
    float *h_V_out)
{
    const int N = B * T;
    int q_dim_x2 = GQA_Q_HEADS * GQA_HEAD_DIM * 2; // 8192
    int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;          // 4096
    int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;        // 512

    // Upload input
    float *d_x = d_scratch; // reuse scratch head for input
    cudaMemcpyAsync(d_x, h_x, (size_t)N * D_MODEL * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Step 1: Q + gate fused projection
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL,
                     gw->d_attn_q, q_dim_x2, d_Q_full, 1.0f, 0.0f);
    // Step 2: K projection
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL,
                     gw->d_attn_k, kv_dim, d_K, 1.0f, 0.0f);
    // Step 2b: V projection
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL,
                     gw->d_attn_v, kv_dim, d_V, 1.0f, 0.0f);

    // Download K and V for KV cache (before RMSNorm overwrites d_K)
    cudaMemcpyAsync(h_K_out, d_K, (size_t)N * kv_dim * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_V_out, d_V, (size_t)N * kv_dim * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    // Step 3-7: Fused GQA attention (RMSNorm Q/K, RoPE, attention, gate, output proj)
    wubu_cuda_gqa_forward(cublas_h, stream,
        B, T,
        d_Q_full, d_K, d_V,
        gw->d_q_norm_w, gw->d_k_norm_w,
        gw->d_attn_out_w,
        d_scratch, d_Q_full, d_sincos);
    // d_scratch now contains the output [N, D_MODEL]

    cudaStreamSynchronize(stream);

    // Download output
    cudaMemcpyAsync(h_output, d_scratch, (size_t)N * D_MODEL * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

#endif /* WUBU_HAVE_CUDA */
