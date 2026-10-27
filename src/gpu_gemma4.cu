/* gpu_gemma4.cu — CUDA kernels for Gemma 4 12B ISWA inference.
 * Supports sm_89 (RTX 4050/3050) and sm_120 (RTX 5050).
 * Q4_K weights uploaded at init (fits in 8GB VRAM).
 * Dequant + cuBLAS SGEMM for matmuls. Attention on GPU.
 */

#include "gpu_gemma4.h"
#include "wubu_gemma4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* ============= Sliding window attention (GPU) ============= */

__global__ void sliding_attn_kernel(const float *__restrict__ q,
                                     const float *__restrict__ k_cache,
                                     const float *__restrict__ v_cache,
                                     float *__restrict__ out,
                                     int N_tokens, int n_heads, int n_kv_heads,
                                     int d_head, int window,
                                     const int *__restrict__ positions,
                                     int kv_size) {
    extern __shared__ float s_scores[];
    int tok = blockIdx.x;
    int hq = blockIdx.y;
    int kv_h = hq % n_kv_heads;
    int tid = threadIdx.x;

    int q_stride = n_heads * d_head;
    int kv_stride = n_kv_heads * d_head;
    int pos = positions[tok];
    int kv_end = kv_size - 1;
    if (kv_end > pos) kv_end = pos;

    int kv_start = (pos > window) ? (pos - window) : 0;
    int kv_len = kv_end - kv_start + 1;
    if (kv_len <= 0) {
        for (int d = tid; d < d_head; d += blockDim.x)
            out[tok * q_stride + hq * d_head + d] = 0.0f;
        return;
    }

    const float *q_vec = q + tok * q_stride + hq * d_head;
    float *scores = s_scores;

    /* Scores */
    float scale = rsqrtf((float)d_head);
    float max_s = -1e30f;
    for (int kp = kv_start + tid; kp <= kv_end; kp += blockDim.x) {
        const float *kv = k_cache + kp * kv_stride + kv_h * d_head;
        float s = 0.0f;
        for (int d = 0; d < d_head; d++)
            s += q_vec[d] * kv[d];
        s *= scale;
        scores[kp - kv_start] = s;
        if (s > max_s) max_s = s;
    }
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) s_scores[tid] = fmaxf(s_scores[tid], s_scores[tid + offset]);
        __syncthreads();
    }
    __syncthreads();
    max_s = s_scores[0];

    /* Softmax */
    float sum_e = 0.0f;
    for (int kp = kv_start + tid; kp <= kv_end; kp += blockDim.x) {
        float e = __expf(scores[kp - kv_start] - max_s);
        scores[kp - kv_start] = e;
        sum_e += e;
    }
    s_scores[tid] = sum_e;
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) s_scores[tid] += s_scores[tid + offset];
        __syncthreads();
    }
    __syncthreads();
    sum_e = s_scores[0];
    float inv_sum = 1.0f / (sum_e + 1e-10f);

    /* V sum */
    for (int d = tid; d < d_head; d += blockDim.x) {
        float val = 0.0f;
        for (int kp = kv_start; kp <= kv_end; kp++)
            val += scores[kp - kv_start] * inv_sum *
                   (v_cache + kp * kv_stride + kv_h * d_head)[d];
        out[tok * q_stride + hq * d_head + d] = val;
    }
}

void g4_gpu_sliding_attn(g4_gpu_ctx_t *ctx,
    const float *d_q, const float *d_k, const float *d_v,
    int N_tokens, int n_heads, int n_kv_heads, int d_head,
    int window, int *d_kv_size,
    float *d_k_cache, float *d_v_cache,
    const int *d_positions, const int *h_kv_src,
    float *d_out)
{
    (void)d_k; (void)d_v; (void)h_kv_src;
    int kv_size = 0;
    cudaMemcpyAsync(&kv_size, d_kv_size, sizeof(int), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    dim3 grid((unsigned int)N_tokens, (unsigned int)n_heads);
    size_t shared_mem = (size_t)(window > 0 ? window : 1024) * sizeof(float);
    int threads = d_head < 128 ? d_head : 128;

    sliding_attn_kernel<<<grid, threads, shared_mem, ctx->stream>>>(
        d_q, d_k_cache, d_v_cache, d_out,
        N_tokens, n_heads, n_kv_heads, d_head,
        window, d_positions, kv_size);
}

/* ============= Full attention (GPU) ============= */

__global__ void full_attn_kernel(const float *__restrict__ q,
                                  const float *__restrict__ k_cache,
                                  const float *__restrict__ v_cache,
                                  float *__restrict__ out,
                                  int N_tokens, int n_heads, int n_kv_heads,
                                  int d_head,
                                  const int *__restrict__ positions,
                                  int kv_size) {
    extern __shared__ float s_scores[];
    int tok = blockIdx.x;
    int hq = blockIdx.y;
    int kv_h = hq % n_kv_heads;
    int tid = threadIdx.x;
    int q_stride = n_heads * d_head;
    int kv_stride = n_kv_heads * d_head;
    int pos = positions[tok];
    int kv_end = kv_size - 1;
    if (kv_end > pos) kv_end = pos;
    int kv_len = kv_end + 1;
    if (kv_len <= 0) {
        for (int d = tid; d < d_head; d += blockDim.x)
            out[tok * q_stride + hq * d_head + d] = 0.0f;
        return;
    }
    const float *q_vec = q + tok * q_stride + hq * d_head;
    float *scores = s_scores;
    float scale = rsqrtf((float)d_head);
    float max_s = -1e30f;
    for (int kp = tid; kp <= kv_end; kp += blockDim.x) {
        const float *kv = k_cache + kp * kv_stride + kv_h * d_head;
        float s = 0.0f;
        for (int d = 0; d < d_head; d++) s += q_vec[d] * kv[d];
        s *= scale;
        scores[kp] = s;
        if (s > max_s) max_s = s;
    }
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) s_scores[tid] = fmaxf(s_scores[tid], s_scores[tid + offset]);
        __syncthreads();
    }
    __syncthreads();
    max_s = s_scores[0];
    float sum_e = 0.0f;
    for (int kp = tid; kp <= kv_end; kp += blockDim.x) {
        float e = __expf(scores[kp] - max_s);
        scores[kp] = e;
        sum_e += e;
    }
    s_scores[tid] = sum_e;
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) s_scores[tid] += s_scores[tid + offset];
        __syncthreads();
    }
    __syncthreads();
    sum_e = s_scores[0];
    float inv_sum = 1.0f / (sum_e + 1e-10f);
    for (int d = tid; d < d_head; d += blockDim.x) {
        float val = 0.0f;
        for (int kp = 0; kp <= kv_end; kp++)
            val += scores[kp] * inv_sum *
                   (v_cache + kp * kv_stride + kv_h * d_head)[d];
        out[tok * q_stride + hq * d_head + d] = val;
    }
}

void g4_gpu_full_attn(g4_gpu_ctx_t *ctx,
    const float *d_q,
    int N_tokens, int n_heads, int n_kv_heads, int d_head,
    float *d_k_cache, float *d_v_cache,
    const int *d_positions, int current_size,
    float *d_out)
{
    dim3 grid((unsigned int)N_tokens, (unsigned int)n_heads);
    size_t shared_mem = (size_t)(current_size > 0 ? current_size : 1024) * sizeof(float);
    int threads = d_head < 128 ? d_head : 128;
    full_attn_kernel<<<grid, threads, shared_mem, ctx->stream>>>(
        d_q, d_k_cache, d_v_cache, d_out,
        N_tokens, n_heads, n_kv_heads, d_head,
        d_positions, current_size);
}

/* ============= Element-wise kernels ============= */

void g4_gpu_rms_norm(g4_gpu_ctx_t *ctx,
    const float *d_x, const float *d_w, int n, int N_tokens, float *d_y)
{
    // Use the shared rms_norm_kernel from cuda_kernels.cu
    // Extern declaration avoids duplicate definition
    extern __global__ void rms_norm_kernel(const float *, const float *, float *, int, int, float);
    int block = n < 256 ? n : 256;
    rms_norm_kernel<<<N_tokens, block, 0, ctx->stream>>>(d_x, d_w, d_y, n, N_tokens, G4_RMS_EPS);
}

__global__ void gelu_kernel(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float x3 = v * v * v;
    float inner = 0.7978845608f * (v + 0.044715f * x3);
    x[i] = 0.5f * v * (1.0f + tanhf(inner));
}

void g4_gpu_gelu(g4_gpu_ctx_t *ctx, float *d_x, int n_elems) {
    int block = 256;
    int grid = (n_elems + block - 1) / block;
    gelu_kernel<<<grid, block, 0, ctx->stream>>>(d_x, n_elems);
}

__global__ void mul_kernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

__global__ void add_kernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void scale_kernel(float *x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void softcap_kernel(float *x, int n, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float scaled = x[i] / cap; x[i] = tanhf(scaled) * cap; }
}

void g4_gpu_mul(g4_gpu_ctx_t *ctx, const float *d_a, const float *d_b, float *d_c, int n_elems) {
    int block = 256, grid = (n_elems + block - 1) / block;
    mul_kernel<<<grid, block, 0, ctx->stream>>>(d_a, d_b, d_c, n_elems);
}
void g4_gpu_add(g4_gpu_ctx_t *ctx, const float *d_a, const float *d_b, float *d_c, int n_elems) {
    int block = 256, grid = (n_elems + block - 1) / block;
    add_kernel<<<grid, block, 0, ctx->stream>>>(d_a, d_b, d_c, n_elems);
}
void g4_gpu_scale(g4_gpu_ctx_t *ctx, float *d_x, float s, int n_elems) {
    int block = 256, grid = (n_elems + block - 1) / block;
    scale_kernel<<<grid, block, 0, ctx->stream>>>(d_x, s, n_elems);
}
void g4_gpu_softcap(g4_gpu_ctx_t *ctx, float *d_x, int n_elems, float cap) {
    int block = 256, grid = (n_elems + block - 1) / block;
    softcap_kernel<<<grid, block, 0, ctx->stream>>>(d_x, n_elems, cap);
}

__global__ void rope_kernel(float *__restrict__ q, float *__restrict__ k, int N_tokens,
                            int d_head, int q_stride, const int *__restrict__ pos,
                            int n_rot, float rope_base, const float *__restrict__ freqs,
                            int is_full) {
    int tok = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x * 2;
    if (i >= n_rot) return;
    int p = pos[tok];
    float freq_val = (is_full && freqs && i/2 < d_head/8) ? freqs[i/2] : 1.0f;
    float theta = (float)p / powf(rope_base * freq_val, (float)i / (float)d_head);
    float cos_t = __cosf(theta);
    float sin_t = __sinf(theta);
    int base = tok * q_stride + h * d_head;
    float q0 = q[base + i];
    float q1 = q[base + i + 1];
    q[base + i]     = q0 * cos_t - q1 * sin_t;
    q[base + i + 1] = q0 * sin_t + q1 * cos_t;
    if (k) {
        float k0 = k[base + i];
        float k1 = k[base + i + 1];
        k[base + i]     = k0 * cos_t - k1 * sin_t;
        k[base + i + 1] = k0 * sin_t + k1 * cos_t;
    }
}

void g4_gpu_rope(g4_gpu_ctx_t *ctx,
    float *d_q, float *d_k, int N_tokens, int d_head, int q_stride,
    const int *d_positions, int n_rot, float rope_base,
    const float *h_rope_freqs, int is_full)
{
    float *d_freqs = NULL;
    if (is_full && h_rope_freqs) {
        cudaMallocAsync(&d_freqs, (size_t)(d_head / 2) * sizeof(float), ctx->stream);
        cudaMemcpyAsync(d_freqs, h_rope_freqs, (size_t)(d_head / 2) * sizeof(float),
                       cudaMemcpyHostToDevice, ctx->stream);
    }
    dim3 grid((unsigned int)N_tokens, (unsigned int)(q_stride / d_head));
    int block = d_head / 2;
    if (block > 128) block = 128;
    rope_kernel<<<grid, block, 0, ctx->stream>>>(d_q, d_k, N_tokens, d_head, q_stride,
                                                  d_positions, n_rot, rope_base, d_freqs, is_full);
    if (d_freqs) cudaFreeAsync(d_freqs, ctx->stream);
}

/* ============= cuBLAS SGEMM ============= */

void g4_gpu_sgemm(g4_gpu_ctx_t *ctx, int M, int N, int K,
    const float *d_A, const float *d_B, float *d_C,
    bool transA, bool transB)
{
    float alpha = 1.0f, beta = 0.0f;
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    cublasSgemm(ctx->cublas, opA, opB, M, N, K, &alpha, d_A, lda, d_B, ldb, &beta, d_C, M);
}

/* ============= Memory ops ============= */

void g4_gpu_h2d(g4_gpu_ctx_t *ctx, const void *host, void *dev, size_t bytes) {
    cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, ctx->stream);
}

void g4_gpu_d2h(g4_gpu_ctx_t *ctx, const void *dev, void *host, size_t bytes) {
    cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, ctx->stream);
}

void g4_gpu_sync(g4_gpu_ctx_t *ctx) {
    cudaStreamSynchronize(ctx->stream);
}

float* g4_gpu_upload_norms(g4_gpu_ctx_t *ctx, const float *h_norms, int n) {
    float *d_norms = NULL;
    cudaMallocAsync(&d_norms, (size_t)n * sizeof(float), ctx->stream);
    cudaMemcpyAsync(d_norms, h_norms, (size_t)n * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    return d_norms;
}

/* ============= Init / Destroy ============= */

g4_gpu_ctx_t* g4_gpu_init(int max_tokens, int max_ctx, int n_layers) {
    g4_gpu_ctx_t *ctx = (g4_gpu_ctx_t *)calloc(1, sizeof(g4_gpu_ctx_t));
    if (!ctx) return NULL;
    cudaSetDevice(0);
    cublasCreate(&ctx->cublas);
    cudaStreamCreate(&ctx->stream);
    cublasSetStream(ctx->cublas, ctx->stream);
    ctx->max_tokens = max_tokens;
    ctx->max_ctx = max_ctx;
    ctx->n_layers = n_layers;

    int max_kv_dim = G4_SW_KV_HEADS * G4_SW_HEAD_DIM;
    int max_q_dim = G4_HEADS * G4_FULL_HEAD_DIM;
    int tok = max_tokens;

    /* Scratch for Q4_K dequant: largest weight is Q proj of full layers [K=3072, N=8192] */
    size_t max_dequant = (size_t)G4_HIDDEN * G4_MAX_Q_DIM; /* 3072 * 8192 */
    size_t scratch_size = max_dequant;
    cudaMallocAsync(&ctx->d_scratch, scratch_size * sizeof(float), ctx->stream);
    cudaMallocAsync(&ctx->d_q, (size_t)tok * max_q_dim * sizeof(float), ctx->stream);
    cudaMallocAsync(&ctx->d_k, (size_t)tok * max_kv_dim * sizeof(float), ctx->stream);
    cudaMallocAsync(&ctx->d_v, (size_t)tok * max_kv_dim * sizeof(float), ctx->stream);
    cudaMallocAsync(&ctx->d_attn_out, (size_t)tok * max_q_dim * sizeof(float), ctx->stream);
    cudaMallocAsync(&ctx->d_ffn_gate, (size_t)tok * G4_FFN * sizeof(float), ctx->stream);
    cudaMallocAsync(&ctx->d_ffn_up, (size_t)tok * G4_FFN * sizeof(float), ctx->stream);
    cudaMallocAsync(&ctx->d_hidden, (size_t)tok * G4_HIDDEN * 2 * sizeof(float), ctx->stream);
    cudaMallocAsync(&ctx->d_normed, (size_t)tok * G4_HIDDEN * sizeof(float), ctx->stream);
    cudaMallocAsync(&ctx->d_logits, (size_t)tok * G4_VOCAB * sizeof(float), ctx->stream);
    cudaMallocAsync(&ctx->d_row_buf, (size_t)(G4_HIDDEN > G4_FFN ? G4_HIDDEN : G4_FFN) * sizeof(float), ctx->stream);

    /* KV cache: [n_layers, max_ctx, max_kv_dim] */
    size_t kv_elems = (size_t)n_layers * max_ctx * max_kv_dim;
    cudaMallocAsync(&ctx->d_k_cache, kv_elems * sizeof(float), ctx->stream);
    cudaMallocAsync(&ctx->d_v_cache, kv_elems * sizeof(float), ctx->stream);
    cudaMallocAsync(&ctx->d_positions, (size_t)tok * sizeof(int), ctx->stream);
    cudaMallocAsync(&ctx->d_kv_sizes, (size_t)n_layers * sizeof(int), ctx->stream);
    cudaMemsetAsync(ctx->d_kv_sizes, 0, (size_t)n_layers * sizeof(int), ctx->stream);
    cudaMemsetAsync(ctx->d_k_cache, 0, kv_elems * sizeof(float), ctx->stream);
    cudaMemsetAsync(ctx->d_v_cache, 0, kv_elems * sizeof(float), ctx->stream);

    ctx->max_row_bytes = (size_t)max_q_dim * 18 / 32 + 18;
    ctx->initialized = true;
    cudaStreamSynchronize(ctx->stream);
    printf("[G4_GPU] sm_89+sm_120, buffers: %.0f MB\n", 
           ((double)scratch_size + (double)tok * (max_q_dim * 2 + max_kv_dim * 2 + G4_FFN * 2 + G4_HIDDEN * 4 + G4_VOCAB) + (double)kv_elems * 2) * 4.0 / 1048576.0);
    return ctx;
}

void g4_gpu_destroy(g4_gpu_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->initialized) {
        cudaStreamSynchronize(ctx->stream);
        if (ctx->d_qweight_row) cudaFree(ctx->d_qweight_row);
        cudaFree(ctx->d_scratch);
        cudaFree(ctx->d_q);
        cudaFree(ctx->d_k);
        cudaFree(ctx->d_v);
        cudaFree(ctx->d_attn_out);
        cudaFree(ctx->d_ffn_gate);
        cudaFree(ctx->d_ffn_up);
        cudaFree(ctx->d_hidden);
        cudaFree(ctx->d_normed);
        cudaFree(ctx->d_logits);
        cudaFree(ctx->d_row_buf);
        cudaFree(ctx->d_k_cache);
        cudaFree(ctx->d_v_cache);
        cudaFree(ctx->d_positions);
        cudaFree(ctx->d_kv_sizes);
        cudaStreamDestroy(ctx->stream);
        cublasDestroy(ctx->cublas);
    }
    free(ctx);
}
