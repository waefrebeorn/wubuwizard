/* gpu_gemma4_forward.cu — Triple-buffer SSD→RAM→VRAM bandwidth prefecture
 * for Gemma 4 12B dual-head-dim ISWA on WSL2.
 *
 * Architecture:
 *   3 CUDA streams: COMPUTE (layer N), UPLOAD (layer N+1), PREFETCH (layer N+2)
 *   3 pinned staging buffers rotate: fread → cudaMemcpy → GPU consume
 *   1 contiguous VRAM weight pool (avoids WSL2 cudaMalloc fragmentation)
 *   64-token batch sync-up: accumulate tokens, batched prefill, then decode
 *
 * WSL2 fix: Uses cudaMallocManaged for weight pool instead of repeated cudaMalloc.
 * Falls back to cudaMalloc + single large allocation if managed fails.
 */

#include "gpu_gemma4.h"
#include "wubu_gemma4.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define Q4K_BLOCK_SIZE 144
#define Q4K_N_ELEMS   256
#define QK_K          256

/* ============= GPU kernels (unchanged from previous version) ============= */

__global__ void dequant_q4_0_kernel(const uint8_t *W_q, float *W_f32, int64_t n_blocks) {
    int64_t b = blockIdx.x;
    if (b >= n_blocks) return;
    int tid = threadIdx.x;
    if (tid >= 32) return;
    const uint8_t *blk = W_q + b * 18;
    float d = ldexpf((float)((blk[0] & 0xFF) | ((blk[1] & 0x7F) << 8) & 0x3FF) / 1024.0f,
                     ((blk[1] >> 7) & 0x1F) - 15) * (blk[1] & 0x80 ? -1.0f : 1.0f);
    const uint8_t *qs = blk + 2;
    int shift = (tid & 1) ? 0 : 4;
    int val = (qs[tid/2] >> shift) & 0xF;
    W_f32[(size_t)b * 32 + tid] = d * (float)(val - 8);
}

__global__ void gpu_rms_norm_kernel(const float *x, const float *w, float *y, int n, int N_tok, float eps) {
    int tid = threadIdx.x, tok = blockIdx.x;
    const float *row = x + tok * n;
    float *out = y + tok * n;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) sum += row[i] * row[i];
    __shared__ float s[256];
    s[tid] = sum;
    for (int ss = blockDim.x/2; ss > 0; ss /= 2) { if (tid < ss) s[tid] += s[tid+ss]; __syncthreads(); }
    __syncthreads();
    float rms = rsqrtf(sum / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x) out[i] = row[i] * rms * w[i];
}

__global__ void gpu_rms_norm_q_kernel(const float *x, const float *w, float *y,
                                       int stride, int N_tok, int head_dim, float eps) {
    int tid = threadIdx.x;
    int tok = blockIdx.x;
    int h = blockIdx.y;
    if (tid >= head_dim) return;
    const float *row = x + (size_t)tok * stride + (size_t)h * head_dim;
    float *out = y + (size_t)tok * stride + (size_t)h * head_dim;
    float val = row[tid];
    extern __shared__ float s[];
    s[tid] = val * val;
    __syncthreads();
    for (int ss = blockDim.x/2; ss > 0; ss /= 2) {
        if (tid < ss) s[tid] += s[tid + ss];
        __syncthreads();
    }
    __syncthreads();
    float rms = rsqrtf(s[0] / (float)head_dim + eps);
    out[tid] = val * rms * w[tid];
}

__global__ void gpu_rope_kernel(float *q, float *k, int N_tok, int d_head, int q_stride,
                                 const int *pos, int n_rot, float rope_base,
                                 const float *freqs, int is_full) {
    int tok = blockIdx.x, h = blockIdx.y, i = threadIdx.x * 2;
    if (i >= n_rot) return;
    int p = pos[tok];
    float fv = (is_full && freqs && i/2 < d_head/8) ? freqs[i/2] : 1.0f;
    float theta = (float)p / powf(rope_base * fv, (float)i / (float)d_head);
    float ct = __cosf(theta), st = __sinf(theta);
    int base = tok * q_stride + h * d_head;
    float q0 = q[base+i], q1 = q[base+i+1];
    q[base+i]   = q0*ct - q1*st;
    q[base+i+1] = q0*st + q1*ct;
    if (k) {
        float k0 = k[base+i], k1 = k[base+i+1];
        k[base+i]   = k0*ct - k1*st;
        k[base+i+1] = k0*st + k1*ct;
    }
}

__global__ void gpu_sliding_attn_kernel(const float *q, const float *k_cache, const float *v_cache,
                                         float *out, int N_tok, int n_heads, int n_kv_heads,
                                         int d_head, int window, const int *pos, int kv_size) {
    extern __shared__ float s_scores[];
    int tok = blockIdx.x, hq = blockIdx.y, kv_h = hq % n_kv_heads, tid = threadIdx.x;
    int q_stride = n_heads * d_head, kv_stride = n_kv_heads * d_head;
    int p = pos[tok], kv_end = kv_size - 1;
    if (kv_end > p) kv_end = p;
    int kv_start = (p > window) ? p - window : 0;
    int kv_len = kv_end - kv_start + 1;
    if (kv_len <= 0) {
        for (int d = tid; d < d_head; d += blockDim.x) out[tok*q_stride + hq*d_head + d] = 0.0f;
        return;
    }
    const float *qv = q + tok*q_stride + hq*d_head;
    float *sc = s_scores;
    float scale = rsqrtf((float)d_head), max_s = -1e30f;
    for (int kp = kv_start + tid; kp <= kv_end; kp += blockDim.x) {
        const float *kv = k_cache + kp*kv_stride + kv_h*d_head;
        float s = 0.0f;
        for (int d = 0; d < d_head; d++) s += qv[d] * kv[d];
        s *= scale;
        sc[kp - kv_start] = s;
        if (s > max_s) max_s = s;
    }
    for (int off = blockDim.x/2; off > 0; off /= 2) { if (tid < off) s_scores[tid] = fmaxf(s_scores[tid], s_scores[tid+off]); __syncthreads(); }
    __syncthreads();
    max_s = s_scores[0];
    float sum_e = 0.0f;
    for (int kp = kv_start + tid; kp <= kv_end; kp += blockDim.x) {
        float e = __expf(sc[kp-kv_start] - max_s);
        sc[kp-kv_start] = e;
        sum_e += e;
    }
    s_scores[tid] = sum_e;
    for (int off = blockDim.x/2; off > 0; off /= 2) { if (tid < off) s_scores[tid] += s_scores[tid+off]; __syncthreads(); }
    __syncthreads();
    sum_e = s_scores[0];
    float inv_sum = 1.0f / (sum_e + 1e-10f);
    for (int d = tid; d < d_head; d += blockDim.x) {
        float val = 0.0f;
        for (int kp = kv_start; kp <= kv_end; kp++)
            val += sc[kp-kv_start] * inv_sum * (v_cache + kp*kv_stride + kv_h*d_head)[d];
        out[tok*q_stride + hq*d_head + d] = val;
    }
}

__global__ void gpu_full_attn_kernel(const float *q, const float *k_cache, const float *v_cache,
                                      float *out, int N_tok, int n_heads, int n_kv_heads,
                                      int d_head, const int *pos, int kv_size) {
    extern __shared__ float s_scores[];
    int tok = blockIdx.x, hq = blockIdx.y, kv_h = hq % n_kv_heads, tid = threadIdx.x;
    int q_stride = n_heads * d_head, kv_stride = n_kv_heads * d_head;
    int p = pos[tok], kv_end = kv_size - 1;
    if (kv_end > p) kv_end = p;
    int kv_len = kv_end + 1;
    if (kv_len <= 0) {
        for (int d = tid; d < d_head; d += blockDim.x) out[tok*q_stride+hq*d_head+d] = 0.0f;
        return;
    }
    const float *qv = q + tok*q_stride + hq*d_head;
    float *sc = s_scores;
    float scale = rsqrtf((float)d_head), max_s = -1e30f;
    for (int kp = tid; kp <= kv_end; kp += blockDim.x) {
        const float *kv = k_cache + kp*kv_stride + kv_h*d_head;
        float s = 0.0f;
        for (int d = 0; d < d_head; d++) s += qv[d] * kv[d];
        s *= scale; sc[kp] = s;
        if (s > max_s) max_s = s;
    }
    for (int off = blockDim.x/2; off > 0; off /= 2) { if (tid < off) s_scores[tid] = fmaxf(s_scores[tid], s_scores[tid+off]); __syncthreads(); }
    __syncthreads(); max_s = s_scores[0];
    float sum_e = 0.0f;
    for (int kp = tid; kp <= kv_end; kp += blockDim.x) {
        float e = __expf(sc[kp] - max_s); sc[kp] = e; sum_e += e;
    }
    s_scores[tid] = sum_e;
    for (int off = blockDim.x/2; off > 0; off /= 2) { if (tid < off) s_scores[tid] += s_scores[tid+off]; __syncthreads(); }
    __syncthreads(); sum_e = s_scores[0];
    float inv_sum = 1.0f / (sum_e + 1e-10f);
    for (int d = tid; d < d_head; d += blockDim.x) {
        float val = 0.0f;
        for (int kp = 0; kp <= kv_end; kp++)
            val += sc[kp] * inv_sum * (v_cache + kp*kv_stride + kv_h*d_head)[d];
        out[tok*q_stride + hq*d_head + d] = val;
    }
}

__global__ void gpu_gelu_kernel(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i], x3 = v*v*v;
    x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * x3)));
}

__global__ void gpu_mul_kernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) c[i] = a[i]*b[i];
}
__global__ void gpu_add_kernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) c[i] = a[i]+b[i];
}
__global__ void gpu_scale_kernel(float *x, float s, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) x[i] *= s;
}
__global__ void gpu_softcap_kernel(float *x, int n, float cap) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) { float sc = x[i]/cap; x[i] = tanhf(sc)*cap; }
}

/* ============= cuBLAS wrappers ============= */

static void gpu_sgemm_rm_cm(cublasHandle_t cublas, cudaStream_t stream,
                            int M, int N, int K,
                            const float *d_A_row, const float *d_B_col, float *d_C_row) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(cublas, stream);
    cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                &alpha, d_B_col, K, d_A_row, K, &beta, d_C_row, N);
}

/* ============= Q4_K dequant + matmul (GPU-native, weights stay compressed) ============= */

static void dequantize_q4_K_block(const uint8_t *block, float *out) {
    uint16_t d_bits, dmin_bits;
    memcpy(&d_bits, block, 2);
    memcpy(&dmin_bits, block + 2, 2);
    int s = (d_bits >> 15) & 1, e = (d_bits >> 10) & 0x1F, m = d_bits & 0x3FF;
    float d = (e == 0) ? ldexpf((float)m / 1024.0f, -14) * (s ? -1.0f : 1.0f)
            : (e == 31) ? (s ? -__builtin_huge_valf() : __builtin_huge_valf())
            : ldexpf(1.0f + (float)m / 1024.0f, e - 15) * (s ? -1.0f : 1.0f);
    s = (dmin_bits >> 15) & 1; e = (dmin_bits >> 10) & 0x1F; m = dmin_bits & 0x3FF;
    float dmin = (e == 0) ? ldexpf((float)m / 1024.0f, -14) * (s ? -1.0f : 1.0f)
               : (e == 31) ? (s ? -__builtin_huge_valf() : __builtin_huge_valf())
               : ldexpf(1.0f + (float)m / 1024.0f, e - 15) * (s ? -1.0f : 1.0f);
    const uint8_t *scales = block + 4;
    const uint8_t *qs = block + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc1, m1, sc2, m2;
        int idx = is;
        if (idx < 4) { sc1 = scales[idx] & 63; m1 = scales[idx + 4] & 63; }
        else { sc1 = (scales[idx+4] & 0xF) | ((scales[idx-4] >> 6) << 4);
               m1  = (scales[idx+4] >>  4) | ((scales[idx  ] >> 6) << 4); }
        idx = is + 1;
        if (idx < 4) { sc2 = scales[idx] & 63; m2 = scales[idx + 4] & 63; }
        else { sc2 = (scales[idx+4] & 0xF) | ((scales[idx-4] >> 6) << 4);
               m2  = (scales[idx+4] >>  4) | ((scales[idx  ] >> 6) << 4); }
        float d1 = d * (float)sc1; float ml1 = dmin * (float)m1;
        float d2 = d * (float)sc2; float ml2 = dmin * (float)m2;
        const uint8_t *bq = qs + j/2;
        for (int l = 0; l < 32; l++) out[j + l]      = d1 * (float)(bq[l] & 0xF) - ml1;
        for (int l = 0; l < 32; l++) out[j + 32 + l] = d2 * (float)(bq[l] >> 4) - ml2;
        is += 2;
    }
}

static __device__ void decode_q4k_scales(const uint8_t *scales, int group_idx,
                                          float d, float dmin,
                                          float *d1, float *ml1, float *d2, float *ml2) {
    uint8_t sc1, m1, sc2, m2;
    int is = group_idx * 2;
    int idx = is;
    if (idx < 4) {
        sc1 = scales[idx] & 63;
        m1  = scales[idx + 4] & 63;
    } else {
        sc1 = (scales[idx+4] & 0xF) | ((scales[idx-4] >> 6) << 4);
        m1  = (scales[idx+4] >>  4) | ((scales[idx  ] >> 6) << 4);
    }
    idx = is + 1;
    if (idx < 4) {
        sc2 = scales[idx] & 63;
        m2  = scales[idx + 4] & 63;
    } else {
        sc2 = (scales[idx+4] & 0xF) | ((scales[idx-4] >> 6) << 4);
        m2  = (scales[idx+4] >>  4) | ((scales[idx  ] >> 6) << 4);
    }
    *d1  = d  * (float)sc1;
    *ml1 = dmin * (float)m1;
    *d2  = d  * (float)sc2;
    *ml2 = dmin * (float)m2;
}

__global__ void gpu_q4k_matmul_kernel(
    const uint8_t *W_q, const float *x, float *y, int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int n_blocks_k = (K + 255) / 256;
    int64_t col_offset = (int64_t)n * n_blocks_k * 144;
    float acc = 0.0f;
    for (int bk = 0; bk < n_blocks_k; bk++) {
        const uint8_t *blk = W_q + col_offset + (int64_t)bk * 144;
        uint16_t d_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        uint16_t dmin_bits = (uint16_t)blk[2] | ((uint16_t)blk[3] << 8);
        int s = (d_bits >> 15) & 1, e = (d_bits >> 10) & 0x1F, m = d_bits & 0x3FF;
        float d = (e == 0) ? ldexpf((float)m / 1024.0f, -14) * (s ? -1.0f : 1.0f)
                  : (e == 31) ? (s ? -__builtin_huge_valf() : __builtin_huge_valf())
                  : ldexpf(1.0f + (float)m / 1024.0f, e - 15) * (s ? -1.0f : 1.0f);
        s = (dmin_bits >> 15) & 1; e = (dmin_bits >> 10) & 0x1F; m = dmin_bits & 0x3FF;
        float dmin = (e == 0) ? ldexpf((float)m / 1024.0f, -14) * (s ? -1.0f : 1.0f)
                   : (e == 31) ? (s ? -__builtin_huge_valf() : __builtin_huge_valf())
                   : ldexpf(1.0f + (float)m / 1024.0f, e - 15) * (s ? -1.0f : 1.0f);
        const uint8_t *scales = blk + 4;
        const uint8_t *qs = blk + 16;
        int k_base = bk * 256;
        for (int group = 0; group < 4; group++) {
            float d1, ml1, d2, ml2;
            decode_q4k_scales(scales, group, d, dmin, &d1, &ml1, &d2, &ml2);
            int group_start = group * 64;
            const uint8_t *bq = qs + group_start / 2;
            int k0 = k_base + group_start;
            for (int l = 0; l < 32 && k0 + l < K; l++) {
                float val = d1 * (float)(bq[l] & 0xF) - ml1;
                acc += x[k0 + l] * val;
            }
            int k1 = k_base + group_start + 32;
            for (int l = 0; l < 32 && k1 + l < K; l++) {
                float val = d2 * (float)(bq[l] >> 4) - ml2;
                acc += x[k1 + l] * val;
            }
        }
    }
    y[n] = acc;
}

/* ============= GPU matmul dispatchers ============= */

static void gpu_matmul_q4k(cudaStream_t stream, cublasHandle_t cublas,
                            float *d_scratch, const uint8_t *d_W,
                            const float *d_x, int M, int N, int K,
                            float *d_y, const uint8_t *h_W) {
    if (M == 1) {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        gpu_q4k_matmul_kernel<<<blocks, threads, 0, stream>>>(d_W, d_x, d_y, N, K);
        return;
    }
    /* Prefill (M>1): CPU dequant + cuBLAS */
    size_t f32_size = (size_t)K * N * sizeof(float);
    float *h_f32 = (float*)malloc(f32_size);
    if (!h_f32) { fprintf(stderr, "OOM in gpu_matmul_q4k\n"); return; }
    gguf_dequantize(h_W, GGML_TYPE_Q4_K, (int64_t)K * N, h_f32);
    cudaMemcpyAsync(d_scratch, h_f32, (size_t)K * N * sizeof(float),
                   cudaMemcpyHostToDevice, stream);
    gpu_sgemm_rm_cm(cublas, stream, M, N, K, d_x, d_scratch, d_y);
    free(h_f32);
}

static void gpu_matmul_q4_0(cudaStream_t stream, cublasHandle_t cublas,
                             float *d_scratch, const uint8_t *d_W,
                             const float *d_x, int M, int N, int K,
                             float *d_y) {
    int blocks_per_col = (K + 31) / 32;
    int total_blocks = blocks_per_col * N;
    dequant_q4_0_kernel<<<total_blocks, 32, 0, stream>>>(d_W, d_scratch, total_blocks);
    gpu_sgemm_rm_cm(cublas, stream, M, N, K, d_x, d_scratch, d_y);
}

static void gpu_matmul(cudaStream_t stream, cublasHandle_t cublas,
                         float *d_scratch, const uint8_t *d_W,
                         const float *d_x, int M, int N, int K,
                         float *d_y, const uint8_t *h_W, int ggml_type) {
    if (ggml_type == GGML_TYPE_Q4_K) {
        gpu_matmul_q4k(stream, cublas, d_scratch, d_W, d_x, M, N, K, d_y, h_W);
    } else if (ggml_type == GGML_TYPE_Q4_0) {
        gpu_matmul_q4_0(stream, cublas, d_scratch, d_W, d_x, M, N, K, d_y);
    } else if (ggml_type == GGML_TYPE_IQ4_NL) {
        // IQ4_NL: 18 bytes per 32 elements (d: fp16 + qs[16])
        // Same block layout as Q4_0 but with non-linear kvalues
        // For now: dequant on host, then cuBLAS
        size_t f32_size = (size_t)K * N * sizeof(float);
        float *h_f32 = (float*)malloc(f32_size);
        if (!h_f32) { fprintf(stderr, "OOM in gpu_matmul (IQ4_NL)\n"); return; }
        gguf_dequantize(h_W, ggml_type, (int64_t)K * N, h_f32);
        cudaMemcpyAsync(d_scratch, h_f32, (size_t)K * N * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
        gpu_sgemm_rm_cm(cublas, stream, M, N, K, d_x, d_scratch, d_y);
        free(h_f32);
    } else if (ggml_type == GGML_TYPE_IQ2_XS) {
        // IQ2_XS: 108 bytes per 256 elements
        // For now: dequant on host, then cuBLAS
        size_t f32_size = (size_t)K * N * sizeof(float);
        float *h_f32 = (float*)malloc(f32_size);
        if (!h_f32) { fprintf(stderr, "OOM in gpu_matmul (IQ2_XS)\n"); return; }
        gguf_dequantize(h_W, ggml_type, (int64_t)K * N, h_f32);
        cudaMemcpyAsync(d_scratch, h_f32, (size_t)K * N * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
        gpu_sgemm_rm_cm(cublas, stream, M, N, K, d_x, d_scratch, d_y);
        free(h_f32);
    } else {
        size_t f32_size = (size_t)K * N * sizeof(float);
        float *h_f32 = (float*)malloc(f32_size);
        if (!h_f32) { fprintf(stderr, "OOM in gpu_matmul\n"); return; }
        gguf_dequantize(h_W, ggml_type, (int64_t)K * N, h_f32);
        cudaMemcpyAsync(d_scratch, h_f32, (size_t)K * N * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
        gpu_sgemm_rm_cm(cublas, stream, M, N, K, d_x, d_scratch, d_y);
        free(h_f32);
    }
}

/* ============= Weight pool sub-allocation (WSL2 fix) ============= */
/* Instead of 336 separate cudaMalloc calls (7 weights × 48 layers), we allocate
 * one big contiguous block and sub-allocate from it. This avoids the WSL2 DXGI
 * shim memory pool corruption that causes "illegal memory access" on layer 1+. */

uint8_t *g4_gpu_weight_alloc(g4_gpu_ctx_t *ctx, size_t bytes) {
    /* Align to 256 bytes for CUDA */
    size_t aligned = (bytes + 255) & ~(size_t)255;
    if (ctx->weight_pool_used + aligned > ctx->weight_pool_size) {
        fprintf(stderr, "[G4_GPU] weight pool exhausted: need %zu, have %zu/%zu\n",
                aligned, ctx->weight_pool_used, ctx->weight_pool_size);
        return NULL;
    }
    uint8_t *ptr = ctx->d_weight_pool + ctx->weight_pool_used;
    ctx->weight_pool_used += aligned;
    return ptr;
}

/* ============= Triple-buffer pipeline state ============= */

static int pipeline_initialized = 0;
static int weights_uploaded = 0;

static void ensure_pipeline_initialized(g4_gpu_ctx_t *ctx) {
    if (pipeline_initialized) return;
    memset(ctx->layer_vram, 0, sizeof(ctx->layer_vram));
    pipeline_initialized = 1;
}

/* Find a free staging buffer index, or -1 if all busy */
static int find_free_buffer(g4_gpu_ctx_t *ctx) {
    for (int i = 0; i < G4_GPU_N_BUFFERS; i++) {
        if (ctx->ram_owner[i] == -1) return i;
    }
    return -1;
}

/* Wait for a buffer's prefetch to complete, then mark it free */
static void release_buffer(g4_gpu_ctx_t *ctx, int buf_idx) {
    if (buf_idx < 0 || buf_idx >= G4_GPU_N_BUFFERS) return;
    /* Only synchronize if this buffer was actually used for an upload */
    if (ctx->ram_owner[buf_idx] != -1) {
        cudaEventSynchronize(ctx->upload_done[buf_idx]);
    }
    ctx->ram_owner[buf_idx] = -1;
}

/* Read weights from file into a specific staging buffer, then async-upload to VRAM.
 * This is the core of the SSD→RAM→VRAM prefecture. */
static int stream_weights_to_vram(g4_gpu_ctx_t *ctx, int buf_idx,
                                   int il, g4_layer_t *l, g4_model_t *model) {
    uint8_t *staging = ctx->ram_buf[buf_idx];
    g4_layer_vram_t *vw = &ctx->layer_vram[il];
    cudaStream_t upload_stream = ctx->stream[1];  /* UPLOAD stream */

    /* Helper: read from file into staging, then async cudaMemcpy to VRAM */
    auto upload_one = [&](const uint8_t *data_ptr, int nbytes, uint8_t **d_out) -> int {
        if (!data_ptr || nbytes <= 0) { *d_out = NULL; return 1; }

        /* Sub-allocate from weight pool instead of cudaMalloc */
        *d_out = g4_gpu_weight_alloc(ctx, (size_t)nbytes);
        if (!d_out) {
            fprintf(stderr, "[G4_GPU] weight_alloc failed for %d bytes (layer %d)\n", nbytes, il);
            return 0;
        }

        /* Read from file into staging buffer */
        uint64_t offset_in_blob = (uint64_t)(data_ptr - (const uint8_t*)model->data_blob);
        uint64_t file_offset = model->gguf_data_offset + offset_in_blob;

        if (nbytes <= (int)G4_GPU_RAW_BUF_SIZE) {
            /* Fits in single staging buffer */
            if (fseek(model->gguf_file, (long)file_offset, SEEK_SET) != 0) {
                fprintf(stderr, "[G4_GPU] fseek failed offset %lu\n", (unsigned long)file_offset);
                return 0;
            }
            size_t n_read = fread(staging, 1, (size_t)nbytes, model->gguf_file);
            if (n_read != (size_t)nbytes) {
                fprintf(stderr, "[G4_GPU] fread failed: %zu/%d\n", n_read, nbytes);
                return 0;
            }
            cudaMemcpyAsync(*d_out, staging, (size_t)nbytes, cudaMemcpyHostToDevice, upload_stream);
        } else {
            /* Large tensor (e.g. token_embd 540MB): chunk it */
            uint8_t *chunk = (uint8_t*)malloc((size_t)nbytes);
            if (!chunk) { fprintf(stderr, "[G4_GPU] malloc failed %d\n", nbytes); return 0; }
            if (fseek(model->gguf_file, (long)file_offset, SEEK_SET) != 0) {
                free(chunk); return 0;
            }
            size_t n_read = fread(chunk, 1, (size_t)nbytes, model->gguf_file);
            if (n_read != (size_t)nbytes) {
                fprintf(stderr, "[G4_GPU] fread failed: %zu/%d\n", n_read, nbytes);
                free(chunk); return 0;
            }
            cudaMemcpyAsync(*d_out, chunk, (size_t)nbytes, cudaMemcpyHostToDevice, upload_stream);
            free(chunk);
        }
        return 1;
    };

    int ok = 1;
    ok = ok && upload_one(l->attn_q.data, l->attn_q.raw_bytes, &vw->d_q);
    vw->q_bytes = l->attn_q.raw_bytes;
    if (!l->share_kv) {
        ok = ok && upload_one(l->attn_k.data, l->attn_k.raw_bytes, &vw->d_k);
        vw->k_bytes = l->attn_k.raw_bytes;
        if (!l->kv_eq) {
            ok = ok && upload_one(l->attn_v.data, l->attn_v.raw_bytes, &vw->d_v);
            vw->v_bytes = l->attn_v.raw_bytes;
        }
    }
    ok = ok && upload_one(l->attn_out.data, l->attn_out.raw_bytes, &vw->d_o);
    vw->o_bytes = l->attn_out.raw_bytes;
    ok = ok && upload_one(l->ffn_gate.data, l->ffn_gate.raw_bytes, &vw->d_g);
    vw->g_bytes = l->ffn_gate.raw_bytes;
    ok = ok && upload_one(l->ffn_up.data, l->ffn_up.raw_bytes, &vw->d_u);
    vw->u_bytes = l->ffn_up.raw_bytes;
    ok = ok && upload_one(l->ffn_down.data, l->ffn_down.raw_bytes, &vw->d_d);
    vw->d_bytes = l->ffn_down.raw_bytes;

    /* Record upload completion event */
    cudaEventRecord(ctx->upload_done[buf_idx], upload_stream);

    if (!ok) {
        fprintf(stderr, "[G4_GPU] ERROR: layer %d weight upload failed\n", il);
        return 0;
    }
    vw->uploaded = 1;
    return 1;
}

/* ============= GPU context init / destroy ============= */

g4_gpu_ctx_t *g4_gpu_init(int max_tokens, int max_ctx, int n_layers) {
    g4_gpu_ctx_t *ctx = (g4_gpu_ctx_t*)calloc(1, sizeof(g4_gpu_ctx_t));
    if (!ctx) return NULL;

    ctx->max_tokens = max_tokens;
    ctx->max_ctx = max_ctx;
    ctx->n_layers = n_layers;

    /* Create CUDA streams */
    for (int i = 0; i < G4_GPU_N_STREAMS; i++) {
        cudaStreamCreate(&ctx->stream[i]);
    }
    /* Create events */
    for (int i = 0; i < G4_GPU_N_BUFFERS; i++) {
        cudaEventCreate(&ctx->upload_done[i]);
        cudaEventCreate(&ctx->prefetch_done[i]);
        ctx->ram_owner[i] = -1;
    }

    /* Allocate pinned staging buffers - disabled on WSL2 due to cudaHostAlloc/cudaFreeHost issues */
    for (int i = 0; i < G4_GPU_N_BUFFERS; i++) {
        ctx->ram_buf[i] = (uint8_t*)malloc(G4_GPU_RAW_BUF_SIZE);
        if (!ctx->ram_buf[i]) {
            fprintf(stderr, "[G4_GPU] Failed to allocate staging buffer %d\n", i);
            g4_gpu_destroy(ctx);
            return NULL;
        }
        ctx->ram_buf_pinned[i] = 0;  /* 0 = malloc'd (pinned disabled on WSL2) */
    }

    /* Allocate weight pool: try cudaMallocManaged first (WSL2 fix), fallback to cudaMalloc */
    /* Total weight size: ~3GB for all layer weights (norms + token_embd are uploaded separately) */
    /* 6GB VRAM - 1GB KV cache - 500MB activations = ~4.5GB available for weights */
    ctx->weight_pool_size = (size_t)2 * 1024 * 1024 * 1024ULL;  /* 2 GB */
    cudaError_t pool_err = cudaMallocManaged(&ctx->d_weight_pool, ctx->weight_pool_size);
    if (pool_err != cudaSuccess) {
        fprintf(stderr, "[G4_GPU] cudaMallocManaged failed (%s), trying cudaMalloc...\n",
                cudaGetErrorString(pool_err));
        pool_err = cudaMalloc(&ctx->d_weight_pool, ctx->weight_pool_size);
        if (pool_err != cudaSuccess) {
            /* Try smaller pool */
            ctx->weight_pool_size = (size_t)1536 * 1024 * 1024ULL;
            pool_err = cudaMalloc(&ctx->d_weight_pool, ctx->weight_pool_size);
            if (pool_err != cudaSuccess) {
                fprintf(stderr, "[G4_GPU] cudaMalloc weight pool failed: %s\n",
                        cudaGetErrorString(pool_err));
                g4_gpu_destroy(ctx);
                return NULL;
            }
        }
    }
    ctx->weight_pool_used = 0;
    fprintf(stderr, "[G4_GPU] Weight pool: %zu MB (%s)\n",
            ctx->weight_pool_size / (1024*1024),
            (pool_err == cudaSuccess) ? "managed" : "device");

    /* Create cuBLAS handle */
    cublasCreate(&ctx->cublas);
    cublasSetMathMode(ctx->cublas, CUBLAS_TENSOR_OP_MATH);

    /* Allocate activation buffers */
    size_t act_size = (size_t)max_tokens * G4_MAX_Q_DIM * sizeof(float);
    cudaMalloc(&ctx->d_scratch, act_size);
    cudaMalloc(&ctx->d_q, act_size);
    cudaMalloc(&ctx->d_k, (size_t)max_tokens * G4_MAX_KV_DIM * sizeof(float));
    cudaMalloc(&ctx->d_v, (size_t)max_tokens * G4_MAX_KV_DIM * sizeof(float));
    cudaMalloc(&ctx->d_attn_out, act_size);
    cudaMalloc(&ctx->d_ffn_gate, (size_t)max_tokens * G4_FFN * sizeof(float));
    cudaMalloc(&ctx->d_ffn_up, (size_t)max_tokens * G4_FFN * sizeof(float));
    cudaMalloc(&ctx->d_hidden, (size_t)max_tokens * G4_HIDDEN * sizeof(float));
    cudaMalloc(&ctx->d_normed, (size_t)max_tokens * G4_HIDDEN * sizeof(float));
    cudaMalloc(&ctx->d_logits, (size_t)max_tokens * G4_VOCAB * sizeof(float));
    cudaMalloc(&ctx->d_row_buf, (size_t)G4_HIDDEN * sizeof(float));

    /* Allocate KV cache on device */
    size_t kv_cache_size = (size_t)n_layers * max_ctx * G4_MAX_KV_DIM * sizeof(float);
    cudaMalloc(&ctx->d_k_cache, kv_cache_size);
    cudaMalloc(&ctx->d_v_cache, kv_cache_size);
    cudaMemset(ctx->d_k_cache, 0, kv_cache_size);
    cudaMemset(ctx->d_v_cache, 0, kv_cache_size);

    cudaMalloc(&ctx->d_positions, (size_t)max_tokens * sizeof(int));
    cudaMalloc(&ctx->d_kv_sizes, (size_t)n_layers * sizeof(int));
    cudaMemset(ctx->d_kv_sizes, 0, (size_t)n_layers * sizeof(int));

    /* Init batch state */
    ctx->batch_len = 0;
    ctx->batch_pos = 0;

    ctx->initialized = 1;
    fprintf(stderr, "[G4_GPU] Initialized: %d layers, max_ctx=%d, max_tokens=%d\n",
            n_layers, max_ctx, max_tokens);
    return ctx;
}

void g4_gpu_destroy(g4_gpu_ctx_t *ctx) {
    if (!ctx) return;
    /* Free pinned buffers FIRST - before destroying streams/context */
    for (int i = 0; i < G4_GPU_N_BUFFERS; i++) {
        if (ctx->ram_buf[i]) {
            if (ctx->ram_buf_pinned[i]) {
                cudaFreeHost(ctx->ram_buf[i]);
            } else {
                free(ctx->ram_buf[i]);
            }
        }
    }
    for (int i = 0; i < G4_GPU_N_STREAMS; i++) {
        if (ctx->stream[i]) cudaStreamDestroy(ctx->stream[i]);
    }
    for (int i = 0; i < G4_GPU_N_BUFFERS; i++) {
        if (ctx->upload_done[i]) cudaEventDestroy(ctx->upload_done[i]);
        if (ctx->prefetch_done[i]) cudaEventDestroy(ctx->prefetch_done[i]);
    }
    if (ctx->d_weight_pool) cudaFree(ctx->d_weight_pool);
    if (ctx->d_scratch) cudaFree(ctx->d_scratch);
    if (ctx->d_q) cudaFree(ctx->d_q);
    if (ctx->d_k) cudaFree(ctx->d_k);
    if (ctx->d_v) cudaFree(ctx->d_v);
    if (ctx->d_attn_out) cudaFree(ctx->d_attn_out);
    if (ctx->d_ffn_gate) cudaFree(ctx->d_ffn_gate);
    if (ctx->d_ffn_up) cudaFree(ctx->d_ffn_up);
    if (ctx->d_hidden) cudaFree(ctx->d_hidden);
    if (ctx->d_normed) cudaFree(ctx->d_normed);
    if (ctx->d_logits) cudaFree(ctx->d_logits);
    if (ctx->d_row_buf) cudaFree(ctx->d_row_buf);
    if (ctx->d_k_cache) cudaFree(ctx->d_k_cache);
    if (ctx->d_v_cache) cudaFree(ctx->d_v_cache);
    if (ctx->d_positions) cudaFree(ctx->d_positions);
    if (ctx->d_kv_sizes) cudaFree(ctx->d_kv_sizes);
    if (ctx->d_token_embd) cudaFree(ctx->d_token_embd);
    if (ctx->d_output_w) cudaFree(ctx->d_output_w);
    if (ctx->d_output_norm) cudaFree(ctx->d_output_norm);
    for (int i = 0; i < 48; i++) {
        if (ctx->d_norm[i]) cudaFree(ctx->d_norm[i]);
        if (ctx->d_qnorm[i]) cudaFree(ctx->d_qnorm[i]);
        if (ctx->d_knorm[i]) cudaFree(ctx->d_knorm[i]);
        if (ctx->d_pnorm[i]) cudaFree(ctx->d_pnorm[i]);
        if (ctx->d_fnorm[i]) cudaFree(ctx->d_fnorm[i]);
        if (ctx->d_pfnorm[i]) cudaFree(ctx->d_pfnorm[i]);
        if (ctx->d_rope_freqs[i]) cudaFree(ctx->d_rope_freqs[i]);
    }
    if (ctx->cublas) cublasDestroy(ctx->cublas);
    free(ctx);
}

/* ============= One-time weight upload (norms, rope, token_embd) ============= */

static int ensure_weights_uploaded(g4_gpu_ctx_t *ctx, g4_model_t *model) {
    if (weights_uploaded) return 1;
    ensure_pipeline_initialized(ctx);
    cudaStream_t s = ctx->stream[0];

    auto upload_f = [&](const float *h, int n) -> float* {
        if (!h) return NULL;
        float *d;
        cudaMalloc(&d, n * sizeof(float));
        cudaMemcpyAsync(d, h, n * sizeof(float), cudaMemcpyHostToDevice, s);
        return d;
    };

    for (int i = 0; i < model->n_layers; i++) {
        g4_layer_t *l = &model->layers[i];
        ctx->d_norm[i] = upload_f(l->attn_norm_weight, G4_HIDDEN);
        ctx->d_qnorm[i] = upload_f(l->attn_q_norm_weight, l->head_dim);
        ctx->d_knorm[i] = upload_f(l->attn_k_norm_weight, l->head_dim);
        ctx->d_pnorm[i] = upload_f(l->post_attn_norm_weight, G4_HIDDEN);
        ctx->d_fnorm[i] = upload_f(l->ffn_norm_weight, G4_HIDDEN);
        ctx->d_pfnorm[i] = upload_f(l->post_ffn_norm_weight, G4_HIDDEN);
        if (l->has_rope_freqs) {
            ctx->d_rope_freqs[i] = upload_f(l->rope_freqs, l->head_dim / 2);
        } else {
            ctx->d_rope_freqs[i] = NULL;
        }
    }
    cudaStreamSynchronize(s);

    /* Upload token_embd via weight pool (sub-allocated) */
    if (model->token_embd.data) {
        ctx->d_token_embd = g4_gpu_weight_alloc(ctx, model->token_embd.raw_bytes);
        if (!ctx->d_token_embd) {
            return 0;
        }
        /* Read from file in chunks */
        uint64_t offset_in_blob = (uint64_t)(model->token_embd.data - (const uint8_t*)model->data_blob);
        uint64_t file_offset = model->gguf_data_offset + offset_in_blob;
        if (fseek(model->gguf_file, (long)file_offset, SEEK_SET) != 0) return 0;

        const size_t CHUNK = 4 * 1024 * 1024;
        uint8_t *h_buf = (uint8_t*)malloc(CHUNK);
        size_t remaining = model->token_embd.raw_bytes;
        size_t gpu_offset = 0;
        while (remaining > 0) {
            size_t chunk = remaining < CHUNK ? remaining : CHUNK;
            size_t n_read = fread(h_buf, 1, chunk, model->gguf_file);
            if (n_read != chunk) break;
            cudaMemcpy(ctx->d_token_embd + gpu_offset, h_buf, chunk, cudaMemcpyHostToDevice);
            gpu_offset += chunk;
            remaining -= chunk;
        }
        free(h_buf);
        if (remaining > 0) return 0;
    }

    if (!model->tied_output && model->output.data) {
        ctx->d_output_w = g4_gpu_weight_alloc(ctx, model->output.raw_bytes);
        if (!ctx->d_output_w) return 0;
        uint64_t offset_in_blob = (uint64_t)(model->output.data - (const uint8_t*)model->data_blob);
        uint64_t file_offset = model->gguf_data_offset + offset_in_blob;
        if (fseek(model->gguf_file, (long)file_offset, SEEK_SET) != 0) return 0;
        const size_t CHUNK2 = 4 * 1024 * 1024;
        uint8_t *h_buf2 = (uint8_t*)malloc(CHUNK2);
        size_t remaining = model->output.raw_bytes;
        size_t gpu_offset = 0;
        while (remaining > 0) {
            size_t chunk = remaining < CHUNK2 ? remaining : CHUNK2;
            size_t n_read = fread(h_buf2, 1, chunk, model->gguf_file);
            if (n_read != chunk) break;
            cudaMemcpy(ctx->d_output_w + gpu_offset, h_buf2, chunk, cudaMemcpyHostToDevice);
            gpu_offset += chunk;
            remaining -= chunk;
        }
        free(h_buf2);
        if (remaining > 0) return 0;
    }

    ctx->d_output_norm = upload_f(model->output_norm_weight, G4_HIDDEN);
    cudaStreamSynchronize(s);

    weights_uploaded = 1;
    fprintf(stderr, "[G4_GPU] One-time weights uploaded. Weight pool used: %zu / %zu MB\n",
            ctx->weight_pool_used / (1024*1024), ctx->weight_pool_size / (1024*1024));
    return 1;
}



/* ============= Evict layer weights (free from VRAM by resetting pool offset) ============= */
/* With the weight pool approach, we don't free individual allocations.
 * Instead, we mark the layer as not-uploaded and the pool space is reclaimed
 * when we reset the pool (between forward passes). For now, just mark. */
static void evict_layer_weights(g4_gpu_ctx_t *ctx, int il) {
    g4_layer_vram_t *vw = &ctx->layer_vram[il];
    if (!vw->uploaded) return;
    vw->uploaded = 0;
    /* Note: actual VRAM is reclaimed by resetting weight_pool_used between batches */
}

/* ============= Triple-buffer forward for one layer ============= */

static void forward_layer_gpu(g4_gpu_ctx_t *ctx, g4_model_t *model, int il,
                               float *d_x, float *d_n, float *d_q_buf,
                               float *d_k_buf, float *d_v_buf, float *d_a_buf,
                               float *d_g_buf, float *d_u_buf, float *d_s_buf,
                               int N, const int *h_pos) {
    g4_layer_t *l = &model->layers[il];
    cudaStream_t compute = ctx->stream[0];
    cublasHandle_t cublas = ctx->cublas;
    int hd = l->head_dim, qd = l->q_dim, kvd = l->kv_dim, kvh = l->kv_heads;

    g4_layer_vram_t *vw = &ctx->layer_vram[il];
    uint8_t *d_wq = vw->d_q;
    uint8_t *d_wk = vw->d_k;
    uint8_t *d_wv = vw->d_v;
    uint8_t *d_wo = vw->d_o;
    uint8_t *d_wg = vw->d_g;
    uint8_t *d_wu = vw->d_u;
    uint8_t *d_wd = vw->d_d;

    /* RMS norm */
    gpu_rms_norm_kernel<<<N, 256, 0, compute>>>(d_x, ctx->d_norm[il], d_n, G4_HIDDEN, N, G4_RMS_EPS);

    /* Q matmul */
    gpu_matmul(compute, cublas, d_s_buf, d_wq, d_n, N, qd, G4_HIDDEN, d_q_buf, l->attn_q.data, l->attn_q.ggml_type);

    /* K, V matmuls */
    if (!l->share_kv) {
        gpu_matmul(compute, cublas, d_s_buf, d_wk, d_n, N, kvd, G4_HIDDEN, d_k_buf, l->attn_k.data, l->attn_k.ggml_type);
        if (l->kv_eq) {
            cudaMemcpyAsync(d_v_buf, d_k_buf, (size_t)N * kvd * sizeof(float), cudaMemcpyDeviceToDevice, compute);
        } else {
            gpu_matmul(compute, cublas, d_s_buf, d_wv, d_n, N, kvd, G4_HIDDEN, d_v_buf, l->attn_v.data, l->attn_v.ggml_type);
        }
    }

    /* Q/K norm + RoPE */
    {
        size_t shm_norm = (size_t)hd * sizeof(float);
        dim3 qn_grid(N, G4_HEADS);
        gpu_rms_norm_q_kernel<<<qn_grid, hd, shm_norm, compute>>>(
            d_q_buf, ctx->d_qnorm[il], d_q_buf, qd, N, hd, G4_RMS_EPS);

        if (!l->share_kv) {
            dim3 kn_grid(N, kvh);
            gpu_rms_norm_q_kernel<<<kn_grid, hd, shm_norm, compute>>>(
                d_k_buf, ctx->d_knorm[il], d_k_buf, kvd, N, hd, G4_RMS_EPS);
        }

        dim3 rope_grid(N, qd / hd);
        int rope_threads = hd / 2;
        if (rope_threads > 128) rope_threads = 128;
        if (!l->share_kv) {
            gpu_rope_kernel<<<rope_grid, rope_threads, 0, compute>>>(
                d_q_buf, d_k_buf, N, hd, qd, ctx->d_positions, l->n_rot, l->rope_base, ctx->d_rope_freqs[il], l->is_full);
        } else {
            gpu_rope_kernel<<<rope_grid, rope_threads, 0, compute>>>(
                d_q_buf, NULL, N, hd, qd, ctx->d_positions, l->n_rot, l->rope_base, ctx->d_rope_freqs[il], l->is_full);
        }
    }

    /* KV cache store */
    if (!l->share_kv) {
        int kv_pos = model->current_pos;
        size_t kv_off = (size_t)il * ctx->max_ctx * G4_MAX_KV_DIM + (size_t)kv_pos * kvd;
        cudaMemcpyAsync(ctx->d_k_cache + kv_off, d_k_buf, (size_t)N * kvd * sizeof(float),
                       cudaMemcpyDeviceToDevice, compute);
        cudaMemcpyAsync(ctx->d_v_cache + kv_off, d_v_buf, (size_t)N * kvd * sizeof(float),
                       cudaMemcpyDeviceToDevice, compute);
    }

    /* Attention */
    static int h_kv_size[48] = {0};
    if (h_kv_size[il] == 0) h_kv_size[il] = N;
    int kv_size = h_kv_size[il];

    if (l->is_full) {
        dim3 grid(N, G4_HEADS);
        size_t shm = (size_t)(kv_size > 0 ? kv_size : 1024) * sizeof(float);
        int threads = hd < 128 ? hd : 128;
        gpu_full_attn_kernel<<<grid, threads, shm, compute>>>(
            d_q_buf, ctx->d_k_cache, ctx->d_v_cache, d_a_buf,
            N, G4_HEADS, kvh, hd, ctx->d_positions, kv_size);
    } else {
        dim3 grid(N, G4_HEADS);
        size_t shm = (size_t)G4_SLIDING_WINDOW * sizeof(float);
        int threads = hd < 128 ? hd : 128;
        gpu_sliding_attn_kernel<<<grid, threads, shm, compute>>>(
            d_q_buf, ctx->d_k_cache, ctx->d_v_cache, d_a_buf,
            N, G4_HEADS, kvh, hd, G4_SLIDING_WINDOW, ctx->d_positions, kv_size);
    }
    h_kv_size[il] = kv_size + N;

    /* Output projection */
    gpu_matmul(compute, cublas, d_s_buf, d_wo, d_a_buf, N, G4_HIDDEN, qd, d_n, l->attn_out.data, l->attn_out.ggml_type);

    /* Post-attention norm + residual */
    gpu_rms_norm_kernel<<<N, 256, 0, compute>>>(d_n, ctx->d_pnorm[il], d_n, G4_HIDDEN, N, G4_RMS_EPS);
    gpu_add_kernel<<<(N*G4_HIDDEN+255)/256, 256, 0, compute>>>(d_x, d_n, d_x, N*G4_HIDDEN);

    /* FFN */
    gpu_rms_norm_kernel<<<N, 256, 0, compute>>>(d_x, ctx->d_fnorm[il], d_n, G4_HIDDEN, N, G4_RMS_EPS);
    gpu_matmul(compute, cublas, d_s_buf, d_wg, d_n, N, G4_FFN, G4_HIDDEN, d_g_buf, l->ffn_gate.data, l->ffn_gate.ggml_type);
    gpu_matmul(compute, cublas, d_s_buf, d_wu, d_n, N, G4_FFN, G4_HIDDEN, d_u_buf, l->ffn_up.data, l->ffn_up.ggml_type);
    gpu_gelu_kernel<<<(N*G4_FFN+255)/256, 256, 0, compute>>>(d_g_buf, N*G4_FFN);
    gpu_mul_kernel<<<(N*G4_FFN+255)/256, 256, 0, compute>>>(d_u_buf, d_g_buf, d_u_buf, N*G4_FFN);
    gpu_matmul(compute, cublas, d_s_buf, d_wd, d_u_buf, N, G4_HIDDEN, G4_FFN, d_n, l->ffn_down.data, l->ffn_down.ggml_type);

    /* Post-FFN norm + residual */
    gpu_rms_norm_kernel<<<N, 256, 0, compute>>>(d_n, ctx->d_pfnorm[il], d_n, G4_HIDDEN, N, G4_RMS_EPS);
    gpu_add_kernel<<<(N*G4_HIDDEN+255)/256, 256, 0, compute>>>(d_x, d_n, d_x, N*G4_HIDDEN);

    if (l->has_out_scale) {
        gpu_scale_kernel<<<(N*G4_HIDDEN+255)/256, 256, 0, compute>>>(d_x, l->layer_out_scale, N*G4_HIDDEN);
    }
}

/* ============= Full GPU forward with triple-buffer pipeline ============= */

int g4_model_forward_gpu(g4_gpu_ctx_t *ctx, void *model_ptr,
                          const float *embeddings, int B, int T, float *logits) {
    g4_model_t *model = (g4_model_t*)model_ptr;
    const int N = B * T;
    cudaStream_t compute = ctx->stream[0];
    cublasHandle_t cublas = ctx->cublas;

    ensure_weights_uploaded(ctx, model);

    /* Upload hidden states */
    float *h_x = (float*)malloc((size_t)N * G4_HIDDEN * sizeof(float));
    memcpy(h_x, embeddings, (size_t)N * G4_HIDDEN * sizeof(float));
    cudaMemcpyAsync(ctx->d_hidden, h_x, (size_t)N * G4_HIDDEN * sizeof(float),
                   cudaMemcpyHostToDevice, compute);

    int *h_pos = (int*)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) h_pos[i] = model->current_pos + i;
    cudaMemcpyAsync(ctx->d_positions, h_pos, (size_t)N * sizeof(int),
                   cudaMemcpyHostToDevice, compute);
    cudaStreamSynchronize(compute);

    float *d_x = ctx->d_hidden;
    float *d_n = ctx->d_normed;
    float *d_q = ctx->d_q;
    float *d_k = ctx->d_k;
    float *d_v = ctx->d_v;
    float *d_a = ctx->d_attn_out;
    float *d_g = ctx->d_ffn_gate;
    float *d_u = ctx->d_ffn_up;
    float *d_s = ctx->d_scratch;

    /* Reset weight pool ONLY on first forward pass (reclaim all layer weight VRAM) */
    static int first_forward = 1;
    if (first_forward) {
        ctx->weight_pool_used = 0;
        for (int i = 0; i < model->n_layers; i++) {
            ctx->layer_vram[i].uploaded = 0;
        }
        first_forward = 0;
    }

    /* Triple-buffer pipeline:
     * For each layer, we:
     *   1. Wait for upload_done[buf] for this layer's weights
     *   2. Run compute on stream[0]
     *   3. On stream[1], upload weights for layer il+1
     *   4. On stream[2], prefetch file data for layer il+2
     *
     * We use a simple rotating buffer scheme:
     *   buf_idx = il % 3 for upload
     *   buf_idx = (il + 1) % 3 for prefetch
     *
     * Weights are uploaded ONCE at first forward pass, then persist in VRAM
     * across subsequent forward passes (weight pool is NOT reset).
     */

    for (int il = 0; il < model->n_layers; il++) {
        fprintf(stderr, "[G4_GPU] forward: layer %d\n", il);
        int upload_buf = il % G4_GPU_N_BUFFERS;

        /* Upload current layer weights (if not already in pool) */
        if (!ctx->layer_vram[il].uploaded) {
            /* Wait for this buffer to be free */
            release_buffer(ctx, upload_buf);
            ctx->ram_owner[upload_buf] = il;

            if (!stream_weights_to_vram(ctx, upload_buf, il, &model->layers[il], model)) {
                fprintf(stderr, "[G4_GPU] FATAL: failed to upload layer %d weights\n", il);
                free(h_x); free(h_pos);
                return 0;
            }
        }

        /* Wait for upload to complete before computing */
        cudaEventSynchronize(ctx->upload_done[upload_buf]);

        /* Forward this layer */
        forward_layer_gpu(ctx, model, il, d_x, d_n, d_q, d_k, d_v, d_a, d_g, d_u, d_s, N, h_pos);

        /* Prefetch next layer's weights on stream[2] (if not last layer) */
        int next_il = il + 1;
        if (next_il < model->n_layers && !ctx->layer_vram[next_il].uploaded) {
            int prefetch_buf = next_il % G4_GPU_N_BUFFERS;
            release_buffer(ctx, prefetch_buf);
            ctx->ram_owner[prefetch_buf] = next_il;
            /* Start async upload for next layer (will be waited on next iteration) */
            stream_weights_to_vram(ctx, prefetch_buf, next_il, &model->layers[next_il], model);
        }
    }

    /* Final norm + LM head */
    gpu_rms_norm_kernel<<<N, 256, 0, compute>>>(d_x, ctx->d_output_norm, d_n, G4_HIDDEN, N, G4_RMS_EPS);

    uint8_t *d_out_w = ctx->d_output_w ? ctx->d_output_w : ctx->d_token_embd;
    gpu_matmul_q4_0(compute, cublas, d_s, d_out_w, d_n, N, G4_VOCAB, G4_HIDDEN, ctx->d_logits);
    gpu_softcap_kernel<<<(N*G4_VOCAB+255)/256, 256, 0, compute>>>(ctx->d_logits, N*G4_VOCAB, G4_SOFTCAP);

    cudaMemcpyAsync(logits, ctx->d_logits, (size_t)N * G4_VOCAB * sizeof(float),
                   cudaMemcpyDeviceToHost, compute);
    cudaStreamSynchronize(compute);

    model->current_pos += N;
    free(h_x);
    free(h_pos);

    return 1;
}

/* ============= Flush remaining batch tokens ============= */

int g4_model_forward_gpu_flush(g4_gpu_ctx_t *ctx, void *model_ptr, float *logits) {
    g4_model_t *model = (g4_model_t*)model_ptr;
    if (ctx->batch_len == 0) return 1;

    /* Run batched prefill for remaining tokens */
    float *embd = (float*)malloc((size_t)ctx->batch_len * G4_HIDDEN * sizeof(float));
    int row_bytes = (int)(model->token_embd.raw_bytes / G4_VOCAB);
    for (int i = 0; i < ctx->batch_len; i++) {
        int tok = ctx->batch_buf[i];
        if (tok < 0 || tok >= G4_VOCAB) tok = 0;
        gguf_dequantize(model->token_embd.data + (size_t)tok * row_bytes,
                       model->token_embd.ggml_type, G4_HIDDEN,
                       embd + (size_t)i * G4_HIDDEN);
    }
    float scale = sqrtf((float)G4_HIDDEN);
    for (int i = 0; i < ctx->batch_len * G4_HIDDEN; i++) embd[i] *= scale;

    int ok = g4_model_forward_gpu(ctx, model_ptr, embd, 1, ctx->batch_len, logits);
    free(embd);
    ctx->batch_len = 0;
    return ok;
}

/* ============= Single-token decode with 64-token batch sync-up ============= */

int g4_model_decode_gpu(g4_gpu_ctx_t *ctx, void *model_ptr, int token, float *logits) {
    g4_model_t *model = (g4_model_t*)model_ptr;

    /* Add token to batch buffer */
    ctx->batch_buf[ctx->batch_len++] = token;

    /* If batch is full (64 tokens), run batched prefill */
    if (ctx->batch_len >= G4_GPU_MAX_BATCH) {
        return g4_model_forward_gpu_flush(ctx, model_ptr, logits);
    }

    /* Otherwise, run single-token decode (fast path for first 64 tokens) */
    float embd[G4_HIDDEN];
    int row_bytes = (int)(model->token_embd.raw_bytes / G4_VOCAB);
    if (token < 0 || token >= G4_VOCAB) token = 0;
    gguf_dequantize(model->token_embd.data + (size_t)token * row_bytes,
                   model->token_embd.ggml_type, G4_HIDDEN, embd);
    float scale = sqrtf((float)G4_HIDDEN);
    for (int i = 0; i < G4_HIDDEN; i++) embd[i] *= scale;

    return g4_model_forward_gpu(ctx, model_ptr, embd, 1, 1, logits);
}
