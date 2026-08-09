/**
 * wubu_model_gpu.cu — GPU-accelerated forward path for wubu_model_t.
 *
 * Provides weight upload, GQA forward, and cleanup — callable from C via
 * extern "C" functions. Compiled with nvcc, linked into gen_text_gpu.
 *
 * Two weight modes:
 *   F32 mode (default): dequant→F32→cuBLAS  (~1GB VRAM for 10 GQA layers)
 *   Quantized mode (future): keep Q5_K on GPU, dequant-on-fly kernel
 */
#include "wubu_model.h"
#include "wubu_backend.h"
#include "wubu_dims.h"
#include "cuda_kernels.h"
#include "bench.h"
#include "gguf_reader.h"
#include "gpu_quant_matmul.h"
#include "gpu_moe_kernel.h"
#include "gpu_ssm_recurrence.h"
#include "wubu_moe.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
// FP16 conversion kernels (defined in cuda_kernels.cu)
__global__ void f32_to_f16_kernel(const float *in, __half *out, int n);
__global__ void f16_to_f32_kernel(const __half *in, float *out, int n);
// Q4_0 KV cache kernels (defined in cuda_kernels.cu)
__global__ void dequant_q4_0_cache_kernel(int n_blocks, const uint8_t *blocks, __half *out);
__global__ void quant_q4_0_cache_kernel(int n, const __half *in, uint8_t *blocks);
// Fused Q4_0 decode attention (C=1)
void wubu_cuda_attn_q4_0_decode(cublasHandle_t handle, cudaStream_t stream,
    const float *d_Q_chunk, const void *d_K_q4, const void *d_V_q4,
    const float *d_gate_full, const float *d_output_w,
    float *d_out, float *d_scratch, void *d_hp_scratch,
    int T_cache, int attn_window);
// Q4_0 KV cache kernels (defined in cuda_kernels.cu)
__global__ void dequant_q4_0_cache_kernel(int n_blocks, const uint8_t *blocks, __half *out);
__global__ void quant_q4_0_cache_kernel(int n, const __half *in, uint8_t *blocks);
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

// ================================================================
// Per-layer GPU GQA weights (F32 dequantized for cuBLAS)
// ================================================================
typedef struct {
    float *d_attn_q;       // [D_MODEL, q_dim*2] — fused Q+gate
    float *d_attn_k;       // [D_MODEL, kv_dim]
    float *d_attn_v;       // [D_MODEL, kv_dim]
    float *d_attn_out_w;   // [q_dim, D_MODEL]
    float *d_q_norm_w;     // [HEAD_DIM]
    float *d_k_norm_w;     // [HEAD_DIM]
} gpu_gqa_layer_t;

// ================================================================
// GPU context — stored as opaque pointer in wubu_model_t
typedef struct {
    // CUDA handles
    cublasHandle_t handle;
    cudaStream_t   stream;
    bool initialized;

    // Dynamic model dimensions (from model)
    int d_model;
    int d_inner;       // VALUE_DIM
    int key_dim;       // KEY_DIM
    int conv_dim;      // CONV_DIM
    int conv_kernel;   // CONV_KERNEL
    int dt_rank;       // DT_RANK
    int ssm_k_heads;   // SSM_K_HEADS
    int ssm_v_heads;   // SSM_V_HEADS
    int ssm_d_state;   // SSM_D_STATE
    int gqa_q_heads;   // GQA_Q_HEADS
    int gqa_kv_heads;  // GQA_KV_HEADS
    int gqa_head_dim;  // GQA_HEAD_DIM
    int rotary_dim;
    int vocab_size;
    int mrope_sec0_pairs;
    int mrope_sec1_pairs;
    int mrope_sec2_pairs;

    // Weights — per-layer GQA (only 10 layers, not all 40)
    gpu_gqa_layer_t *gqa_weights;  // [n_layers], NULL for SSM layers
    int n_layers;
    int n_gqa_layers;
    int max_ctx;
    int chunk_sz;
    int attn_window;    // sliding window for GQA (0=full, 16384=1 tile)

    // Persistent GPU KV cache (one per GQA layer, growable, Q4_0/FP16)
    void **d_k_cache;  // [n_gqa_layers][cache_capacity, kv_dim] __half
    void **d_v_cache;
    int *cache_len;     // current fill length per layer
    int *cache_cap;     // current allocated capacity per layer

    // RoPE sin/cos table
    float *d_sincos;    // [max_ctx, ROTARY_DIM]

    // Scratch buffers (reusable, sized for one chunk)
    float *d_x;         // [chunk_sz, D_MODEL] input
    float *d_scr;       // [chunk_sz, q_dim*2] fused Q+gate
    float *d_ktmp;      // [chunk_sz, kv_dim] K projection
    float *d_vtmp;      // [chunk_sz, kv_dim] V projection
    float *d_gout;      // [chunk_sz, D_MODEL] GQA output
    float *d_score_scr; // chunked attn score scratch
    float *d_qtmp;      // [chunk_sz, q_dim] Q-contiguous buffer

    // SSM quantized weights (kept on GPU in native Q5_K/Q6_K format)
    // Per-layer quantized weight data on GPU (dynamic allocation)
    uint8_t **d_attn_qkv_q;   // [n_layers] Q5_K attn_qkv
    uint8_t **d_attn_gate_q;  // [n_layers] Q5_K attn_gate
    uint8_t **d_ssm_out_q;    // [n_layers] Q6_K ssm_out
    int *ssm_qkv_type;
    int *ssm_gate_type;
    int *ssm_out_type;
    int64_t *ssm_qkv_col_stride;   // bytes per column
    int64_t *ssm_gate_col_stride;
    int64_t *ssm_out_col_stride;
    // Small F32 weights
    float **d_ssm_beta;      // [D_MODEL, DT_RANK]
    float **d_ssm_alpha;
    float **d_ssm_dt_bias;   // [DT_RANK]
    float **d_ssm_a;         // [DT_RANK]
    float **d_ssm_conv1d;    // [CONV_KERNEL, CONV_DIM]
    float **d_ssm_norm;      // [SSM_D_STATE]
    // F32 dequantized SSM weights for cuBLAS matmul (correct row-major layout)
    float **d_qkv_f32;       // [D_MODEL, CONV_DIM] F32 dequant from Q5_K
    float **d_gate_f32;      // [D_MODEL, VALUE_DIM] F32 dequant from Q5_K
    float **d_out_f32;       // [VALUE_DIM, D_MODEL] F32 dequant from Q6_K
    // Pre-allocated SSM output buffers (avoid per-call alloc/free)
    float *d_ssm_qkv_out;       // [chunk_sz, CONV_DIM]
    float *d_ssm_z_out;         // [chunk_sz, VALUE_DIM]
    // Pre-allocated MoE buffers (avoid per-call alloc/free for expert weights)
    uint8_t *d_moe_gate;        // [8][gate_bytes_per_expert]
    uint8_t *d_moe_up;          // [8][up_bytes_per_expert]
    uint8_t *d_moe_down;        // [8][down_bytes_per_expert]
    float   *d_moe_x;           // [D_MODEL] pre-allocated MoE input buffer
    float   *d_moe_out;         // [8][D_MODEL]
    float   *d_moe_weights;     // [8]
    // SSM recurrence persistent state (per layer, updated in-place)
    float **d_ssm_state;        // [n_layers][V_HEADS][D_STATE][D_STATE]
    // SSM conv state on GPU (per layer, updated in-place)
    float **d_conv_state;       // [n_layers][CONV_KERNEL-1][CONV_DIM]
    // MoE expert cache (per layer, last-used 8 experts, avoids H2D on routing stability)
    int **moe_cache_eid;        // [n_layers][8] expert IDs
    uint8_t ***moe_cache_w;     // [n_layers][3][8] GPU ptrs for gate/up/down weight blobs
    // Each cache slot: 8 * expert_raw_size bytes per weight type
    // SSM recurrence temp buffers
    float *d_ssm_q_all;         // [V_HEADS][D_STATE]
    float *d_ssm_k_all;         // [V_HEADS][D_STATE]
    float *d_ssm_v_all;         // [V_HEADS][D_STATE]
    float *d_ssm_beta_arr;      // [V_HEADS]
    float *d_ssm_gate_arr;      // [V_HEADS]
        float *d_ssm_delta_out;     // [V_HEADS][D_STATE]
        // SSM full forward scratch buffer
        float *d_ssm_scratch;       // reusable scratch for conv1d/SiLU/split/L2norm/recurrence/gated_norm
        size_t  d_ssm_scratch_sz;   // allocation size in bytes
        // FP16 scratch for chunked attention (Q + score tile)
        __half *d_hp_scratch;       // [n_q * hd + ATTEN_TILE * C]

        // Q4_0 KV cache mode flag
        bool use_q4_0_kv_cache;
    } gpu_ctx_t;

// ================================================================
// Helpers
// ================================================================
static int count_gqa_layers(const wubu_model_t *model) {
    int n = 0;
    for (int i = 0; i < model->n_layers; i++)
        if (!model->layers[i].is_ssm) n++;
    return n;
}

static int gqa_layer_idx(const wubu_model_t *model, int l) {
    int idx = 0;
    for (int i = 0; i < l; i++)
        if (!model->layers[i].is_ssm) idx++;
    return idx;
}

// ================================================================
// RoPE precomputation (on device, matches infer_text_gpu pattern)
// ================================================================
static void precompute_rotary_host(float *h_sc, int maxT) {
    for (int p = 0; p < maxT; p++) {
        for (int i = 0; i < ROTARY_DIM / 2; i++) {
            // MRoPE sections: [11, 11, 10, 0] — each section restarts freq
            int pair = i;
            int offset_in_section;
            if (pair < 11) {
                offset_in_section = pair;                    // section 0: pairs 0..10
            } else if (pair < 22) {
                offset_in_section = pair - 11;               // section 1: pairs 11..21
            } else {
                offset_in_section = pair - 22;               // section 2: pairs 22..31
            }
            double freq = pow(ROPE_THETA, -2.0 * offset_in_section / ROTARY_DIM);
            double angle = (double)p * freq;
            h_sc[p * ROTARY_DIM + i * 2]     = (float)sin(angle);
            h_sc[p * ROTARY_DIM + i * 2 + 1] = (float)cos(angle);
        }
    }
}

extern "C" int g_use_gpu_backend;

// ================================================================
// Init: upload GQA weights, create KV cache, allocate scratch
// ================================================================
extern "C"
int wubu_model_gpu_init(wubu_model_t *model, int max_ctx, int chunk_sz) {
    if (model->gpu_ctx) {
        fprintf(stderr, "GPU: already initialized\n");
        return 1;
    }

    gpu_ctx_t *gpu = (gpu_ctx_t *)calloc(1, sizeof(gpu_ctx_t));
    if (!gpu) return 0;

    // Copy model dimensions to GPU context
    gpu->d_model = WUBU_DIMS.d_model;
    gpu->d_inner = WUBU_DIMS.value_dim;
    gpu->key_dim = WUBU_DIMS.key_dim;
    gpu->conv_dim = WUBU_DIMS.conv_dim;
    gpu->conv_kernel = WUBU_DIMS.conv_kernel;
    gpu->dt_rank = WUBU_DIMS.dt_rank;
    gpu->ssm_k_heads = WUBU_DIMS.ssm_k_heads;
    gpu->ssm_v_heads = WUBU_DIMS.ssm_v_heads;
    gpu->ssm_d_state = WUBU_DIMS.ssm_d_state;
    gpu->gqa_q_heads = WUBU_DIMS.gqa_q_heads;
    gpu->gqa_kv_heads = WUBU_DIMS.gqa_kv_heads;
    gpu->gqa_head_dim = WUBU_DIMS.gqa_head_dim;
    gpu->rotary_dim = WUBU_DIMS.rope_head_dim;
    gpu->vocab_size = model->vocab_size;

    // CUDA context
    cudaError_t ce = cudaSetDevice(0);
    if (ce != cudaSuccess) { free(gpu); return 0; }
    cublasCreate(&gpu->handle);
    cudaStreamCreate(&gpu->stream);
    cublasSetStream(gpu->handle, gpu->stream);
    // Enable TF32
    cublasSetMathMode(gpu->handle, CUBLAS_TF32_TENSOR_OP_MATH);

    // Upload IQ1_S grid table to GPU constant memory (needed for IQ1_M dequant)
    const uint64_t *grid = gguf_get_iq1s_grid();
    if (grid) wubu_cuda_quant_matmul_set_iq1s_grid(grid);

    gpu->n_layers = model->n_layers;
    gpu->n_gqa_layers = count_gqa_layers(model);
    gpu->max_ctx = max_ctx;
    gpu->chunk_sz = chunk_sz;
    gpu->attn_window = 0;  // 0 = full attention (no sliding window by default), set via env GQA_WINDOW

    // Q4_0 KV cache mode: 4:1 compression. Toggle via GPU_Q4_0_KV=0 for FP16.
    int use_q4_0 = 1;
    const char *q4_0_env = getenv("GPU_Q4_0_KV");
    if (q4_0_env && atoi(q4_0_env) == 0) use_q4_0 = 0;
    gpu->use_q4_0_kv_cache = (use_q4_0 != 0);
    printf("GPU: KV cache mode: %s (4:1 compression vs FP16)\n",
           gpu->use_q4_0_kv_cache ? "Q4_0" : "FP16");

    printf("GPU: model dims: D_MODEL=%d D_INNER=%d KEY_DIM=%d CONV_DIM=%d CONV_KERNEL=%d DT_RANK=%d\n",
           gpu->d_model, gpu->d_inner, gpu->key_dim, gpu->conv_dim, gpu->conv_kernel, gpu->dt_rank);
    printf("GPU: SSM: K_HEADS=%d V_HEADS=%d D_STATE=%d\n",
           gpu->ssm_k_heads, gpu->ssm_v_heads, gpu->ssm_d_state);
    printf("GPU: GQA: Q_HEADS=%d KV_HEADS=%d HEAD_DIM=%d ROTARY_DIM=%d\n",
           gpu->gqa_q_heads, gpu->gqa_kv_heads, gpu->gqa_head_dim, gpu->rotary_dim);

    // Allocate dynamic arrays for per-layer weights
    gpu->gqa_weights = (gpu_gqa_layer_t *)calloc(model->n_layers, sizeof(gpu_gqa_layer_t));
    gpu->d_attn_qkv_q = (uint8_t**)calloc(model->n_layers, sizeof(uint8_t*));
    gpu->d_attn_gate_q = (uint8_t**)calloc(model->n_layers, sizeof(uint8_t*));
    gpu->d_ssm_out_q = (uint8_t**)calloc(model->n_layers, sizeof(uint8_t*));
    gpu->ssm_qkv_type = (int*)calloc(model->n_layers, sizeof(int));
    gpu->ssm_gate_type = (int*)calloc(model->n_layers, sizeof(int));
    gpu->ssm_out_type = (int*)calloc(model->n_layers, sizeof(int));
    gpu->ssm_qkv_col_stride = (int64_t*)calloc(model->n_layers, sizeof(int64_t));
    gpu->ssm_gate_col_stride = (int64_t*)calloc(model->n_layers, sizeof(int64_t));
    gpu->ssm_out_col_stride = (int64_t*)calloc(model->n_layers, sizeof(int64_t));
    gpu->d_ssm_beta = (float**)calloc(model->n_layers, sizeof(float*));
    gpu->d_ssm_alpha = (float**)calloc(model->n_layers, sizeof(float*));
    gpu->d_ssm_dt_bias = (float**)calloc(model->n_layers, sizeof(float*));
    gpu->d_ssm_a = (float**)calloc(model->n_layers, sizeof(float*));
    gpu->d_ssm_conv1d = (float**)calloc(model->n_layers, sizeof(float*));
    gpu->d_ssm_norm = (float**)calloc(model->n_layers, sizeof(float*));
    gpu->d_qkv_f32 = (float**)calloc(model->n_layers, sizeof(float*));
    gpu->d_gate_f32 = (float**)calloc(model->n_layers, sizeof(float*));
    gpu->d_out_f32 = (float**)calloc(model->n_layers, sizeof(float*));

        if (!gpu->gqa_weights || !gpu->d_attn_qkv_q || !gpu->d_attn_gate_q || !gpu->d_ssm_out_q ||
            !gpu->ssm_qkv_type || !gpu->ssm_gate_type || !gpu->ssm_out_type ||
            !gpu->ssm_qkv_col_stride || !gpu->ssm_gate_col_stride || !gpu->ssm_out_col_stride ||
            !gpu->d_ssm_beta || !gpu->d_ssm_alpha || !gpu->d_ssm_dt_bias || !gpu->d_ssm_a ||
            !gpu->d_ssm_conv1d || !gpu->d_ssm_norm || !gpu->d_qkv_f32 || !gpu->d_gate_f32 || !gpu->d_out_f32) {
            fprintf(stderr, "GPU: allocation failed for dynamic arrays\n");
            wubu_model_gpu_free(model);
            return 0;
        }

        // Find GGUF context with buffered data
        gguf_ctx *ctx = model->gguf_ctx;
        if (!ctx || !ctx->data_blob) {
            fprintf(stderr, "GPU: no GGUF context with buffered data\n");
            wubu_model_gpu_free(model);
            return 0;
        }

        printf("GPU: uploading %d GQA layer weights...\n", gpu->n_gqa_layers);

        const int q_dim = gpu->gqa_q_heads * gpu->gqa_head_dim;
        const int q_dim_x2 = q_dim * 2;
        const int kv_dim = gpu->gqa_kv_heads * gpu->gqa_head_dim;

        for (int l = 0; l < model->n_layers; l++) {
            if (model->layers[l].is_ssm) {
                gpu->gqa_weights[l].d_attn_q = NULL;
                continue;
            }

            gpu_gqa_layer_t *gw = &gpu->gqa_weights[l];
            float *h_buf;
            gguf_tensor_info *t;
            char name[256];

            // Allocate device memory using dynamic dimensions
            gw->d_attn_q     = wubu_cuda_alloc((size_t)gpu->d_model * q_dim_x2 * sizeof(float));
            gw->d_attn_k     = wubu_cuda_alloc((size_t)gpu->d_model * kv_dim * sizeof(float));
            gw->d_attn_v     = wubu_cuda_alloc((size_t)gpu->d_model * kv_dim * sizeof(float));
            gw->d_attn_out_w = wubu_cuda_alloc((size_t)q_dim * gpu->d_model * sizeof(float));
            gw->d_q_norm_w   = wubu_cuda_alloc((size_t)gpu->gqa_head_dim * sizeof(float));
            gw->d_k_norm_w   = wubu_cuda_alloc((size_t)gpu->gqa_head_dim * sizeof(float));

            if (!gw->d_attn_q || !gw->d_attn_k || !gw->d_attn_v ||
                !gw->d_attn_out_w || !gw->d_q_norm_w || !gw->d_k_norm_w) {
                fprintf(stderr, "GPU: allocation failed for layer %d\n", l);
                wubu_model_gpu_free(model);
                return 0;
            }

            // attn_q.weight — fused Q+gate
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); wubu_model_gpu_free(model); return 0; }
            h_buf = (float *)malloc((size_t)gpu->d_model * q_dim_x2 * sizeof(float));
            gguf_read_tensor_f32(ctx, t, h_buf, gpu->d_model * q_dim_x2);
            wubu_cuda_to_device(h_buf, gw->d_attn_q, (size_t)gpu->d_model * q_dim_x2 * sizeof(float), gpu->stream);
            free(h_buf);

            // attn_k.weight
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); wubu_model_gpu_free(model); return 0; }
            h_buf = (float *)malloc((size_t)gpu->d_model * kv_dim * sizeof(float));
            gguf_read_tensor_f32(ctx, t, h_buf, gpu->d_model * kv_dim);
            wubu_cuda_to_device(h_buf, gw->d_attn_k, (size_t)gpu->d_model * kv_dim * sizeof(float), gpu->stream);
            free(h_buf);

            // attn_v.weight
            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); wubu_model_gpu_free(model); return 0; }
            h_buf = (float *)malloc((size_t)gpu->d_model * kv_dim * sizeof(float));
            gguf_read_tensor_f32(ctx, t, h_buf, gpu->d_model * kv_dim);
            wubu_cuda_to_device(h_buf, gw->d_attn_v, (size_t)gpu->d_model * kv_dim * sizeof(float), gpu->stream);
            free(h_buf);

            // attn_output.weight
            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); wubu_model_gpu_free(model); return 0; }
            h_buf = (float *)malloc((size_t)q_dim * gpu->d_model * sizeof(float));
            gguf_read_tensor_f32(ctx, t, h_buf, q_dim * gpu->d_model);
            wubu_cuda_to_device(h_buf, gw->d_attn_out_w, (size_t)q_dim * gpu->d_model * sizeof(float), gpu->stream);
            free(h_buf);

            // attn_q_norm.weight
            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); wubu_model_gpu_free(model); return 0; }
            h_buf = (float *)malloc((size_t)gpu->gqa_head_dim * sizeof(float));
            gguf_read_tensor_f32(ctx, t, h_buf, gpu->gqa_head_dim);
            wubu_cuda_to_device(h_buf, gw->d_q_norm_w, (size_t)gpu->gqa_head_dim * sizeof(float), gpu->stream);
            free(h_buf);

            // attn_k_norm.weight
            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); wubu_model_gpu_free(model); return 0; }
            h_buf = (float *)malloc((size_t)gpu->gqa_head_dim * sizeof(float));
            gguf_read_tensor_f32(ctx, t, h_buf, gpu->gqa_head_dim);
            wubu_cuda_to_device(h_buf, gw->d_k_norm_w, (size_t)gpu->gqa_head_dim * sizeof(float), gpu->stream);
            free(h_buf);
        }
        cudaStreamSynchronize(gpu->stream);
        printf("GPU: GQA weights uploaded\n");

    // Persistent GPU KV cache (one per GQA layer, growable, Q4_0)
    // Each layer: grows from KV_CACHE_INIT up to max_ctx.
    // Q4_0 format: 18 bytes per 32 elements = 0.5625 bytes/elem
    // vs FP16: 2 bytes/elem. 4:1 compression.
    const int kv_cache_init = 4096;
    gpu->d_k_cache = (void**)calloc(model->n_layers, sizeof(void*));
    gpu->d_v_cache = (void**)calloc(model->n_layers, sizeof(void*));
    gpu->cache_len = (int *)calloc(model->n_layers, sizeof(int));
    gpu->cache_cap = (int *)calloc(model->n_layers, sizeof(int));
    if (!gpu->d_k_cache || !gpu->d_v_cache || !gpu->cache_len || !gpu->cache_cap) {
        wubu_model_gpu_free(model); return 0;
    }
    for (int l = 0; l < model->n_layers; l++) {
        if (model->layers[l].is_ssm) continue;
        int initial = (max_ctx < kv_cache_init) ? max_ctx : kv_cache_init;
        gpu->cache_cap[l] = initial;
        size_t cb;
        if (gpu->use_q4_0_kv_cache) {
            int64_t n_blocks = ((int64_t)initial * kv_dim + QK4_CACHE - 1) / QK4_CACHE;
            cb = (size_t)n_blocks * sizeof(block_q4_0_cache);
        } else {
            cb = (size_t)initial * kv_dim * sizeof(__half);
        }
        gpu->d_k_cache[l] = wubu_cuda_alloc(cb);
        gpu->d_v_cache[l] = wubu_cuda_alloc(cb);
        cudaMemset(gpu->d_k_cache[l], 0, cb);
        cudaMemset(gpu->d_v_cache[l], 0, cb);
        gpu->cache_len[l] = 0;
    }
    int64_t fp16_full_bytes = (int64_t)max_ctx * kv_dim * sizeof(__half) * 2 * gpu->n_gqa_layers;
    int64_t q4_0_full_bytes = ((int64_t)max_ctx * kv_dim + QK4_CACHE - 1) / QK4_CACHE * sizeof(block_q4_0_cache) * 2 * gpu->n_gqa_layers;
    printf("GPU: KV cache allocated (%d layers x %d init, max %d ctx, %s: %.0f MB vs %.0f MB FP16 full)\n",
           gpu->n_gqa_layers, kv_cache_init, max_ctx,
           gpu->use_q4_0_kv_cache ? "Q4_0" : "FP16",
           (double)q4_0_full_bytes / (1024*1024),
           (double)fp16_full_bytes / (1024*1024));

    // RoPE sin/cos table (host computed, device stored)
    float *h_sc = (float *)malloc((size_t)max_ctx * gpu->rotary_dim * sizeof(float));
    precompute_rotary_host(h_sc, max_ctx);
    gpu->d_sincos = wubu_cuda_alloc((size_t)max_ctx * gpu->rotary_dim * sizeof(float));
    wubu_cuda_to_device(h_sc, gpu->d_sincos, (size_t)max_ctx * gpu->rotary_dim * sizeof(float), gpu->stream);
    free(h_sc);

    // Scratch buffers
    gpu->d_x = wubu_cuda_alloc((size_t)chunk_sz * gpu->d_model * sizeof(float));
    gpu->d_scr = wubu_cuda_alloc((size_t)chunk_sz * q_dim_x2 * sizeof(float));
    gpu->d_ktmp = wubu_cuda_alloc((size_t)chunk_sz * kv_dim * sizeof(float));
    gpu->d_vtmp = wubu_cuda_alloc((size_t)chunk_sz * kv_dim * sizeof(float));
    gpu->d_gout = wubu_cuda_alloc((size_t)chunk_sz * gpu->d_model * sizeof(float));

    // Score scratch for chunked attention
    size_t score_bytes = wubu_cuda_chunked_attn_query_scratch(chunk_sz, max_ctx);
    gpu->d_score_scr = wubu_cuda_alloc(score_bytes);

    // FP16 scratch for chunked attention (Q + score tile)
    // Q4_0 mode needs extra space: 2 * ATTEN_TILE * kv_dim for dequantized K/V tiles
    // FP16 mode only needs: n_q * hd + ATTEN_TILE * C
    int hp_total;
    if (gpu->use_q4_0_kv_cache) {
        hp_total = gpu->gqa_q_heads * gpu->gqa_head_dim + 2 * 16384 * kv_dim + 16384 * chunk_sz;
    } else {
        hp_total = gpu->gqa_q_heads * gpu->gqa_head_dim + 16384 * chunk_sz;
    }
    gpu->d_hp_scratch = (__half*)wubu_cuda_alloc((size_t)hp_total * sizeof(__half));

    // Q-contiguous buffer: [chunk_sz, q_dim] floats
    gpu->d_qtmp = wubu_cuda_alloc((size_t)chunk_sz * q_dim * sizeof(float));

    // ============================================================
    // Upload SSM quantized weights to GPU (kept in native Q5_K/Q6_K format)
    // ============================================================
    printf("GPU: uploading SSM quantized weights (%d SSM layers)...\n", model->n_layers - gpu->n_gqa_layers);
    {
        int n_ssm_uploaded = 0;
        double total_mb = 0;
        for (int l = 0; l < model->n_layers; l++) {
            if (!model->layers[l].is_ssm) continue;
            wubu_layer_t *layer = &model->layers[l];
            gguf_tensor_info *t;
            char name[256];

            // Initialize to NULL for safe cleanup
            gpu->d_attn_qkv_q[l] = NULL;
            gpu->d_attn_gate_q[l] = NULL;
            gpu->d_ssm_out_q[l] = NULL;
            gpu->d_qkv_f32[l] = NULL;
            gpu->d_gate_f32[l] = NULL;
            gpu->d_out_f32[l] = NULL;

            // attn_qkv.weight (Q5_K)
            snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "SSM GPU: missing %s\n", name); continue; }
            int64_t qkv_n_elems = t->dims[0] * (t->n_dims > 1 ? t->dims[1] : 1);
            int64_t qkv_raw = gguf_raw_size(t->ggml_type, qkv_n_elems);
            gpu->ssm_qkv_type[l] = t->ggml_type;
            gpu->ssm_qkv_col_stride[l] = gguf_raw_size(t->ggml_type, t->dims[1] > 0 ? t->dims[1] : t->dims[0]);
            gpu->d_attn_qkv_q[l] = (uint8_t*)wubu_cuda_alloc((size_t)qkv_raw);
            cudaMemcpyAsync(gpu->d_attn_qkv_q[l], (const uint8_t*)ctx->data_blob + t->data_offset,
                           (size_t)qkv_raw, cudaMemcpyHostToDevice, gpu->stream);
            total_mb += qkv_raw / (1024.0 * 1024.0);
            // F32 dequant disabled (dead code — forward_full() uses quantized row_major kernel)
            // Saves ~2.2 GB VRAM. Struct fields d_qkv_f32[] preserved for init/free compatibility.
#if 0
            {
                float *h_f32 = (float*)malloc((size_t)qkv_n_elems * sizeof(float));
                int n_read = gguf_read_tensor_f32(ctx, t, h_f32, qkv_n_elems);
                if (h_f32 && n_read > 0) {
                    gpu->d_qkv_f32[l] = wubu_cuda_alloc((size_t)qkv_n_elems * sizeof(float));
                    wubu_cuda_to_device(h_f32, gpu->d_qkv_f32[l], (size_t)qkv_n_elems * sizeof(float), gpu->stream);
                    total_mb += (double)qkv_n_elems * sizeof(float) / (1024.0 * 1024.0);
                }
                free(h_f32);
            }
#endif

            // attn_gate.weight (Q5_K)
            snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "SSM GPU: missing %s\n", name); continue; }
            int64_t gate_n_elems = t->dims[0] * (t->n_dims > 1 ? t->dims[1] : 1);
            int64_t gate_raw = gguf_raw_size(t->ggml_type, gate_n_elems);
            gpu->ssm_gate_type[l] = t->ggml_type;
            gpu->ssm_gate_col_stride[l] = gguf_raw_size(t->ggml_type, t->dims[1] > 0 ? t->dims[1] : t->dims[0]);
            gpu->d_attn_gate_q[l] = (uint8_t*)wubu_cuda_alloc((size_t)gate_raw);
            cudaMemcpyAsync(gpu->d_attn_gate_q[l], (const uint8_t*)ctx->data_blob + t->data_offset,
                           (size_t)gate_raw, cudaMemcpyHostToDevice, gpu->stream);
            total_mb += gate_raw / (1024.0 * 1024.0);
            // F32 dequant disabled (dead code)
#if 0
            {
                float *hf = (float*)malloc((size_t)gate_n_elems * sizeof(float));
                if (hf && gguf_read_tensor_f32(ctx, t, hf, gate_n_elems) > 0) {
                    gpu->d_gate_f32[l] = wubu_cuda_alloc((size_t)gate_n_elems * sizeof(float));
                    wubu_cuda_to_device(hf, gpu->d_gate_f32[l], (size_t)gate_n_elems * sizeof(float), gpu->stream);
                    total_mb += (double)gate_n_elems * sizeof(float) / (1024.0 * 1024.0);
                }
                free(hf);
            }
#endif

            // ssm_out.weight (Q6_K)
            snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "SSM GPU: missing %s\n", name); continue; }
            int64_t out_n_elems = t->dims[0] * (t->n_dims > 1 ? t->dims[1] : 1);
            int64_t out_raw = gguf_raw_size(t->ggml_type, out_n_elems);
            gpu->ssm_out_type[l] = t->ggml_type;
            gpu->ssm_out_col_stride[l] = gguf_raw_size(t->ggml_type, t->dims[1] > 0 ? t->dims[1] : t->dims[0]);
            gpu->d_ssm_out_q[l] = (uint8_t*)wubu_cuda_alloc((size_t)out_raw);
            cudaMemcpyAsync(gpu->d_ssm_out_q[l], (const uint8_t*)ctx->data_blob + t->data_offset,
                           (size_t)out_raw, cudaMemcpyHostToDevice, gpu->stream);
            total_mb += out_raw / (1024.0 * 1024.0);
            // F32 dequant disabled (dead code)
#if 0
            {
                float *hf = (float*)malloc((size_t)out_n_elems * sizeof(float));
                if (hf && gguf_read_tensor_f32(ctx, t, hf, out_n_elems) > 0) {
                    gpu->d_out_f32[l] = wubu_cuda_alloc((size_t)out_n_elems * sizeof(float));
                    wubu_cuda_to_device(hf, gpu->d_out_f32[l], (size_t)out_n_elems * sizeof(float), gpu->stream);
                    total_mb += (double)out_n_elems * sizeof(float) / (1024.0 * 1024.0);
                }
                free(hf);
            }
#endif

            // Upload SSM small F32 weights
            {
                // ssm_beta.weight [d_model, dt_rank]
                snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", l);
                t = gguf_find_tensor(ctx, name);
                if (t) {
                    float *h_buf = (float*)malloc(gpu->d_model * gpu->dt_rank * sizeof(float));
                    if (h_buf && gguf_read_tensor_f32(ctx, t, h_buf, gpu->d_model * gpu->dt_rank) > 0) {
                        gpu->d_ssm_beta[l] = wubu_cuda_alloc(gpu->d_model * gpu->dt_rank * sizeof(float));
                        wubu_cuda_to_device(h_buf, gpu->d_ssm_beta[l], gpu->d_model * gpu->dt_rank * sizeof(float), gpu->stream);
                    }
                    free(h_buf);
                }

                // ssm_alpha.weight [d_model, dt_rank]
                snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", l);
                t = gguf_find_tensor(ctx, name);
                if (t) {
                    float *h_buf = (float*)malloc(gpu->d_model * gpu->dt_rank * sizeof(float));
                    if (h_buf && gguf_read_tensor_f32(ctx, t, h_buf, gpu->d_model * gpu->dt_rank) > 0) {
                        gpu->d_ssm_alpha[l] = wubu_cuda_alloc(gpu->d_model * gpu->dt_rank * sizeof(float));
                        wubu_cuda_to_device(h_buf, gpu->d_ssm_alpha[l], gpu->d_model * gpu->dt_rank * sizeof(float), gpu->stream);
                    }
                    free(h_buf);
                }

                // ssm_dt.bias [dt_rank]
                snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", l);
                t = gguf_find_tensor(ctx, name);
                if (t) {
                    float *h_buf = (float*)malloc(gpu->dt_rank * sizeof(float));
                    if (h_buf && gguf_read_tensor_f32(ctx, t, h_buf, gpu->dt_rank) > 0) {
                        gpu->d_ssm_dt_bias[l] = wubu_cuda_alloc(gpu->dt_rank * sizeof(float));
                        wubu_cuda_to_device(h_buf, gpu->d_ssm_dt_bias[l], gpu->dt_rank * sizeof(float), gpu->stream);
                    }
                    free(h_buf);
                }

                // ssm_a [dt_rank]
                snprintf(name, sizeof(name), "blk.%d.ssm_a", l);
                t = gguf_find_tensor(ctx, name);
                if (t) {
                    float *h_buf = (float*)malloc(gpu->dt_rank * sizeof(float));
                    if (h_buf && gguf_read_tensor_f32(ctx, t, h_buf, gpu->dt_rank) > 0) {
                        gpu->d_ssm_a[l] = wubu_cuda_alloc(gpu->dt_rank * sizeof(float));
                        wubu_cuda_to_device(h_buf, gpu->d_ssm_a[l], gpu->dt_rank * sizeof(float), gpu->stream);
                    }
                    free(h_buf);
                }

                // ssm_conv1d.weight [conv_kernel, conv_dim]
                snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", l);
                t = gguf_find_tensor(ctx, name);
                if (t) {
                    float *h_buf = (float*)malloc(gpu->conv_kernel * gpu->conv_dim * sizeof(float));
                    if (h_buf && gguf_read_tensor_f32(ctx, t, h_buf, gpu->conv_kernel * gpu->conv_dim) > 0) {
                        gpu->d_ssm_conv1d[l] = wubu_cuda_alloc(gpu->conv_kernel * gpu->conv_dim * sizeof(float));
                        wubu_cuda_to_device(h_buf, gpu->d_ssm_conv1d[l], gpu->conv_kernel * gpu->conv_dim * sizeof(float), gpu->stream);
                    }
                    free(h_buf);
                }

                // ssm_norm.weight [ssm_d_state]
                snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", l);
                t = gguf_find_tensor(ctx, name);
                if (t) {
                    float *h_buf = (float*)malloc(gpu->ssm_d_state * sizeof(float));
                    if (h_buf && gguf_read_tensor_f32(ctx, t, h_buf, gpu->ssm_d_state) > 0) {
                        gpu->d_ssm_norm[l] = wubu_cuda_alloc(gpu->ssm_d_state * sizeof(float));
                        wubu_cuda_to_device(h_buf, gpu->d_ssm_norm[l], gpu->ssm_d_state * sizeof(float), gpu->stream);
                    }
                    free(h_buf);
                }
            }

            n_ssm_uploaded++;
        }
        cudaStreamSynchronize(gpu->stream);
        printf("GPU: SSM quantized weights uploaded: %d layers, %.0f MB\n", n_ssm_uploaded, total_mb);
    cudaStreamSynchronize(gpu->stream);  // ensure all async uploads complete
    }

    cudaStreamSynchronize(gpu->stream);
    gpu->initialized = true;
    model->gpu_ctx = (void *)gpu;
    model->backend = wubu_backend_cuda_get();

    // Allocate SSM output buffers (reused per-layer, sized for max chunk)
    gpu->d_ssm_qkv_out = wubu_cuda_alloc((size_t)gpu->chunk_sz * gpu->conv_dim * sizeof(float));
    gpu->d_ssm_z_out   = wubu_cuda_alloc((size_t)gpu->chunk_sz * gpu->d_inner * sizeof(float));
    printf("GPU: SSM output buffers: %d x %dx%d floats\n", gpu->chunk_sz, gpu->conv_dim, gpu->d_inner);

    // Allocate SSM full forward scratch buffer
    // The wubu_cuda_ssm_forward uses a large scratch; compute needed size
    size_t ssm_scratch_needed = (size_t)wubu_cuda_ssm_forward_query_scratch(gpu->chunk_sz, 1,
        gpu->conv_dim, gpu->d_inner, gpu->dt_rank, gpu->conv_kernel);
    // Also need room for conv_input which depends on conv_kernel-1 overhead
    size_t ssm_scratch_extra = (size_t)gpu->chunk_sz * (gpu->conv_dim * 4 + gpu->d_inner * 4 + gpu->dt_rank * 8);
    gpu->d_ssm_scratch_sz = (ssm_scratch_needed > ssm_scratch_extra) ? ssm_scratch_needed : ssm_scratch_extra;
    gpu->d_ssm_scratch = wubu_cuda_alloc(gpu->d_ssm_scratch_sz);
    printf("GPU: SSM scratch buffer: %zu MB\n", gpu->d_ssm_scratch_sz / (1024*1024));

    // Initialize GPU MoE lookup tables (IQ2_XXS grid in constant memory)
    wubu_gpu_moe_init();
    printf("GPU: MoE lookup tables initialized\n");

    // Allocate MoE expert cache (n_layers × 8 experts × 3 weights)
    // Each expert weight: gate_bytes/up_bytes/down_bytes for IQ2_XXS/IQ3_XXS/IQ4_XS
    // The exact sizes vary by layer (down_exps type: IQ3_XXS or IQ4_XS)
    // We allocate the maximum possible to ensure uniform sizing
    // Max per-expert: ceil(d_model*d_ff/256) * IQ4_XS_BLOCK_SIZE = 4096 * 136 = 557056
    const int64_t moe_max_per_expert = 557056;  // max across all layers
    gpu->moe_cache_eid = (int**)calloc(model->n_layers, sizeof(int*));
    gpu->moe_cache_w = (uint8_t***)calloc(model->n_layers, sizeof(uint8_t**));
    int n_cached = 0;
    for (int l = 0; l < model->n_layers; l++) {
        moe_weights_t *moe = &model->layers[l].moe;
        if (!moe->loaded) continue;
        gpu->moe_cache_eid[l] = (int*)calloc(8, sizeof(int));
        for (int k = 0; k < 8; k++) gpu->moe_cache_eid[l][k] = -1;
        gpu->moe_cache_w[l] = (uint8_t**)calloc(3, sizeof(uint8_t*));
        gpu->moe_cache_w[l][0] = (uint8_t*)wubu_cuda_alloc((size_t)(8 * moe_max_per_expert));
        gpu->moe_cache_w[l][1] = (uint8_t*)wubu_cuda_alloc((size_t)(8 * moe_max_per_expert));
        gpu->moe_cache_w[l][2] = (uint8_t*)wubu_cuda_alloc((size_t)(8 * moe_max_per_expert));
        n_cached++;
    }
    printf("GPU: MoE expert cache allocated: %d layers × 8 experts × 3 weights\n", n_cached);

    // Allocate MoE persistent buffers (for 8 active experts)
    // 557056 = max per-expert raw size (IQ4_XS at 136 bytes/block × 4096 blocks)
    const int64_t moe_bytes = 557056;
    gpu->d_moe_gate    = (uint8_t*)wubu_cuda_alloc((size_t)(8 * moe_bytes));
    gpu->d_moe_up      = (uint8_t*)wubu_cuda_alloc((size_t)(8 * moe_bytes));
    gpu->d_moe_down    = (uint8_t*)wubu_cuda_alloc((size_t)(8 * moe_bytes));
    gpu->d_moe_x       = wubu_cuda_alloc((size_t)gpu->d_model * sizeof(float));
    gpu->d_moe_out     = wubu_cuda_alloc((size_t)(8 * gpu->d_model * sizeof(float)));
    gpu->d_moe_weights = wubu_cuda_alloc((size_t)(8 * sizeof(float)));
    printf("GPU: MoE buffers allocated (3x%dKB + %dKB)\n",
           (int)(8 * moe_bytes / 1024), (int)(8 * gpu->d_model * 4 / 1024));

    // Allocate SSM recurrence state: one [ssm_v_heads][ssm_d_state][ssm_d_state] per SSM layer
    gpu->d_ssm_state = (float**)calloc(gpu->n_layers, sizeof(float*));
    gpu->d_conv_state = (float**)calloc(gpu->n_layers, sizeof(float*));  // may be NULL for GQA layers
    for (int l = 0; l < gpu->n_layers; l++) {
        if (model->layers[l].is_ssm) {
            gpu->d_ssm_state[l] = wubu_cuda_alloc(
                (size_t)gpu->ssm_v_heads * gpu->ssm_d_state * gpu->ssm_d_state * sizeof(float));
            cudaMemsetAsync(gpu->d_ssm_state[l], 0,
                (size_t)gpu->ssm_v_heads * gpu->ssm_d_state * gpu->ssm_d_state * sizeof(float),
                gpu->stream);
            // Also allocate GPU conv state
            gpu->d_conv_state[l] = wubu_cuda_alloc(
                (size_t)(gpu->conv_kernel - 1) * gpu->conv_dim * sizeof(float));
            cudaMemsetAsync(gpu->d_conv_state[l], 0,
                (size_t)(gpu->conv_kernel - 1) * gpu->conv_dim * sizeof(float),
                gpu->stream);
        } else {
            gpu->d_ssm_state[l] = NULL;
            gpu->d_conv_state[l] = NULL;
        }
    }
    cudaStreamSynchronize(gpu->stream);
    printf("GPU: SSM state allocated: %d SSM layers × %d V_HEADS × %d D_STATE²\n",
           gpu->n_gqa_layers, gpu->ssm_v_heads, gpu->ssm_d_state);

    // Pre-allocate SSM recurrence temp buffers
    gpu->d_ssm_q_all = wubu_cuda_alloc((size_t)gpu->ssm_v_heads * gpu->ssm_d_state * sizeof(float));
    gpu->d_ssm_k_all = wubu_cuda_alloc((size_t)gpu->ssm_v_heads * gpu->ssm_d_state * sizeof(float));
    gpu->d_ssm_v_all = wubu_cuda_alloc((size_t)gpu->ssm_v_heads * gpu->ssm_d_state * sizeof(float));
    gpu->d_ssm_beta_arr = wubu_cuda_alloc((size_t)gpu->ssm_v_heads * sizeof(float));
    gpu->d_ssm_gate_arr = wubu_cuda_alloc((size_t)gpu->ssm_v_heads * sizeof(float));
    gpu->d_ssm_delta_out = wubu_cuda_alloc((size_t)gpu->ssm_v_heads * gpu->ssm_d_state * sizeof(float));
    printf("GPU: SSM temp buffers allocated\n");

    printf("GPU: SSM recurrence buffers allocated (%d heads × %d state)\n",
           gpu->ssm_v_heads, gpu->ssm_d_state);

    printf("GPU: init complete (1040.0 MB GQA weights)\n");
    const char *gqa_win = getenv("GQA_WINDOW");
    if (gqa_win) {
        gpu->attn_window = atoi(gqa_win);
        printf("GPU: sliding window attention active (window=%d tokens)\n", gpu->attn_window);
    }
    g_use_gpu_backend = 1;  /* signal wubu_ssm.c proj_matmul to use GPU quantized matmul */
    printf("GPU: GQA acceleration active (max_ctx=%d, chunk=%d)\n", max_ctx, chunk_sz);
    return 1;
}

// ================================================================
// GPU GQA Forward: process one GQA layer on GPU
// Input:  h_norm [C, D_MODEL] — already RMSNorm'd on CPU
// Output: h_attn [C, D_MODEL] — attention output (will be resid-added on CPU)
// Also appends K,V to persistent GPU KV cache.
// ================================================================
extern "C"
int wubu_model_gpu_chunk_sz(wubu_model_t *model) {
    gpu_ctx_t *gpu = (gpu_ctx_t *)model->gpu_ctx;
    return gpu ? gpu->chunk_sz : 0;
}

extern "C"
int wubu_model_gpu_gqa_forward(wubu_model_t *model, int layer_idx,
                                const float *h_norm, int C, float *h_attn) {
    gpu_ctx_t *gpu = (gpu_ctx_t *)model->gpu_ctx;
    if (!gpu || !gpu->initialized) return 0;
    if (model->layers[layer_idx].is_ssm) return 0;

    gpu_gqa_layer_t *gw = &gpu->gqa_weights[layer_idx];
    if (!gw->d_attn_q) return 0;

    cublasHandle_t ch = gpu->handle;
    cudaStream_t st = gpu->stream;
    const int q_dim = gpu->gqa_q_heads * gpu->gqa_head_dim;
    const int q_dim_x2 = q_dim * 2;
    const int kv_dim = gpu->gqa_kv_heads * gpu->gqa_head_dim;

    // Upload normed input to GPU
    cudaMemcpyAsync(gpu->d_x, h_norm, (size_t)C * gpu->d_model * sizeof(float),
                    cudaMemcpyHostToDevice, st);

    // === Q (fused Q+gate), K, V projections via cuBLAS ===
    wubu_cuda_matmul(ch, gpu->d_x, C, gpu->d_model, gw->d_attn_q, q_dim_x2, gpu->d_scr, 1.0f, 0.0f);
    wubu_cuda_matmul(ch, gpu->d_x, C, gpu->d_model, gw->d_attn_k, kv_dim, gpu->d_ktmp, 1.0f, 0.0f);
    wubu_cuda_matmul(ch, gpu->d_x, C, gpu->d_model, gw->d_attn_v, kv_dim, gpu->d_vtmp, 1.0f, 0.0f);

    // === Extract Q and gate from fused ===
    wubu_cuda_copy_q_from_fused(gpu->d_qtmp, gpu->d_scr, C, q_dim, st);
    wubu_cuda_copy_gate_from_fused(gpu->d_scr, gpu->d_scr, C, q_dim, st);

    // === RMSNorm Q and K ===
    wubu_cuda_rms_norm_heads(C * gpu->gqa_q_heads, gpu->gqa_head_dim,
        gpu->d_qtmp, gw->d_q_norm_w, 1e-6f, gpu->d_qtmp, st);
    wubu_cuda_rms_norm_heads(C * gpu->gqa_kv_heads, gpu->gqa_head_dim,
        gpu->d_ktmp, gw->d_k_norm_w, 1e-6f, gpu->d_ktmp, st);

    // === Apply RoPE to Q and K ===
    int cache_start = gpu->cache_len[layer_idx];
    wubu_cuda_apply_rotary_to_qk(gpu->d_qtmp, gpu->d_ktmp,
        C, C, gpu->gqa_q_heads, gpu->gqa_kv_heads, gpu->gqa_head_dim,
        gpu->d_sincos + (size_t)cache_start * gpu->rotary_dim, st);

    // === Append K, V to persistent cache (F32→Q4_0 or F32→FP16) ===
    int total_needed = cache_start + C;
    if (total_needed > gpu->cache_cap[layer_idx]) {
        // Grow cache: double capacity up to max_ctx
        int new_cap = gpu->cache_cap[layer_idx] * 2;
        if (new_cap > gpu->max_ctx) new_cap = gpu->max_ctx;
        if (new_cap < total_needed) new_cap = total_needed;
        size_t new_bytes, old_bytes;
        if (gpu->use_q4_0_kv_cache) {
            int64_t n_blocks_new = ((int64_t)new_cap * kv_dim + QK4_CACHE - 1) / QK4_CACHE;
            int64_t n_blocks_old = ((int64_t)gpu->cache_cap[layer_idx] * kv_dim + QK4_CACHE - 1) / QK4_CACHE;
            new_bytes = (size_t)n_blocks_new * sizeof(block_q4_0_cache);
            old_bytes = (size_t)n_blocks_old * sizeof(block_q4_0_cache);
        } else {
            new_bytes = (size_t)new_cap * kv_dim * sizeof(__half);
            old_bytes = (size_t)gpu->cache_cap[layer_idx] * kv_dim * sizeof(__half);
        }

        void *new_k = wubu_cuda_alloc(new_bytes);
        void *new_v = wubu_cuda_alloc(new_bytes);
        if (!new_k || !new_v) {
            fprintf(stderr, "GPU: KV cache grow failed for layer %d (%d->%d)\n",
                    layer_idx, gpu->cache_cap[layer_idx], new_cap);
            wubu_cuda_free((float*)new_k);
            wubu_cuda_free((float*)new_v);
            return 0;
        }
        cudaMemcpyAsync(new_k, gpu->d_k_cache[layer_idx], old_bytes,
                        cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(new_v, gpu->d_v_cache[layer_idx], old_bytes,
                        cudaMemcpyDeviceToDevice, st);
        wubu_cuda_free((float*)gpu->d_k_cache[layer_idx]);
        wubu_cuda_free((float*)gpu->d_v_cache[layer_idx]);
        gpu->d_k_cache[layer_idx] = new_k;
        gpu->d_v_cache[layer_idx] = new_v;
        gpu->cache_cap[layer_idx] = new_cap;
        fprintf(stderr, "GPU: KV cache layer %d grew to %d\n", layer_idx, new_cap);
    }

    int n_elems_k = C * kv_dim;
    int block = 256;

    if (gpu->use_q4_0_kv_cache) {
        // Q4_0 path: convert F32 K/V to Q4_0 blocks
        int n_q4_blocks = (n_elems_k + 256 - 1) / 256;
        int grid_f32 = (n_elems_k + block - 1) / block;

        // Convert F32 K to FP16 temp, then quantize to Q4_0 blocks
        f32_to_f16_kernel<<<grid_f32, block, 0, st>>>(
            gpu->d_ktmp, (__half*)gpu->d_score_scr, n_elems_k);
        quant_q4_0_cache_kernel<<<n_q4_blocks, block, sizeof(float) * 256, st>>>(
            n_elems_k, (__half*)gpu->d_score_scr,
            (uint8_t*)gpu->d_k_cache[layer_idx] + (size_t)cache_start * kv_dim / 256 * 18);

        // Convert F32 V to FP16 temp, then quantize to Q4_0 blocks
        f32_to_f16_kernel<<<grid_f32, block, 0, st>>>(
            gpu->d_vtmp, (__half*)gpu->d_score_scr, n_elems_k);
        quant_q4_0_cache_kernel<<<n_q4_blocks, block, sizeof(float) * 256, st>>>(
            n_elems_k, (__half*)gpu->d_score_scr,
            (uint8_t*)gpu->d_v_cache[layer_idx] + (size_t)cache_start * kv_dim / 256 * 18);

        // Update cache length
        gpu->cache_len[layer_idx] = cache_start + C;

        // === Chunked attention (Q4_0 KV cache) ===
        if (C == 1) {
            // Fused decode kernel: Q4_0 K→scores directly, no FP16 scratch
            wubu_cuda_attn_q4_0_decode(ch, st, gpu->d_qtmp,
                gpu->d_k_cache[layer_idx], gpu->d_v_cache[layer_idx],
                gpu->d_scr, gw->d_attn_out_w,
                gpu->d_gout, gpu->d_score_scr, gpu->d_hp_scratch,
                cache_start + C, gpu->attn_window);
        } else {
            wubu_cuda_chunked_attn_q4_0(ch, st, C, cache_start + C,
            gpu->d_qtmp, gpu->d_k_cache[layer_idx], gpu->d_v_cache[layer_idx],
            gpu->d_scr, gw->d_attn_out_w, gpu->d_gout, gpu->d_score_scr,
            gpu->d_hp_scratch, gpu->attn_window);
        }
    } else {
        // FP16 path: convert F32 K→FP16, write to cache
        int grid = (n_elems_k + block - 1) / block;
        f32_to_f16_kernel<<<grid, block, 0, st>>>(gpu->d_ktmp,
            (__half*)gpu->d_score_scr, n_elems_k);
        cudaMemcpyAsync((__half*)gpu->d_k_cache[layer_idx] + (size_t)cache_start * kv_dim,
                        gpu->d_score_scr, (size_t)n_elems_k * sizeof(__half),
                        cudaMemcpyDeviceToDevice, st);

        // Convert F32 V→FP16, write to cache
        f32_to_f16_kernel<<<grid, block, 0, st>>>(gpu->d_vtmp,
            (__half*)gpu->d_score_scr, n_elems_k);
        cudaMemcpyAsync((__half*)gpu->d_v_cache[layer_idx] + (size_t)cache_start * kv_dim,
                        gpu->d_score_scr, (size_t)n_elems_k * sizeof(__half),
                        cudaMemcpyDeviceToDevice, st);

        // Update cache length
        gpu->cache_len[layer_idx] = cache_start + C;

        // === Chunked attention (FP16 KV cache) with optional sliding window ===
        wubu_cuda_chunked_attn_fp16(ch, st, C, cache_start + C,
            gpu->d_qtmp, gpu->d_k_cache[layer_idx], gpu->d_v_cache[layer_idx],
            gpu->d_scr, gw->d_attn_out_w, gpu->d_gout, gpu->d_score_scr,
            gpu->d_hp_scratch, gpu->attn_window);
    }

    // Download output back to CPU
    cudaMemcpyAsync(h_attn, gpu->d_gout, (size_t)C * D_MODEL * sizeof(float),
                    cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);

    return 1;
}

// ================================================================
// GPU SSM hybrid forward: 3 quantized matmuls on GPU, rest on CPU.
// Uploads x, runs Q5_K/Q6_K matmuls, downloads results.
// Uses pre-allocated d_ssm_qkv_out/d_ssm_z_out buffers.
// ================================================================
extern "C"
int wubu_model_gpu_ssm_project(wubu_model_t *model, int layer_idx,
                                const float *h_norm, int C,
                                float *qkv_out, float *z_out,
                                float *ssm_out_out) {
    gpu_ctx_t *gpu = (gpu_ctx_t *)model->gpu_ctx;
    if (!gpu || !gpu->initialized) return 0;
    if (!model->layers[layer_idx].is_ssm) return 0;
    if (model->layers[layer_idx].ssm.attn_qkv_weight_q == NULL) return 0;

    cudaStream_t st = gpu->stream;

    // Upload input to GPU
    cudaMemcpyAsync(gpu->d_x, h_norm, (size_t)C * D_MODEL * sizeof(float),
                    cudaMemcpyHostToDevice, st);

    // === attn_qkv: quantized matmul (batched) ===
    // x [C, D_MODEL] @ W_qkv [D_MODEL, CONV_DIM] → d_qkv [C, CONV_DIM]
    wubu_cuda_quant_matmul_batched(gpu->d_x, C,
        gpu->d_attn_qkv_q[layer_idx], gpu->ssm_qkv_type[layer_idx],
        D_MODEL, CONV_DIM, gpu->d_ssm_qkv_out, st);

    // === attn_gate: quantized matmul (batched) ===
    // x [C, D_MODEL] @ W_gate [D_MODEL, VALUE_DIM] → d_z [C, VALUE_DIM]
    wubu_cuda_quant_matmul_batched(gpu->d_x, C,
        gpu->d_attn_gate_q[layer_idx], gpu->ssm_gate_type[layer_idx],
        D_MODEL, VALUE_DIM, gpu->d_ssm_z_out, st);

    // Both matmuls are enqueued on the same stream, so they run sequentially.
    // One sync, then download both results.
    cudaStreamSynchronize(st);

    cudaMemcpyAsync(qkv_out, gpu->d_ssm_qkv_out, (size_t)C * CONV_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, st);
    cudaMemcpyAsync(z_out, gpu->d_ssm_z_out, (size_t)C * VALUE_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, st);

    // ssm_out projection not yet implemented on GPU (handled on CPU for now)
    (void)ssm_out_out;

    cudaStreamSynchronize(st);
    return 1;
}

// GPU MoE expert forward: replaces 8 expert matmuls with GPU quantized kernel
// Input: x_s [D_MODEL], top-8 expert indices + weights
// Uses per-layer expert cache: on cache hit, reads weights directly from GPU
// (no H2D transfer). On cache miss, uploads and updates cache.
// Shared expert and router remain on CPU.
extern "C"
void wubu_model_gpu_moe_experts(
    const moe_weights_t *w,
    const float *x_s,
    const int *indices_s,        // [8] expert indices
    const float *weights_s,      // [8] routing weights
    float *expert_contribs,
    void *model_ptr)             // wubu_model_t* for CUDA stream access
{
    if (!w->ffn_gate_exps_q) return;
    // Get CUDA stream from model's GPU context
    wubu_model_t *model = (wubu_model_t *)model_ptr;
    gpu_ctx_t *gpu = (gpu_ctx_t *)model->gpu_ctx;
    if (!gpu || !gpu->initialized) return;
    cudaStream_t stream = gpu->stream;

    static int printed_types = 0;
    if (!printed_types) { printed_types = 1;
        fprintf(stderr, "GPU MoE types: gate=%d up=%d down=%d\n",
                w->ffn_gate_exps_q_type, w->ffn_up_exps_q_type, w->ffn_down_exps_q_type);
    }
 
     // Compute per-expert byte sizes
    int64_t gate_bytes = gguf_raw_size(w->ffn_gate_exps_q_type, (int64_t)D_MODEL * D_FF);
    int64_t up_bytes   = gguf_raw_size(w->ffn_up_exps_q_type,   (int64_t)D_MODEL * D_FF);
    int64_t down_bytes = gguf_raw_size(w->ffn_down_exps_q_type, (int64_t)D_FF * D_MODEL);

    // Determine current layer index (for cache lookup)
    // Walk through model->layers to find which layer owns these MoE weights
    int layer_idx = -1;
    for (int l = 0; l < model->n_layers; l++) {
        if (&model->layers[l].moe == w) { layer_idx = l; break; }
    }

    // Prepare 8 expert weight data pointers (GPU if cached, host if not)
    const uint8_t *gate_q_ptrs[8], *up_q_ptrs[8], *down_q_ptrs[8];
    float wgts[8];
    int n_active = 0;
    bool use_gpu_ptrs = false;

    // Check if this layer has a cache entry
    if (layer_idx >= 0 && gpu->moe_cache_eid && gpu->moe_cache_eid[layer_idx]) {
        int *cached = gpu->moe_cache_eid[layer_idx];
        bool all_match = true;
        for (int k = 0; k < 8; k++) {
            if (cached[k] != indices_s[k]) { all_match = false; break; }
        }
        if (all_match && cached[0] != -1) {
            use_gpu_ptrs = true;  // cache hit: data already on GPU
        }
    }

    for (int k = 0; k < 8; k++) {
        int e = indices_s[k];
        if (e < 0 || weights_s[k] < 1e-30f) {
            memset(expert_contribs + (size_t)k * D_MODEL, 0, D_MODEL * sizeof(float));
            continue;
        }
        if (use_gpu_ptrs && layer_idx >= 0) {
            // Cache hit: use GPU pointers directly (no H2D)
            uint8_t **cache_layer = gpu->moe_cache_w[layer_idx];
            gate_q_ptrs[n_active] = cache_layer[0] + (size_t)k * gate_bytes;
            up_q_ptrs[n_active]   = cache_layer[1] + (size_t)k * up_bytes;
            down_q_ptrs[n_active] = cache_layer[2] + (size_t)k * down_bytes;
        } else {
            // Cache miss: use host pointers (will be uploaded by kernel func)
            gate_q_ptrs[n_active] = w->ffn_gate_exps_q + (int64_t)e * gate_bytes;
            up_q_ptrs[n_active]   = w->ffn_up_exps_q   + (int64_t)e * up_bytes;
            down_q_ptrs[n_active] = w->ffn_down_exps_q + (int64_t)e * down_bytes;
        }
        wgts[n_active] = weights_s[k];
        n_active++;
    }

    if (n_active == 0) return;

    // Allocate temp output buffer on host (8 experts × D_MODEL)
    float *gpu_out = (float*)calloc(8 * D_MODEL, sizeof(float));
    if (!gpu_out) return;

    // Run GPU MoE — either with cached GPU pointers or uploaded host pointers
    wubu_gpu_moe_forward_experts(
        x_s,
        (const uint8_t**)gate_q_ptrs, gate_bytes,
        (const uint8_t**)up_q_ptrs, up_bytes,
        (const uint8_t**)down_q_ptrs, down_bytes,
        w->ffn_gate_exps_q_type, w->ffn_up_exps_q_type, w->ffn_down_exps_q_type,
        wgts, gpu_out, stream,
        gpu->d_moe_gate, gpu->d_moe_up, gpu->d_moe_down,
        gpu->d_moe_x, gpu->d_moe_out, gpu->d_moe_weights,
        use_gpu_ptrs);  // TRUE = pointers are GPU, skip H2D

    // On cache miss: update cache from the scratch buffers we just uploaded
    if (!use_gpu_ptrs && layer_idx >= 0 && gpu->moe_cache_w && gpu->moe_cache_w[layer_idx]) {
        int *cached = gpu->moe_cache_eid[layer_idx];
        uint8_t **cache_layer = gpu->moe_cache_w[layer_idx];
        for (int k = 0; k < 8; k++) {
            cached[k] = indices_s[k];
        }
        // The scratch buffers (d_moe_gate/d_moe_up/d_moe_down) now have the
        // uploaded data. D2D copy to cache buffers for persistence.
        cudaMemcpyAsync(cache_layer[0], gpu->d_moe_gate, (size_t)8 * gate_bytes,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(cache_layer[1], gpu->d_moe_up, (size_t)8 * up_bytes,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(cache_layer[2], gpu->d_moe_down, (size_t)8 * down_bytes,
                        cudaMemcpyDeviceToDevice, stream);
    }

    // Distribute output across per-expert contribution buffers
    int active_idx = 0;
    for (int k = 0; k < 8; k++) {
        int e = indices_s[k];
        if (e < 0 || weights_s[k] < 1e-30f) continue;
        memcpy(expert_contribs + (size_t)k * D_MODEL, gpu_out + active_idx * D_MODEL,
               D_MODEL * sizeof(float));
        active_idx++;
    }

    free(gpu_out);
}
// ================================================================
// GPU SSM full forward: ALL steps on GPU, only final output downloaded.
// Keeps qkv/gate/z on GPU after quantized matmuls, runs conv1d/SiLU/split/
// L2 norm/recurrence/gated norm/ssm_out projection entirely on device.
// For N=1 decode: H2D transfers for q/k/v/beta/gate (small, 8KB each).
// For C>1 prefill: token-by-token loop (each token is independent).
// ================================================================
extern "C"
int wubu_model_gpu_ssm_forward_full(wubu_model_t *model, int layer_idx,
                                     const float *h_norm, int C,
                                     float *h_attn_out) {
    gpu_ctx_t *gpu = (gpu_ctx_t *)model->gpu_ctx;
    if (!gpu || !gpu->initialized) return 0;
    if (!model->layers[layer_idx].is_ssm) return 0;
    if (model->layers[layer_idx].ssm.attn_qkv_weight_q == NULL) return 0;

    cudaStream_t st = gpu->stream;
    cublasHandle_t ch = gpu->handle;
    const int Cdim = gpu->conv_dim;
    const int kdim = gpu->key_dim;
    const int vdim = gpu->d_inner;
    const int dr = gpu->dt_rank;

    // Upload input to GPU (forward_full)
    cudaError_t ce = cudaMemcpyAsync(gpu->d_x, h_norm, (size_t)C * gpu->d_model * sizeof(float),
                    cudaMemcpyHostToDevice, st);
    if (ce != cudaSuccess) { fprintf(stderr, "GPU SSM fwd_full: d_x upload: %s\\n", cudaGetErrorString(ce)); return 0; }

    // === Step 1+2: Quantized matmuls — batched (all C tokens at once) ===
    // === Step 1+2: Quantized matmuls — batched (all C tokens at once) ===
    // x [C, D_MODEL] @ W_qkv [D_MODEL, CONV_DIM] → qkv [C, CONV_DIM]
    // x [C, D_MODEL] @ W_gate [D_MODEL, VALUE_DIM] → z [C, VALUE_DIM]
    if (!gpu->d_attn_qkv_q[layer_idx]) { fprintf(stderr, "GPU SSM: qkv weights NULL!\n"); return 0; }
    if (!gpu->d_attn_gate_q[layer_idx]) { fprintf(stderr, "GPU SSM: gate weights NULL!\n"); return 0; }

    int r1 = wubu_cuda_quant_matmul_batched(gpu->d_x, C,
        gpu->d_attn_qkv_q[layer_idx], gpu->ssm_qkv_type[layer_idx],
        gpu->d_model, gpu->conv_dim, gpu->d_ssm_qkv_out, st);
    int r2 = wubu_cuda_quant_matmul_batched(gpu->d_x, C,
        gpu->d_attn_gate_q[layer_idx], gpu->ssm_gate_type[layer_idx],
        gpu->d_model, gpu->d_inner, gpu->d_ssm_z_out, st);
    if (!r1 || !r2) {
        fprintf(stderr, "GPU SSM: batched quant_matmul failed (types %d, %d)\n",
            gpu->ssm_qkv_type[layer_idx], gpu->ssm_gate_type[layer_idx]);
        return 0;
    }
    cudaStreamSynchronize(st);

    // === Step 3+4: Beta/Alpha + sigmoid/softplus/gate ===
    // For C==1 decode: use fused kernel (no cuBLAS, no separate element-wise launches)
    // For C>1 prefill: use cuBLAS + individual kernels
    size_t off = 0;
    float *d_beta_sig  = gpu->d_ssm_scratch + off; off += (size_t)C * dr;
    float *d_gate_final = gpu->d_ssm_scratch + off; off += (size_t)C * dr;

    if (C == 1) {
        if (!gpu->d_ssm_beta[layer_idx] || !gpu->d_ssm_alpha[layer_idx] || !gpu->d_ssm_dt_bias[layer_idx] || !gpu->d_ssm_a[layer_idx]) {
            fprintf(stderr, "GPU SSM: ssm_beta/alpha/dt_bias/a NULL for layer %d\n", layer_idx);
            return 0;
        }
        ssm_beta_alpha_fused_decode_wrapper(st,
            gpu->d_x, gpu->d_ssm_beta[layer_idx], gpu->d_ssm_alpha[layer_idx],
            gpu->d_ssm_dt_bias[layer_idx], gpu->d_ssm_a[layer_idx],
            d_beta_sig, d_gate_final, dr);
        cudaStreamSynchronize(st);
        cudaError_t cem = cudaGetLastError();
        if (cem != cudaSuccess) { fprintf(stderr, "GPU SSM: beta/alpha kernel error: %s\n", cudaGetErrorString(cem)); return 0; }
    } else {
        // C>1 prefill: forward_full has inherent precision issues vs hybrid.
        // Fall back so the hybrid (CPU SSM + GPU quant matmuls) path handles it.
        // Batched quant matmul infrastructure exists (wubu_cuda_quant_matmul_batched)
        // but forward_full C>1 needs verification against CPU reference first.
        fprintf(stderr, "GPU SSM C>1: forward_full has accuracy issues, falling back to hybrid\n");
        return 0;
    }

    // === Step 5-10: For C==1, use fused conv1d+SiLU+split kernel ===
    // For C>1, use original separate steps
    float *d_q_conv, *d_k_conv, *d_v_conv, *d_q_norm, *d_k_norm;
    if (C == 1) {
        // Fused: no intermediate conv_input/conv_output scratch needed
        d_q_conv = gpu->d_ssm_scratch + off; off += (size_t)C * kdim;
        d_k_conv = gpu->d_ssm_scratch + off; off += (size_t)C * kdim;
        d_v_conv = gpu->d_ssm_scratch + off; off += (size_t)C * vdim;
        ssm_conv_silu_split_decode_wrapper(st,
            gpu->d_conv_state[layer_idx], gpu->d_ssm_qkv_out,
            gpu->d_ssm_conv1d[layer_idx],
            d_q_conv, d_k_conv, d_v_conv,
            gpu->d_conv_state[layer_idx]);  // in-place update
        cudaStreamSynchronize(st);
        cudaError_t cem2 = cudaGetLastError();
        if (cem2 != cudaSuccess) { fprintf(stderr, "GPU SSM: conv kernel error: %s\\n", cudaGetErrorString(cem2)); return 0; }
    } else {
        // Original multistep path for prefill (C>1)
        int k_1 = gpu->conv_kernel - 1;
        float *d_conv_input = gpu->d_ssm_scratch + off;
        off += (size_t)(k_1 + C) * Cdim;
        cudaMemcpyAsync(d_conv_input, gpu->d_conv_state[layer_idx],
                        (size_t)k_1 * Cdim * sizeof(float),
                        cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(d_conv_input + (size_t)k_1 * Cdim, gpu->d_ssm_qkv_out,
                        (size_t)C * Cdim * sizeof(float),
                        cudaMemcpyDeviceToDevice, st);
        // Conv1d + SiLU
        float *d_conv_output = gpu->d_ssm_scratch + off; off += (size_t)C * Cdim;
        wubu_cuda_conv1d(1, C, Cdim, gpu->conv_kernel, d_conv_input,
                         gpu->d_ssm_conv1d[layer_idx], d_conv_output, st);
        wubu_cuda_silu(C * Cdim, d_conv_output, d_conv_output, st);
        // Update conv_state
        float *last = d_conv_input + (size_t)C * Cdim;
        cudaMemcpyAsync(gpu->d_conv_state[layer_idx], last,
                        (size_t)k_1 * Cdim * sizeof(float),
                        cudaMemcpyDeviceToDevice, st);
        // Split QKV
        d_q_conv = gpu->d_ssm_scratch + off; off += (size_t)C * kdim;
        d_k_conv = gpu->d_ssm_scratch + off; off += (size_t)C * kdim;
        d_v_conv = gpu->d_ssm_scratch + off; off += (size_t)C * vdim;
        wubu_cuda_split_qkv(C, kdim, vdim, d_conv_output,
                            d_q_conv, d_k_conv, d_v_conv, st);
    }

    // L2 norm (needed for both paths)
    d_q_norm = gpu->d_ssm_scratch + off; off += (size_t)C * kdim;
    d_k_norm = gpu->d_ssm_scratch + off; off += (size_t)C * kdim;
    wubu_cuda_l2_norm(1, C, gpu->ssm_k_heads, gpu->ssm_d_state, d_q_conv, 1e-12f, d_q_norm, st);
    cudaStreamSynchronize(st);
    wubu_cuda_l2_norm(1, C, gpu->ssm_k_heads, gpu->ssm_d_state, d_k_conv, 1e-12f, d_k_norm, st);
    cudaStreamSynchronize(st);

        // === Step 11: Recurrence — parallel scan for C>1, token-by-token for C==1 ===
        float *d_delta_out = gpu->d_ssm_scratch + off; off += (size_t)C * vdim;

        if (C > 1) {
            // Batched prefill: process all C tokens in one parallel scan call
            wubu_cuda_ssm_parallel_scan(1, C,
                d_q_norm,    // [N, 16, 128]
                d_k_norm,    // [N, 16, 128]
                d_v_conv,    // [N, 32, 128]
                d_gate_final,// [N, 32]
                d_beta_sig,  // [N, 32]
                gpu->d_ssm_state[layer_idx],  // [1, 32, 128, 128]
                d_delta_out, // [N, 32, 128]
                st,
                gpu->ssm_k_heads, gpu->ssm_d_state, gpu->ssm_v_heads);
        } else {
            // Single-token decode: repeat K heads 16→32 on GPU, run recurrence kernel
            for (int t = 0; t < C; t++) {
                float *state = gpu->d_ssm_state[layer_idx];
                float *d_q_t = d_q_norm + (size_t)t * kdim;
                float *d_k_t = d_k_norm + (size_t)t * kdim;
                float *d_v_t = d_v_conv + (size_t)t * vdim;
                float *db = d_beta_sig + (size_t)t * dr;
                float *dg = d_gate_final + (size_t)t * dr;

                wubu_gpu_repeat_kheads(d_q_t, d_k_t, d_v_t, db, dg,
                    gpu->d_ssm_q_all, gpu->d_ssm_k_all, gpu->d_ssm_v_all,
                    gpu->d_ssm_beta_arr, gpu->d_ssm_gate_arr, st);
                wubu_gpu_ssm_recurrence(state,
                    gpu->d_ssm_q_all, gpu->d_ssm_k_all, gpu->d_ssm_v_all,
                    gpu->d_ssm_beta_arr, gpu->d_ssm_gate_arr,
                    gpu->d_ssm_delta_out, st);

                cudaMemcpyAsync(d_delta_out + (size_t)t * vdim, gpu->d_ssm_delta_out,
                    (size_t)gpu->ssm_v_heads * gpu->ssm_d_state * sizeof(float),
                    cudaMemcpyDeviceToDevice, st);
                cudaStreamSynchronize(st);
                cudaError_t cem4 = cudaGetLastError();
            }
        }
        cudaStreamSynchronize(st);
        cudaError_t cem5 = cudaGetLastError();

    // === Step 12-13: z = SiLU(z_all) + Gated norm ===
    float *d_z_silu = gpu->d_ssm_scratch + off; off += (size_t)C * vdim;
    // Ensure we have enough scratch
    if ((size_t)off * sizeof(float) > gpu->d_ssm_scratch_sz) {
        fprintf(stderr, "GPU SSM: scratch overflow (need %zu bytes, have %zu)\n",
                (size_t)off * sizeof(float), gpu->d_ssm_scratch_sz);
        return 0;
    }

    wubu_cuda_silu(C * vdim, gpu->d_ssm_z_out, d_z_silu, st);
    cudaStreamSynchronize(st);

    wubu_cuda_gated_norm(1, C, gpu->ssm_v_heads, gpu->ssm_d_state,
                         d_delta_out, gpu->d_ssm_norm[layer_idx], d_z_silu, st);
    cudaStreamSynchronize(st);

        // === Step 14: SSM output projection via quantized matmul (column-major Q6_K) ===
        // d_delta_out [C, vdim] @ ssm_out [vdim, D_MODEL] → gout [C, D_MODEL]
        for (int t = 0; t < C; t++) {
            float *d_delta_t = d_delta_out + (size_t)t * vdim;
            float *d_gout_t = gpu->d_gout + (size_t)t * D_MODEL;
            int r3 = wubu_cuda_quant_matmul(d_delta_t, gpu->d_ssm_out_q[layer_idx],
                gpu->ssm_out_type[layer_idx], vdim, D_MODEL, d_gout_t, NULL, 0, st);
            if (!r3) { fprintf(stderr, "GPU SSM: ssm_out matmul failed\n"); return 0; }
        }

    // === Step 15: Download final output ===
    cudaStreamSynchronize(st);
    cudaMemcpyAsync(h_attn_out, gpu->d_gout,
                    (size_t)C * D_MODEL * sizeof(float),
                    cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);

    // DEBUG: dump intermediate stages for first few calls (only for C==1 to avoid overflow)
    {
        static int call_cnt = 0;
        if (call_cnt++ < 1 && C == 1) {
            float *h = (float*)malloc((size_t)Cdim * sizeof(float));

            // qkv output
            cudaMemcpy(h, gpu->d_ssm_qkv_out, (size_t)C * Cdim * sizeof(float), cudaMemcpyDeviceToHost);
            float mx=0; for(int i=0;i<Cdim;i++){float f=fabsf(h[i]);if(f>mx)mx=f;}
            printf("DBG qkv: max=%.6f [0..3]=%.6f %.6f %.6f %.6f\n", mx, h[0],h[1],h[2],h[3]);

            // z output
            cudaMemcpy(h, gpu->d_ssm_z_out, (size_t)C * vdim * sizeof(float), cudaMemcpyDeviceToHost);
            mx=0; for(int i=0;i<vdim;i++){float f=fabsf(h[i]);if(f>mx)mx=f;}
            printf("DBG z: max=%.6f [0..3]=%.6f %.6f %.6f %.6f\n", mx, h[0],h[1],h[2],h[3]);

            // beta/alpha: scratch[0..31]=beta_sig, scratch[32..63]=gate
            cudaMemcpy(h, gpu->d_ssm_scratch, (size_t)2 * dr * sizeof(float), cudaMemcpyDeviceToHost);
            printf("DBG beta_sig[0..4]: %.6f %.6f %.6f %.6f %.6f\n", h[0],h[1],h[2],h[3],h[4]);
            printf("DBG gate[0..4]: %.6f %.6f %.6f %.6f %.6f\n", h[32],h[33],h[34],h[35],h[36]);

            // conv/silu/split outputs at scratch[64], scratch[64+2048], scratch[64+4096]
            // d_q_conv at scratch[64..2111], d_k_conv at scratch[2112..4159], d_v_conv at scratch[4160..8255]
            size_t kv_off = (size_t)(2 * C * dr);
            cudaMemcpy(h, gpu->d_ssm_scratch + kv_off, (size_t)C * kdim * sizeof(float), cudaMemcpyDeviceToHost);
            mx=0; for(int i=0;i<kdim;i++){float f=fabsf(h[i]);if(f>mx)mx=f;}
            printf("DBG q_conv: max=%.6f [0..3]=%.6f %.6f %.6f %.6f\n", mx, h[0],h[1],h[2],h[3]);

            // L2 normed Q at scratch[64+8192..10303]
            size_t l2_off = kv_off + (size_t)(2 * C * kdim);
            cudaMemcpy(h, gpu->d_ssm_scratch + l2_off, (size_t)C * kdim * sizeof(float), cudaMemcpyDeviceToHost);
            mx=0; for(int i=0;i<kdim;i++){float f=fabsf(h[i]);if(f>mx)mx=f;}
            printf("DBG q_norm: max=%.6f [0..3]=%.6f %.6f %.6f %.6f\n", mx, h[0],h[1],h[2],h[3]);

            // delta_out at scratch[64+12288..16447]
            size_t delta_off = l2_off + (size_t)(C * kdim);
            cudaMemcpy(h, gpu->d_ssm_scratch + delta_off, (size_t)C * vdim * sizeof(float), cudaMemcpyDeviceToHost);
            mx=0; for(int i=0;i<vdim;i++){float f=fabsf(h[i]);if(f>mx)mx=f;}
            printf("DBG delta_out: max=%.6f [0..3]=%.6f %.6f %.6f %.6f\n", mx, h[0],h[1],h[2],h[3]);

            // z_silu at scratch[64+16448..20543]
            size_t zs_off = delta_off + (size_t)(C * vdim);
            cudaMemcpy(h, gpu->d_ssm_scratch + zs_off, (size_t)C * vdim * sizeof(float), cudaMemcpyDeviceToHost);
            mx=0; for(int i=0;i<vdim;i++){float f=fabsf(h[i]);if(f>mx)mx=f;}
            printf("DBG z_silu: max=%.6f [0..3]=%.6f %.6f %.6f %.6f\n", mx, h[0],h[1],h[2],h[3]);

            // Final output
            cudaMemcpy(h, gpu->d_gout, (size_t)C * gpu->d_model * sizeof(float), cudaMemcpyDeviceToHost);
            mx=0; for(int i=0;i<gpu->d_model;i++){float f=fabsf(h[i]);if(f>mx)mx=f;}
            printf("DBG gout: max=%.6f [0..3]=%.6f %.6f %.6f %.6f\n", mx, h[0],h[1],h[2],h[3]);
            
            free(h);
        }
    }

    return 1;
}
extern "C"
int wubu_model_gpu_quant_matmul(const float *x, int n_rows, int n_cols,
                                 const uint8_t *d_W_q, int quant_type,
                                 float *y, cudaStream_t stream) {
    float *d_x, *d_y;
    cudaMalloc(&d_x, (size_t)n_rows * sizeof(float));
    cudaMalloc(&d_y, (size_t)n_cols * sizeof(float));
    cudaMemcpyAsync(d_x, x, (size_t)n_rows * sizeof(float), cudaMemcpyHostToDevice, stream);
    int ret = wubu_cuda_quant_matmul(d_x, d_W_q, quant_type, n_rows, n_cols, d_y, NULL, 0, stream);
    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(y, d_y, (size_t)n_cols * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_x);
    cudaFree(d_y);
    return ret;
}

// ================================================================
// Set SSM layer GPU pointers for hybrid (CPU SSM + GPU recurrence) mode
// ================================================================
extern "C"
void wubu_gpu_set_ssm_hybrid(void *gpu_ctx_ptr, int layer_idx, ssm_layer_weights *ssm) {
    gpu_ctx_t *gpu = (gpu_ctx_t *)gpu_ctx_ptr;
    ssm->gpu_ssm_state = (void*)gpu->d_ssm_state[layer_idx];
    ssm->gpu_q_buf     = (void*)gpu->d_ssm_q_all;
    ssm->gpu_k_buf     = (void*)gpu->d_ssm_k_all;
    ssm->gpu_v_buf     = (void*)gpu->d_ssm_v_all;
    ssm->gpu_beta_buf  = (void*)gpu->d_ssm_beta_arr;
    ssm->gpu_gate_buf  = (void*)gpu->d_ssm_gate_arr;
    ssm->gpu_delta_buf = (void*)gpu->d_ssm_delta_out;
    ssm->gpu_stream    = (void*)gpu->stream;
}

// ================================================================
// Sync CPU SSM state + conv state to GPU (after hybrid prefill)
// ================================================================
extern "C"
void wubu_gpu_sync_ssm_state_to_gpu(void *gpu_ctx_ptr, int layer_idx,
                                     const float *cpu_ssm_state,
                                     const float *cpu_conv_state) {
    gpu_ctx_t *gpu = (gpu_ctx_t *)gpu_ctx_ptr;
    if (!gpu || !gpu->initialized) return;
    cudaStream_t st = gpu->stream;
    // Copy ssm_state: [V_HEADS, D_STATE, D_STATE] floats
    cudaMemcpyAsync(gpu->d_ssm_state[layer_idx], cpu_ssm_state,
        (size_t)gpu->ssm_v_heads * gpu->ssm_d_state * gpu->ssm_d_state * sizeof(float),
        cudaMemcpyHostToDevice, st);
    // Copy conv_state: [CONV_KERNEL-1, CONV_DIM] floats
    cudaMemcpyAsync(gpu->d_conv_state[layer_idx], cpu_conv_state,
        (size_t)(gpu->conv_kernel - 1) * gpu->conv_dim * sizeof(float),
        cudaMemcpyHostToDevice, st);
    cudaStreamSynchronize(st);
}

// ================================================================
// Sync GPU SSM state + conv state back to CPU (after forward_full decode)
// ================================================================
extern "C"
void wubu_gpu_sync_ssm_state_to_cpu(void *gpu_ctx_ptr, int layer_idx,
                                     float *cpu_ssm_state,
                                     float *cpu_conv_state) {
    gpu_ctx_t *gpu = (gpu_ctx_t *)gpu_ctx_ptr;
    if (!gpu || !gpu->initialized) return;
    cudaStream_t st = gpu->stream;
    cudaMemcpyAsync(cpu_ssm_state, gpu->d_ssm_state[layer_idx],
        (size_t)SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float),
        cudaMemcpyDeviceToHost, st);
    cudaMemcpyAsync(cpu_conv_state, gpu->d_conv_state[layer_idx],
        (size_t)(CONV_KERNEL - 1) * CONV_DIM * sizeof(float),
        cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
}

// ================================================================
// Free GPU resources
// ================================================================
extern "C"
void wubu_model_gpu_free(wubu_model_t *model) {
    gpu_ctx_t *gpu = (gpu_ctx_t *)model->gpu_ctx;
    if (!gpu) return;

    // Free SSM quantized weights + F32 dequant copies
    if (gpu->d_attn_qkv_q) {
        for (int i = 0; i < gpu->n_layers; i++) {
            wubu_cuda_free((float*)gpu->d_attn_qkv_q[i]);
            wubu_cuda_free((float*)gpu->d_attn_gate_q[i]);
            wubu_cuda_free((float*)gpu->d_ssm_out_q[i]);
            wubu_cuda_free(gpu->d_qkv_f32[i]);
            wubu_cuda_free(gpu->d_gate_f32[i]);
            wubu_cuda_free(gpu->d_out_f32[i]);
        }
    }

    // Free GQA weights
    if (gpu->gqa_weights) {
        for (int l = 0; l < gpu->n_layers; l++) {
            gpu_gqa_layer_t *gw = &gpu->gqa_weights[l];
            wubu_cuda_free(gw->d_attn_q);
            wubu_cuda_free(gw->d_attn_k);
            wubu_cuda_free(gw->d_attn_v);
            wubu_cuda_free(gw->d_attn_out_w);
            wubu_cuda_free(gw->d_q_norm_w);
            wubu_cuda_free(gw->d_k_norm_w);
        }
        free(gpu->gqa_weights);
    }

    // Free KV cache
    if (gpu->d_k_cache) {
        for (int l = 0; l < gpu->n_layers; l++)
            wubu_cuda_free((float*)gpu->d_k_cache[l]);
        free(gpu->d_k_cache);
    }
    if (gpu->d_v_cache) {
        for (int l = 0; l < gpu->n_layers; l++)
            wubu_cuda_free((float*)gpu->d_v_cache[l]);
        free(gpu->d_v_cache);
    }
    free(gpu->cache_len);
    free(gpu->cache_cap);

    // Free RoPE table
    wubu_cuda_free(gpu->d_sincos);

    // Free scratch
    wubu_cuda_free(gpu->d_x);
    wubu_cuda_free((float*)gpu->d_scr);
    wubu_cuda_free((float*)gpu->d_ktmp);
    wubu_cuda_free((float*)gpu->d_vtmp);
    wubu_cuda_free(gpu->d_gout);
    wubu_cuda_free(gpu->d_score_scr);
    wubu_cuda_free((float*)gpu->d_hp_scratch);
    wubu_cuda_free(gpu->d_qtmp);

    // Free SSM output buffers
    wubu_cuda_free(gpu->d_ssm_qkv_out);
    wubu_cuda_free(gpu->d_ssm_z_out);

    // Free MoE buffers
    wubu_cuda_free((float*)gpu->d_moe_gate);
    wubu_cuda_free((float*)gpu->d_moe_up);
    wubu_cuda_free((float*)gpu->d_moe_down);
    wubu_cuda_free(gpu->d_moe_x);
    wubu_cuda_free(gpu->d_moe_out);
    wubu_cuda_free(gpu->d_moe_weights);

    // Free MoE expert cache
    if (gpu->moe_cache_w) {
        for (int l = 0; l < gpu->n_layers; l++) {
            if (gpu->moe_cache_w[l]) {
                wubu_cuda_free((float*)gpu->moe_cache_w[l][0]);
                wubu_cuda_free((float*)gpu->moe_cache_w[l][1]);
                wubu_cuda_free((float*)gpu->moe_cache_w[l][2]);
                free(gpu->moe_cache_w[l]);
            }
        }
        free(gpu->moe_cache_w);
    }
    if (gpu->moe_cache_eid) {
        for (int l = 0; l < gpu->n_layers; l++) free(gpu->moe_cache_eid[l]);
        free(gpu->moe_cache_eid);
    }

    // Free SSM recurrence state and temp buffers
    if (gpu->d_ssm_state) {
        for (int l = 0; l < gpu->n_layers; l++)
            wubu_cuda_free(gpu->d_ssm_state[l]);
        free(gpu->d_ssm_state);
    }
    if (gpu->d_conv_state) {
        for (int l = 0; l < gpu->n_layers; l++)
            wubu_cuda_free(gpu->d_conv_state[l]);
        free(gpu->d_conv_state);
    }
    wubu_cuda_free(gpu->d_ssm_q_all);
    wubu_cuda_free(gpu->d_ssm_k_all);
    wubu_cuda_free(gpu->d_ssm_v_all);
    wubu_cuda_free(gpu->d_ssm_beta_arr);
    wubu_cuda_free(gpu->d_ssm_gate_arr);
    wubu_cuda_free(gpu->d_ssm_delta_out);

    // Free SSM full forward scratch
    wubu_cuda_free(gpu->d_ssm_scratch);

    // Free SSM small F32 weights
    for (int i = 0; i < 40 && i < gpu->n_layers; i++) {
        wubu_cuda_free(gpu->d_ssm_beta[i]);
        wubu_cuda_free(gpu->d_ssm_alpha[i]);
        wubu_cuda_free(gpu->d_ssm_dt_bias[i]);
        wubu_cuda_free(gpu->d_ssm_a[i]);
        wubu_cuda_free(gpu->d_ssm_conv1d[i]);
        wubu_cuda_free(gpu->d_ssm_norm[i]);
    }

    // Destroy CUDA context
    if (gpu->handle) cublasDestroy(gpu->handle);
    if (gpu->stream) cudaStreamDestroy(gpu->stream);

    memset(gpu, 0, sizeof(*gpu));
    free(gpu);
    model->gpu_ctx = NULL;
    model->backend = wubu_backend_cpu_get();
}
