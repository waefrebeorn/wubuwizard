#ifndef WUBU_GPU_GEMMA4_H
#define WUBU_GPU_GEMMA4_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
typedef struct g4_gpu_ctx_t g4_gpu_ctx_t;

/* ============================================================
 *  Triple-buffer pipeline: SSD → RAM → VRAM bandwidth prefecture
 *
 *  3 CUDA streams overlap:
 *    stream[0] = COMPUTE  — GPU kernels for layer N
 *    stream[1] = UPLOAD   — cudaMemcpy staging→VRAM for layer N+1
 *    stream[2] = PREFETCH — fread file→staging RAM for layer N+2
 *
 *  Staging buffers (pinned host memory):
 *    ram_buf[0] = being filled by fread (prefetch)
 *    ram_buf[1] = being copied to VRAM (upload)
 *    ram_buf[2] = ready / being consumed by compute
 *
 *  CUDA events synchronize the handoff:
 *    upload_done[3]  — stream[1] finished copying ram_buf[i] → VRAM
 *    prefetch_done[3] — stream[2] finished fread into ram_buf[i]
 *
 *  64-token batch sync-up:
 *    Tokens accumulate in a buffer. At 64 tokens (or on flush),
 *    we run a batched prefill (much faster than token-by-token).
 *    During decode, we maintain a lookahead buffer for speed gains
 *    during QAT MTP (Multi-Token Prediction) process.
 * ============================================================ */

#define G4_GPU_RAW_BUF_SIZE  (64 * 1024 * 1024)   /* 64 MB staging buffer */
#define G4_GPU_MAX_BATCH     64                     /* batch sync-up threshold */
#define G4_GPU_N_STREAMS     3                      /* compute / upload / prefetch */
#define G4_GPU_N_BUFFERS     3                      /* triple-buffer rotation */

/* Per-layer weight storage in VRAM (sub-allocated from one big pool) */
typedef struct {
    uint8_t *d_q, *d_k, *d_v, *d_o, *d_g, *d_u, *d_d;
    int q_bytes, k_bytes, v_bytes, o_bytes, g_bytes, u_bytes, d_bytes;
    int uploaded;
} g4_layer_vram_t;

/* GPU context */
struct g4_gpu_ctx_t {
    cublasHandle_t cublas;
    cudaStream_t   stream[G4_GPU_N_STREAMS];  /* 0=compute, 1=upload, 2=prefetch */
    cudaEvent_t    upload_done[G4_GPU_N_BUFFERS];
    cudaEvent_t    prefetch_done[G4_GPU_N_BUFFERS];

    /* Pinned staging buffers for SSD→RAM→VRAM pipeline */
    uint8_t *ram_buf[G4_GPU_N_BUFFERS];       /* pinned host memory */
    int      ram_owner[G4_GPU_N_BUFFERS];      /* which layer owns this buffer, -1=free */
    int      ram_buf_pinned[G4_GPU_N_BUFFERS]; /* 1 = cudaHostAlloc, 0 = malloc */

    /* Single large VRAM allocation — sub-allocated to avoid WSL2 cudaMalloc fragmentation */
    uint8_t *d_weight_pool;                    /* big contiguous VRAM block */
    size_t   weight_pool_size;
    size_t   weight_pool_used;

    /* Device buffers for activations (ping-pong) */
    float *d_scratch;       /* dequant workspace */
    float *d_q, *d_k, *d_v;
    float *d_attn_out;
    float *d_ffn_gate, *d_ffn_up;
    float *d_hidden;        /* [N, 3840] */
    float *d_normed;
    float *d_logits;
    float *d_row_buf;

    /* KV cache on device */
    float *d_k_cache, *d_v_cache;
    int   *d_positions;
    int   *d_kv_sizes;

    /* Norm/rope weights (tiny, always resident) */
    float *d_norm[48], *d_qnorm[48], *d_knorm[48];
    float *d_pnorm[48], *d_fnorm[48], *d_pfnorm[48];
    float *d_rope_freqs[48];

    /* Token embedding + output weights (uploaded once at init) */
    uint8_t *d_token_embd;
    uint8_t *d_output_w;
    float   *d_output_norm;

    /* Layer weights in VRAM (sub-allocated from pool) */
    g4_layer_vram_t layer_vram[48];

    /* Batch sync-up state */
    int   batch_buf[G4_GPU_MAX_BATCH];  /* token buffer */
    int   batch_len;                     /* current buffer fill */
    int   batch_pos;                     /* position in current batch */

    int max_tokens;
    int max_ctx;
    int n_layers;
    int initialized;
};

/* Init / destroy */
g4_gpu_ctx_t *g4_gpu_init(int max_tokens, int max_ctx, int n_layers);
void          g4_gpu_destroy(g4_gpu_ctx_t *ctx);

/* Full GPU forward (batch-aware: accumulates tokens, runs at 64-token threshold) */
int g4_model_forward_gpu(g4_gpu_ctx_t *gpu_ctx, void *model_ptr,
                          const float *embeddings, int B, int T,
                          float *logits);

/* Flush remaining tokens in batch buffer (call at end of generation) */
int g4_model_forward_gpu_flush(g4_gpu_ctx_t *gpu_ctx, void *model_ptr,
                                float *logits);

/* Single-token decode (uses batch internally) */
int g4_model_decode_gpu(g4_gpu_ctx_t *gpu_ctx, void *model_ptr,
                        int token, float *logits);

/* Sub-allocation from weight pool (avoids repeated cudaMalloc on WSL2) */
uint8_t *g4_gpu_weight_alloc(g4_gpu_ctx_t *ctx, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_GPU_GEMMA4_H */
