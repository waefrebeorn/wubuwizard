#ifndef WUBU_MODEL_H
#define WUBU_MODEL_H

#include "wubu_ssm.h"
#include "wubu_moe.h"
#include <stdbool.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Layer configuration
typedef struct {
    int layer_idx;
    bool is_ssm;  // false = GQA
    
    // Weights (loaded from GGUF)
    ssm_layer_weights ssm;      // valid if is_ssm
    gqa_layer_weights gqa;      // valid if !is_ssm
    
    // Layer norm (pre-attention for all layers)
    float *attn_norm_weight;    // [D_MODEL], RMSNorm
    
    // Post-attention norm
    float *post_attn_norm_weight; // [D_MODEL], RMSNorm
    
    // MoE (FFN) weights
    moe_weights_t moe;
} wubu_layer_t;

// Complete model
#define GQA_MAX_CTX 524288  // max cached positions for KV cache (512k context)
#define GQA_KV_DIM (GQA_KV_HEADS * GQA_HEAD_DIM)  // 512

// KV cache format: 0=F32, 1=F16 (halves memory at cost of conversion)
#ifndef KV_CACHE_F16
#define KV_CACHE_F16 1  // default to F16 for memory efficiency
#endif

// Expert recorder: max tokens to capture per profiling run
#define MAX_EXPERT_RECORDER_TOKENS 256

// F16 <-> F32 conversion helpers (used by KV cache)
static inline float fp16_to_fp32(uint16_t v) {
    int sign = (v >> 15) & 1;
    int exp  = (v >> 10) & 0x1F;
    int mant =  v        & 0x03FF;
    if (exp == 0) return ldexpf((float)mant / 1024.0f, -14) * (sign ? -1.0f : 1.0f);
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    return ldexpf(1.0f + (float)mant / 1024.0f, exp - 15) * (sign ? -1.0f : 1.0f);
}
static inline uint16_t fp32_to_fp16(float v) {
    uint32_t bits; memcpy(&bits, &v, 4);
    int sign = (bits >> 31) & 1;
    int exp  = (bits >> 23) & 0xFF;
    int mant = bits & 0x7FFFFF;
    uint16_t fp16;
    if (exp == 0) { fp16 = (sign << 15) | (0) | (mant >> 13); }
    else if (exp == 0xFF) { fp16 = (sign << 15) | (31 << 10) | (mant >> 13); }
    else {
        int newexp = exp - 127 + 15;
        if (newexp >= 31) fp16 = (sign << 15) | (31 << 10);
        else if (newexp <= 0) fp16 = (sign << 15);
        else fp16 = (sign << 15) | (newexp << 10) | (mant >> 13);
    }
    return fp16;
}

// KV cache access helpers
static inline float kv_cache_read_elem(const void *cache, int64_t idx) {
#if KV_CACHE_F16
    return fp16_to_fp32(((const uint16_t *)cache)[idx]);
#else
    return ((const float *)cache)[idx];
#endif
}
// KV cache quantization options
// KV_CACHE_F16: half-precision (default, 2 bytes/elem)
// KV_CACHE_F32: full precision (4 bytes/elem, fallback)
// KV_CACHE_Q4_0: 4-bit quantized (0.5 bytes/elem for payload, ~0.56 bytes with scale)

#ifndef KV_CACHE_Q4_0
#define KV_CACHE_Q4_0 1  // Q4_0 format for KV cache (4:1 compression vs F16)
#endif

// Q4_0 block: 32 elements, 4-bit each + fp16 scale
typedef struct {
    uint16_t d;    // scale factor (fp16)
    uint8_t qs[16];  // 32 × 4-bit nibbles
} block_q4_0_cache;

#define QK4_CACHE 32

// Quantize 32 floats to Q4_0 block (symmetric, signed)
static inline void quantize_q4_0_cache_block(const float *x, block_q4_0_cache *b) {
    float amax = 0.0f;
    for (int i = 0; i < QK4_CACHE; i++) {
        float ax = fabsf(x[i]);
        if (ax > amax) amax = ax;
    }
    if (amax == 0.0f) {
        b->d = 0;
        memset(b->qs, 0, sizeof(b->qs));
        return;
    }
    const float d = amax / 7.0f;  // symmetric signed: [-7, 7] → [1, 15]
    const float id = 1.0f / d;
    b->d = fp32_to_fp16(d);
    for (int i = 0; i < QK4_CACHE; i++) {
        int q = (int)(x[i] * id + 8.0f);
        if (q < 0) q = 0;
        if (q > 15) q = 15;
        b->qs[i / 2] |= (uint8_t)(q << (4 * (i % 2)));
    }
}

// Dequantize one Q4_0 block
static inline void dequantize_q4_0_cache_block(const block_q4_0_cache *b, float *x) {
    const float d = fp16_to_fp32(b->d);
    for (int i = 0; i < QK4_CACHE; i++) {
        int q = (b->qs[i / 2] >> (4 * (i % 2))) & 0xF;
        x[i] = ((float)q - 8.0f) * d;
    }
}

// KV cache read: one head (n floats) from Q4_0 cache
static inline void kv_cache_read_head(const void *cache, int64_t offset,
                                       float *buf, int n) {
#if KV_CACHE_Q4_0
    // Q4_0: offset is in float indices, convert to block index
    const int block_n = QK4_CACHE;
    int start_block = (int)(offset / block_n);
    int start_elem = (int)(offset % block_n);
    const block_q4_0_cache *blocks = (const block_q4_0_cache *)cache;
    
    int done = 0;
    while (done < n) {
        float tmp[QK4_CACHE];
        dequantize_q4_0_cache_block(&blocks[start_block + (start_elem + done) / block_n], tmp);
        int blk_off = (start_elem + done) % block_n;
        int to_copy = n - done;
        if (to_copy > block_n - blk_off) to_copy = block_n - blk_off;
        for (int i = 0; i < to_copy; i++) buf[done + i] = tmp[blk_off + i];
        done += to_copy;
    }
#elif KV_CACHE_F16
    const uint16_t *src = (const uint16_t *)cache + offset;
    for (int i = 0; i < n; i++) buf[i] = fp16_to_fp32(src[i]);
#else
    memcpy(buf, (const float *)cache + offset, n * sizeof(float));
#endif
}

// Batch write one head to Q4_0 cache
static inline void kv_cache_write_head(void *cache, int64_t offset,
                                        const float *buf, int n) {
#if KV_CACHE_Q4_0
    const int block_n = QK4_CACHE;
    int start_block = (int)(offset / block_n);
    int start_elem = (int)(offset % block_n);
    int end_elem = start_elem + n;
    block_q4_0_cache *blocks = (block_q4_0_cache *)cache;
    
    if (start_elem == 0) {
        // Start is aligned — handle whole blocks fast
        int n_aligned = n - (end_elem % block_n);
        if (n_aligned < 0) n_aligned = 0;
        // Write whole blocks
        for (int bi = 0; bi < n_aligned / block_n; bi++) {
            quantize_q4_0_cache_block(buf + bi * block_n, &blocks[start_block + bi]);
        }
        // Remaining partial block at the end
        int rem = n - n_aligned;
        if (rem > 0) {
            int bi = n_aligned / block_n;
            float tmp[QK4_CACHE];
            dequantize_q4_0_cache_block(&blocks[start_block + bi], tmp);
            for (int i = 0; i < rem; i++) tmp[i] = buf[n_aligned + i];
            quantize_q4_0_cache_block(tmp, &blocks[start_block + bi]);
        }
    } else {
        // Misaligned start: handle first partial block + aligned blocks + last partial
        // First partial block
        int first_rem = block_n - start_elem;
        if (first_rem > n) first_rem = n;
        {
            float tmp[QK4_CACHE];
            dequantize_q4_0_cache_block(&blocks[start_block], tmp);
            for (int i = 0; i < first_rem; i++) tmp[start_elem + i] = buf[i];
            quantize_q4_0_cache_block(tmp, &blocks[start_block]);
        }
        // Aligned bulk
        int remaining = n - first_rem;
        if (remaining > 0) {
            kv_cache_write_head(cache, offset + first_rem, buf + first_rem, remaining);
        }
    }
#elif KV_CACHE_F16
    uint16_t *dst = (uint16_t *)cache + offset;
    for (int i = 0; i < n; i++) dst[i] = fp32_to_fp16(buf[i]);
#else
    memcpy((float *)cache + offset, buf, n * sizeof(float));
#endif
}

// KV cache allocation: returns number of bytes needed for n_elems
static inline int64_t kv_cache_alloc_size(int64_t n_elems) {
#if KV_CACHE_Q4_0
    int64_t n_blocks = (n_elems + QK4_CACHE - 1) / QK4_CACHE;
    return n_blocks * (int64_t)sizeof(block_q4_0_cache);
#elif KV_CACHE_F16
    return n_elems * (int64_t)sizeof(uint16_t);
#else
    return n_elems * (int64_t)sizeof(float);
#endif
}

// MTP (Multi-Token Prediction) head for speculative decode
// Architecture: h_39 → hnorm → concat(hnorm, enorm(embd)) → eh_proj → blk.40 → shared_head_norm → output
typedef struct {
    bool loaded;
    
    // Nextn norms (F32, all [D_MODEL])
    float *nextn_hnorm;             // [2048] — hidden state norm
    float *nextn_enorm;             // [2048] — token embedding norm
    float *nextn_shared_head_norm;  // [2048] — output norm
    
    // eh_proj weight (F32 dequantized): concat([h_norm | e_norm], dim=4096) → [2048]
    float *nextn_eh_proj_f32;   // [4096, 2048] F32
    int64_t nextn_eh_proj_dim;         // 4096 (concat dim)
    
    // Blk.40 (a full GQA+MoE layer)
    wubu_layer_t blk40;
    
    // KV cache for blk.40's GQA attention
    float *k_cache;  // [GQA_MAX_CTX * GQA_KV_DIM]
    float *v_cache;  // [GQA_MAX_CTX * GQA_KV_DIM]
    int cache_len;
} mtp_head_t;
typedef struct {
    int n_layers;
    wubu_layer_t *layers;
    
    // Token embedding
    float *token_embd;       // [vocab_size, D_MODEL] or NULL if using embedding file
    float *output_weight;    // [D_MODEL, vocab_size] or NULL
    const uint8_t *output_weight_q;   // raw Q4_K quantized
    int output_weight_type;
    
    // Embedding file (from Phase 1)
    bool use_embedding_file;
    int vocab_size;
    
    // Norms
    float *norm_weight;  // final RMSNorm [D_MODEL]
    
    // State buffers (reused across calls)
    float *ssm_states;    // [max_layers, SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
    float *conv_states;   // [max_layers, B, CONV_KERNEL-1, CONV_DIM]
    
    // GQA KV cache (10 GQA layers, max 256k context)
    void *gqa_k_cache;  // [10 * GQA_MAX_CTX * GQA_KV_DIM] F32 or F16
    void *gqa_v_cache;  // [10 * GQA_MAX_CTX * GQA_KV_DIM]
    int gqa_cache_len;   // how many tokens cached per layer (all 10 layers same len)
    
    // GGUF context (for per-layer MoE lazy loading)
    // Model state save/restore (for speculative decode rollback)
    float *ssm_states_saved;    // same size as ssm_states
    float *conv_states_saved;   // same size as conv_states
    int gqa_cache_len_saved;
    int mtp_cache_len_saved;

    // GGUF context (for per-layer MoE lazy loading)
    gguf_ctx *gguf_ctx;
    
    // Enable MoE during forward (default: false for memory reasons)
    bool enable_moe;
    
    // Skip output projection in forward (for GPU offload)
    bool skip_output_proj;

    // Expert selection recorder (for demoscene profiling, P3 prefetch matrix)
    // Set to a [n_layers * MAX_EXPERT_RECORDER_TOKENS][N_ACTIVE_EXPTS] buffer before forward.
    // Captured from wubu_moe_forward's selected_experts output parameter.
    int (*expert_recorder)[N_ACTIVE_EXPTS];
    int expert_recorder_tokens;  // number of tokens recorded so far

    // MoE test: only load MoE for first N layers (0 = all)
    int moe_max_layers;
    
    // MTP (Multi-Token Prediction) head for speculative decode
    mtp_head_t mtp;
    
    // Last hidden state capture (for MTP: set to a [D_MODEL] buffer before forward)
    float *save_last_hidden;

    // GPU acceleration context (opaque pointer, managed by wubu_model_gpu.cu)
    // When non-NULL, GQA layers run on GPU via chunked attention.
    void *gpu_ctx;

    // Expert prefetch history (P3: prompt-aware prefetch matrix)
    // Records which 8 experts each layer selected on the last forward pass.
    // Expert prefetch history (P3: prompt-aware prefetch matrix)
    // Records which 8 experts each layer selected on the last forward pass.
    int expert_history[40][N_ACTIVE_EXPTS];
    uint64_t last_prompt_hash;
    bool expert_history_valid;

    // Fast decode: cache previous token's per-layer hidden states
    float *prev_hidden;    // [n_layers][D_MODEL] — hidden state at each layer for prev token
    float *prev_ssm_out;   // [n_layers][D_MODEL] — SSM output at each layer for prev token
    bool fast_decode;      // enable layer skipping
    int fast_skip_count;   // how many layers were skipped (for stats)

    // Logit cache: stores previous token's full logits + argmax
    float *logit_cache;       // [vocab_size] — full logits from previous forward
    int logit_cache_argmax;   // argmax from cache
    bool logit_cache_valid;
    int logit_cache_steps;    // count of consecutive cache uses
    int logit_cache_max_hits; // adaptive max consecutive cache hits (default 2)
    int logit_cache_argmax_prev; // previous forward's argmax (for stability detection)
    
    // Logit subset cache: top-K token IDs from last full output proj
    // Used for fast refresh: compute only these K tokens instead of all 248k
#define LOGIT_SUBSET_K 1000
    int logit_subset_ids[LOGIT_SUBSET_K];  // token IDs
    float logit_subset_vals[LOGIT_SUBSET_K]; // cached logit values
    bool logit_subset_valid;
} wubu_model_t;

// Create model, load from GGUF
bool wubu_model_init(wubu_model_t *model, const char *gguf_path);

// Free model resources
void wubu_model_free(wubu_model_t *model);

// Forward pass through all layers
// Input: token_ids [B, T], Output: logits [B, T, vocab_size]
void wubu_model_forward(wubu_model_t *model,
                        const int *token_ids, int B, int T,
                        float *logits);

// Forward pass from embeddings (bypass token lookup)
// Input: embeddings [B, T, D_MODEL], Output: logits [B, T, vocab_size]
void wubu_model_forward_from_embd(wubu_model_t *model,
                                   const float *embeddings, int B, int T,
                                   float *logits);

// ================================================================
// GPU-Accelerated Forward Path (wubu_model_gpu.cu)
// ================================================================

// Initialize GPU context: upload GQA weights, allocate KV cache + scratch.
// max_ctx: maximum KV cache positions (e.g. 262144).
// chunk_sz: max tokens per GPU batch (e.g. 512).
// Returns 1 on success, 0 on failure.
// When GPU context is active, wubu_model_forward() automatically uses
// GPU for GQA attention layers.
int wubu_model_gpu_init(wubu_model_t *model, int max_ctx, int chunk_sz);

// Run one GQA layer on GPU.
// Internal: called by wubu_model_forward when gpu_ctx != NULL.
int wubu_model_gpu_gqa_forward(wubu_model_t *model, int layer_idx,
                                const float *h_norm, int C, float *h_attn);

// Get GPU chunk size (max tokens per batched GPU call).
// Returns 0 if GPU not initialized.
int wubu_model_gpu_chunk_sz(wubu_model_t *model);

// Run SSM projections (qkv, gate) on GPU via quantized matmul kernels.
// h_norm: [C, D_MODEL] input
// C: number of tokens (1 for decode)
// qkv_out: [C, CONV_DIM] output (host)
// z_out: [C, VALUE_DIM] output (host)
// ssm_out_out: unused (future: ssm output projection)
int wubu_model_gpu_ssm_project(wubu_model_t *model, int layer_idx,
                                const float *h_norm, int C,
                                float *qkv_out, float *z_out,
                                float *ssm_out_out);

// Run GPU SSM completely on GPU: quantized matmuls → conv1d → SiLU → split
// → L2 norm → recurrence → gated norm → ssm_out projection.
// Returns 1 on success, 0 on fallback to CPU.
int wubu_model_gpu_ssm_forward_full(wubu_model_t *model, int layer_idx,
                                     const float *h_norm, int C,
                                     float *h_attn_out);

// Set SSM layer GPU pointers from gpu_ctx for hybrid (CPU SSM + GPU recurrence).
// Called by wubu_model_forward fallback paths when gpu_ctx exists.
// gpu_ctx is model->gpu_ctx (void*), ssm is layer->ssm to fill.
void wubu_gpu_set_ssm_hybrid(void *gpu_ctx, int layer_idx, ssm_layer_weights *ssm);

// Sync CPU SSM state + conv state to GPU before forward_full decode.
// Call after hybrid prefill path updates CPU state, so subsequent
// forward_full decode uses the correct accumulated state.
void wubu_gpu_sync_ssm_state_to_gpu(void *gpu_ctx, int layer_idx,
                                     const float *cpu_ssm_state,
                                     const float *cpu_conv_state);

// Sync GPU SSM state + conv state back to CPU after forward_full decode.
// Ensures CPU state tracks GPU state for next hybrid prefill.
void wubu_gpu_sync_ssm_state_to_cpu(void *gpu_ctx, int layer_idx,
                                     float *cpu_ssm_state,
                                     float *cpu_conv_state);

// Run MoE experts via GPU kernel, replacing CPU quantized matmul loop.
// Shared expert and router remain on CPU.
// Called per-token from wubu_moe_forward's expert loop.
void wubu_model_gpu_moe_experts(const moe_weights_t *w,
    const float *x_s,
    const int *indices_s, const float *weights_s,
    float expert_contribs[8][D_MODEL],
    void *model_ptr);

// Free all GPU resources and reset gpu_ctx to NULL.
void wubu_model_gpu_free(wubu_model_t *model);

// Model-level backward pass
// Requires saved layer outputs from forward (normed, attn_out, normed2, ffn_out arrays)
// All arrays are [n_layers * B * T * D_MODEL] flattened
// SSM/GQA intermediates arrays (per layer) — see wubu_ssm_backward / wubu_gqa_backward
// For MoE: gradient passes through (identity backward)
// Backward pass from embeddings
void wubu_model_backward_from_embd(
    const wubu_model_t *model,
    const float *embeddings,
    const float *logits, const float *d_logits,
    const float *saved_normed,     // [n_layers * N * D_MODEL]
    const float *saved_attn_out,   // [n_layers * N * D_MODEL]
    const float *saved_normed2,    // [n_layers * N * D_MODEL]
    const float *saved_ffn_out,    // [n_layers * N * D_MODEL]
    float *d_embeddings,
    int B, int T);

// MTP: Load MTP head from a separate GGUF model file
// Must be called AFTER wubu_model_init on the main model
// Pass the MTP GGUF model path
bool wubu_mtp_load(mtp_head_t *mtp, const char *mtp_gguf_path,
                   gguf_ctx *main_ctx, const uint8_t *main_blob);

// MTP: Draft forward — predict next tokens from last hidden state
// x: [D_MODEL] — last hidden state from main model (layer 39 output, post-residual)
// token_embd: [B, D_MODEL] — embeddings of candidate continuation tokens
// B: number of draft candidates to evaluate
// logits_out: [B, vocab_size] — output logits for each candidate
// Returns: number of tokens consumed from token_embd (for KV cache tracking)
int wubu_mtp_draft_forward(wubu_model_t *model,
                           const float *x,
                           const float *token_embd, int B,
                           float *logits_out);

// Free MTP head resources
void wubu_mtp_free(mtp_head_t *mtp);

// Model state save/restore for speculative decode rollback
bool wubu_model_checkpoint(wubu_model_t *model);
void wubu_model_rollback(wubu_model_t *model);

#ifdef __cplusplus
}
#endif

#endif // WUBU_MODEL_H
