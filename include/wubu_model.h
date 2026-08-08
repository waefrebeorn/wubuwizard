#ifndef WUBU_MODEL_H
#define WUBU_MODEL_H

#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "wubu_dense_ffn.h" /* opaque dense FFN for hybrid layers */
#include "wubu_safetensors_shard.h"
#include "wubu_kvcache_quant.h"
#include "wubu_kv_select.h"
#include "wubu_kv_runtime.h"
#include "wubu_kvvq.h"  /* KB2: VQ codebook for KV compression */
#include "wubu_arena.h" /* C01: arena allocator for forward-buffer OOM safety */
#include "wubu_gguf_names.h"  /* role-based tensor resolver */
#include <stdbool.h>
#include <math.h>
#include <string.h>

/* ADR-003 (kv-cache-is-a-filesystem): forward declarations only —
 * wubu_model.h stays include-free of wubu_kvfs.h. The kvfs handle is
 * an opaque resolved-path pointer created once and reused for hot I/O;
 * the layer-handles array gives the speed kernel O(1) per-layer KV
 * addressing with zero string ops. */
typedef struct wubu_kvfs wubu_kvfs_t;
typedef struct wubu_kvfs_handle wubu_kvfs_handle_t;

#ifdef __cplusplus
extern "C" {
#endif

// Layer configuration
typedef struct wubu_layer_t {
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

    // Dense SwiGLU FFN (hybrid models: Qwen3.5 family use dense
    // ffn_gate/up/down, no MoE exps). NULL when the layer is MoE.
    wubu_dense_ffn_t *dense_ffn;
} wubu_layer_t;

// Complete model
#define GQA_MAX_CTX 8192  // default max cached positions (overridden by WUBU_MAX_CTX env)
// At 512K context, SWA (sliding window) handles long-range attention.
// KV cache is pre-allocated to this size; larger contexts use auto-eviction.
// Set WUBU_MAX_CTX=524288 to enable full 512K pre-allocation (needs ~64GB RAM).
/* GQA_KV_DIM is provided by wubu_dims.h (WUBU_DIMS.gqa_kv_dim) so it
 * resolves to the model's real kv dim at load time. */

// KV cache format: 0=F32, 1=F16 (halves memory at cost of conversion)
#ifndef KV_CACHE_F16
#define KV_CACHE_F16 1  // default to F16 for memory efficiency
#endif

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

// Our own Q8_0 KV-cache block (8-bit, block-32 absmax symmetric) -- routes to
// the tested wubu_kvcache_quant module. 4:1 vs F16, near-lossless (Roofline +
// llama.cpp + KIVI convergence: decode is BW-bound, halving KV bytes = faster).
typedef struct {
    int8_t qs[32];   // 32 int8 values
    float  d;        // absmax scale (fp32)
} block_q8_0_cache;
#define QK8_CACHE 32

// KB1: Adaptive KV block (doc 001, Ecco). Variable bit-width per block:
// 2-bit, 4-bit, or 8-bit stored as width_bits + scale + packed bytes.
// Layout: 16 bytes (width_bits + scale + 14 bytes packed) = same as Q4_0 block.
typedef struct {
    uint8_t width_bits;   // 2, 4, or 8
    uint8_t _pad[3];      // alignment padding
    float   scale;        // absmax scale
    uint8_t qs[24];       // packed values (32 × 8-bit max, fewer for 2/4-bit)
} block_adaptive_cache;
#define ADAPTIVE_CACHE 32

// KIVI per-token V block: head_dim used at alloc time to size the per-token
// fp32 scales. Must match the model's attention head_dim (Qwen-class = 128).
#ifndef KV_KIVI_HEADDIM
#define KV_KIVI_HEADDIM 128
#endif

// KIVI layout note: one fp32 scale per token's head_dim int8 values.
// (KIVI paper: V per-token, K per-channel. We store K as Q8_0-block which is
// per-block absmax -- near the per-channel intent at block granularity, and is
// computed streamingly; V per-token is exact KIVI since each write is 1 token.)
// Storage per token = head_dim int8 + 1 fp32 scale.
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
// KV cache read: one head (n floats). DISPATCHES on the runtime g_kv_scheme
// (set at model load by the Roofline auto-selector) instead of a compile-time
// #if, so the engine picks precision per-model. Per-scheme bodies below.
extern int g_kv_scheme;            /* defined in wubu_kv_runtime.c */
extern void wubu_kv_set_scheme(int);
static inline void kv_cache_read_head_q4(const void *cache, int64_t offset, float *buf, int n);
static inline void kv_cache_read_head_q8(const void *cache, int64_t offset, float *buf, int n);
static inline void kv_cache_read_head_kivi(const void *cache, int64_t offset, float *buf, int n);
static inline void kv_cache_read_head_adaptive(const void *cache, int64_t offset, float *buf, int n);
static inline void kv_cache_read_head_f16(const void *cache, int64_t offset, float *buf, int n);
static inline void kv_cache_read_head_f32(const void *cache, int64_t offset, float *buf, int n);

static inline void kv_cache_read_head(const void *cache, int64_t offset,
                                       float *buf, int n) {
    switch (g_kv_scheme) {
        case WUBU_KV_Q4_0: kv_cache_read_head_q4(cache, offset, buf, n); break;
        case WUBU_KV_Q8:   kv_cache_read_head_q8(cache, offset, buf, n); break;
        case WUBU_KV_KIVI: kv_cache_read_head_kivi(cache, offset, buf, n); break;
        case WUBU_KV_ADAPTIVE: kv_cache_read_head_adaptive(cache, offset, buf, n); break;
        case WUBU_KV_F16:  kv_cache_read_head_f16(cache, offset, buf, n); break;
        case WUBU_KV_4KV:  kv_cache_read_head_f32(cache, offset, buf, n); break; /* 4KV uses F32+quant at layer level */
        case WUBU_KV_3BIT: kv_cache_read_head_f32(cache, offset, buf, n); break; /* 3BIT uses F32+quant at layer level */
        default:           kv_cache_read_head_f32(cache, offset, buf, n); break;
    }
}

static inline void kv_cache_read_head_q4(const void *cache, int64_t offset,
                                         float *buf, int n) {
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
}

static inline void kv_cache_read_head_q8(const void *cache, int64_t offset,
                                         float *buf, int n) {
    // Our Q8_0 block-32 (near-lossless, routes to wubu_kvcache_quant).
    const int block_n = QK8_CACHE;
    int start_block = (int)(offset / block_n);
    int start_elem = (int)(offset % block_n);
    const block_q8_0_cache *blocks = (const block_q8_0_cache *)cache;
    int done = 0;
    while (done < n) {
        float tmp[QK8_CACHE];
        wubu_kvq_q8_dequant(blocks[start_block + (start_elem + done) / block_n].qs,
                             blocks[start_block + (start_elem + done) / block_n].d,
                             tmp, block_n);
        int blk_off = (start_elem + done) % block_n;
        int to_copy = n - done;
        if (to_copy > block_n - blk_off) to_copy = block_n - blk_off;
        for (int i = 0; i < to_copy; i++) buf[done + i] = tmp[blk_off + i];
        done += to_copy;
    }
}

static inline void kv_cache_read_head_kivi(const void *cache, int64_t offset,
                                           float *buf, int n) {
    int hd = g_kv_head_dim > 0 ? g_kv_head_dim : KV_KIVI_HEADDIM;
    if (n == hd) {
        // Fast path: single token
        int t0 = (int)(offset / hd), p0 = (int)(offset % hd);
        const uint8_t *base = (const uint8_t *)cache + (size_t)t0 * (hd + (int)sizeof(float));
            const uint8_t *q = base;
            float scale = *(const float *)(base + hd);
            float tmp[512];
            int work_hd = hd > 512 ? 512 : hd;
            wubu_kvq_kivi_dequant_V(q, &scale, tmp, 1, work_hd);
            for (int i = 0; i < n; i++) buf[i] = tmp[p0 + i];
        } else {
        // Batch path: read multiple tokens
        int tokens = n / hd;
        for (int t = 0; t < tokens; t++) {
            int token_offset = offset + t * hd;
            int t0 = (int)(token_offset / hd), p0 = (int)(token_offset % hd);
            const uint8_t *base = (const uint8_t *)cache + (size_t)t0 * (hd + (int)sizeof(float));
            const uint8_t *q = base;
            float scale = *(const float *)(base + hd);
            float tmp[512];
            int work_hd = hd > 512 ? 512 : hd;
            wubu_kvq_kivi_dequant_V(q, &scale, tmp, 1, work_hd);
            for (int i = 0; i < hd; i++) buf[t * hd + i] = tmp[p0 + i];
        }
    }
}

static inline void kv_cache_read_head_f16(const void *cache, int64_t offset,
                                          float *buf, int n) {
    const uint16_t *src = (const uint16_t *)cache + offset;
    for (int i = 0; i < n; i++) buf[i] = fp16_to_fp32(src[i]);
}

/* KB1: Adaptive KV read (doc 001). Uses wubu_kvq_adaptive_dequant
 * for per-block variable bit-width dequantization. */
static inline void kv_cache_read_head_adaptive(const void *cache, int64_t offset,
                                                  float *buf, int n) {
    const int block_n = ADAPTIVE_CACHE;
    int start_block = (int)(offset / block_n);
    int start_elem = (int)(offset % block_n);
    const block_adaptive_cache *blocks = (const block_adaptive_cache *)cache;
    int done = 0;
    while (done < n) {
        float tmp[ADAPTIVE_CACHE];
        const block_adaptive_cache *blk = &blocks[start_block + (start_elem + done) / block_n];
        /* Dequantize: unpack based on width_bits */
        int bits = blk->width_bits;
        float scale = blk->scale;
        int vals_per_byte = (bits == 2) ? 4 : (bits == 4) ? 2 : 1;
        for (int i = 0; i < block_n; i++) {
            int byte_idx = i / vals_per_byte;
            int pos = i % vals_per_byte;
            uint8_t raw;
            if (bits == 2) {
                raw = (blk->qs[byte_idx] >> (2 * pos)) & 0x3;
            } else if (bits == 4) {
                raw = (blk->qs[byte_idx] >> (4 * pos)) & 0xF;
            } else {
                raw = blk->qs[byte_idx];
            }
            /* Convert to signed: unsigned [0, qmax] → signed [-half, half-1] */
            int half = (1 << (bits - 1));
            int sq = (int)raw - half;
            tmp[i] = (float)sq * scale;
        }
        int blk_off = (start_elem + done) % block_n;
        int to_copy = n - done;
        if (to_copy > block_n - blk_off) to_copy = block_n - blk_off;
        for (int i = 0; i < to_copy; i++) buf[done + i] = tmp[blk_off + i];
        done += to_copy;
    }
}

static inline void kv_cache_read_head_f32(const void *cache, int64_t offset,
                                          float *buf, int n) {
    memcpy(buf, (const float *)cache + offset, n * sizeof(float));
}

// Batch write one head to KV cache (runtime dispatch).
static inline void kv_cache_write_head_q4(void *cache, int64_t offset, const float *buf, int n);
static inline void kv_cache_write_head_q8(void *cache, int64_t offset, const float *buf, int n);
static inline void kv_cache_write_head_kivi(void *cache, int64_t offset, const float *buf, int n);
static inline void kv_cache_write_head_adaptive(void *cache, int64_t offset, const float *buf, int n);
static inline void kv_cache_write_head_f16(void *cache, int64_t offset, const float *buf, int n);
static inline void kv_cache_write_head_f32(void *cache, int64_t offset, const float *buf, int n);
static inline void kv_cache_write_head(void *cache, int64_t offset,
                                        const float *buf, int n) {
    switch (g_kv_scheme) {
        case WUBU_KV_Q4_0: kv_cache_write_head_q4(cache, offset, buf, n); break;
        case WUBU_KV_Q8:   kv_cache_write_head_q8(cache, offset, buf, n); break;
        case WUBU_KV_KIVI: kv_cache_write_head_kivi(cache, offset, buf, n); break;
        case WUBU_KV_ADAPTIVE: kv_cache_write_head_adaptive(cache, offset, buf, n); break;
        case WUBU_KV_F16:  kv_cache_write_head_f16(cache, offset, buf, n); break;
        case WUBU_KV_4KV:  kv_cache_write_head_f32(cache, offset, buf, n); break;
        case WUBU_KV_3BIT: kv_cache_write_head_f32(cache, offset, buf, n); break;
        default:           kv_cache_write_head_f32(cache, offset, buf, n); break;
    }
}

static inline void kv_cache_write_head_q4(void *cache, int64_t offset,
                                           const float *buf, int n) {
    const int block_n = QK4_CACHE;
    int start_block = (int)(offset / block_n);
    int start_elem = (int)(offset % block_n);
    int end_elem = start_elem + n;
    block_q4_0_cache *blocks = (block_q4_0_cache *)cache;
    if (start_elem == 0) {
        int n_aligned = n - (end_elem % block_n);
        if (n_aligned < 0) n_aligned = 0;
        for (int bi = 0; bi < n_aligned / block_n; bi++) {
            quantize_q4_0_cache_block(buf + bi * block_n, &blocks[start_block + bi]);
        }
        int rem = n - n_aligned;
        if (rem > 0) {
            int bi = n_aligned / block_n;
            float tmp[QK4_CACHE];
            dequantize_q4_0_cache_block(&blocks[start_block + bi], tmp);
            for (int i = 0; i < rem; i++) tmp[i] = buf[n_aligned + i];
            quantize_q4_0_cache_block(tmp, &blocks[start_block + bi]);
        }
    } else {
        int first_rem = block_n - start_elem;
        if (first_rem > n) first_rem = n;
        {
            float tmp[QK4_CACHE];
            dequantize_q4_0_cache_block(&blocks[start_block], tmp);
            for (int i = 0; i < first_rem; i++) tmp[start_elem + i] = buf[i];
            quantize_q4_0_cache_block(tmp, &blocks[start_block]);
        }
        int remaining = n - first_rem;
        if (remaining > 0) {
            kv_cache_write_head_q4(cache, offset + first_rem, buf + first_rem, remaining);
        }
    }
}

static inline void kv_cache_write_head_q8(void *cache, int64_t offset,
                                          const float *buf, int n) {
    const int block_n = QK8_CACHE;
    int start_block = (int)(offset / block_n);
    int start_elem = (int)(offset % block_n);
    block_q8_0_cache *blocks = (block_q8_0_cache *)cache;
    int done = 0;
    while (done < n) {
        int bn = block_n - ((start_elem + done) % block_n);
        if (bn > n - done) bn = n - done;
        int blk = start_block + (start_elem + done) / block_n;
        if (bn == block_n) {
            wubu_kvq_q8_quant(buf + done, blocks[blk].qs, &blocks[blk].d, block_n);
        } else {
            float tmp[QK8_CACHE];
            wubu_kvq_q8_dequant(blocks[blk].qs, blocks[blk].d, tmp, block_n);
            int blk_off = (start_elem + done) % block_n;
            for (int i = 0; i < bn; i++) tmp[blk_off + i] = buf[done + i];
            wubu_kvq_q8_quant(tmp, blocks[blk].qs, &blocks[blk].d, block_n);
        }
        done += bn;
    }
}

static inline void kv_cache_write_head_kivi(void *cache, int64_t offset,
                                            const float *buf, int n) {
    int hd = g_kv_head_dim > 0 ? g_kv_head_dim : KV_KIVI_HEADDIM;
    if (n == hd) {
        // Fast path: single token
        int t0 = (int)(offset / hd), p0 = (int)(offset % hd);
        uint8_t *base = (uint8_t *)cache + (size_t)t0 * (hd + (int)sizeof(float));
        uint8_t *q = base;
        float scale = *(const float *)(base + hd);
        float tmp[512];
        int work_hd = hd > 512 ? 512 : hd;
        wubu_kvq_kivi_dequant_V(q, &scale, tmp, 1, work_hd);
        for (int i = 0; i < n; i++) tmp[p0 + i] = buf[i];
        wubu_kvq_kivi_quant_V(tmp, q, &scale, 1, work_hd);
        *(float *)(base + hd) = scale;
    } else {
        // Batch path: write multiple tokens
        int tokens = n / hd;
        for (int t = 0; t < tokens; t++) {
            int token_offset = offset + t * hd;
            int t0 = (int)(token_offset / hd), p0 = (int)(token_offset % hd);
            uint8_t *base = (uint8_t *)cache + (size_t)t0 * (hd + (int)sizeof(float));
            uint8_t *q = base;
            float scale = *(const float *)(base + hd);
            float tmp[512];
            int work_hd = hd > 512 ? 512 : hd;
            wubu_kvq_kivi_dequant_V(q, &scale, tmp, 1, work_hd);
            for (int i = 0; i < hd; i++) tmp[p0 + i] = buf[t * hd + i];
            wubu_kvq_kivi_quant_V(tmp, q, &scale, 1, work_hd);
            *(float *)(base + hd) = scale;
        }
    }
}

static inline void kv_cache_write_head_f16(void *cache, int64_t offset,
                                           const float *buf, int n) {
    uint16_t *dst = (uint16_t *)cache + offset;
    for (int i = 0; i < n; i++) dst[i] = fp32_to_fp16(buf[i]);
}

static inline void kv_cache_write_head_f32(void *cache, int64_t offset,
                                           const float *buf, int n) {
    memcpy((float *)cache + offset, buf, n * sizeof(float));
}

/* KB1: Adaptive KV write (doc 001). Uses Ecco entropy-aware bit-width
 * selection: low-variance blocks → 2-bit, high-variance → 8-bit. */
static inline void kv_cache_write_head_adaptive(void *cache, int64_t offset,
                                                 const float *buf, int n) {
    const int block_n = ADAPTIVE_CACHE;
    int start_block = (int)(offset / block_n);
    int start_elem = (int)(offset % block_n);
    block_adaptive_cache *blocks = (block_adaptive_cache *)cache;
    int done = 0;
    while (done < n) {
        int bn = block_n - ((start_elem + done) % block_n);
        if (bn > n - done) bn = n - done;
        int blk_idx = start_block + (start_elem + done) / block_n;
        int blk_off = (start_elem + done) % block_n;

        if (bn == block_n) {
            /* Full block: quantize directly */
            float tmp[block_n];
            for (int i = 0; i < block_n; i++) tmp[i] = buf[done + i];
            /* Compute absmax and variance for bit-width selection */
            float amax = 0.0f, mean = 0.0f;
            for (int i = 0; i < block_n; i++) {
                float ax = fabsf(tmp[i]);
                if (ax > amax) amax = ax;
                mean += tmp[i];
            }
            mean /= (float)block_n;
            float var = 0.0f;
            for (int i = 0; i < block_n; i++) {
                float d = tmp[i] - mean;
                var += d * d;
            }
            var /= (float)block_n;
            /* Select bit-width: low variance → fewer bits */
            int bits = (var < 0.01f) ? 2 : (var < 0.1f) ? 4 : 8;
            int half = (1 << (bits - 1));
            /* Symmetric signed: [-amax, +amax] → [-half, half-1] */
            /* scale maps amax to half-1 so the full range is representable */
            float scale = (amax > 1e-8f) ? amax / (float)(half - 1) : 0.0f;
            if (scale == 0.0f) scale = 1e-8f;
            blocks[blk_idx].width_bits = (uint8_t)bits;
            blocks[blk_idx].scale = scale;
            /* Pack values — symmetric signed quantization */
            int vals_per_byte = (bits == 2) ? 4 : (bits == 4) ? 2 : 1;
            memset(blocks[blk_idx].qs, 0, sizeof(blocks[blk_idx].qs));
            for (int i = 0; i < block_n; i++) {
                int q = (int)roundf(tmp[i] / scale);
                /* Clamp to signed range [-half, half-1] */
                if (q >= half) q = half - 1;
                if (q < -half) q = -half;
                /* Convert to unsigned: [-half, half-1] → [0, qmax] */
                q += half;
                int byte_idx = i / vals_per_byte;
                int pos = i % vals_per_byte;
                if (bits == 2) {
                    blocks[blk_idx].qs[byte_idx] |= (uint8_t)(q << (2 * pos));
                } else if (bits == 4) {
                    blocks[blk_idx].qs[byte_idx] |= (uint8_t)(q << (4 * pos));
                } else {
                    blocks[blk_idx].qs[byte_idx] = (uint8_t)q;
                }
            }
        } else {
            /* Partial block: read existing, merge, re-quantize */
            float tmp[block_n];
            /* Dequantize existing block */
            const block_adaptive_cache *blk = &blocks[blk_idx];
            int bits = blk->width_bits;
            float scale = blk->scale;
            int vals_per_byte = (bits == 2) ? 4 : (bits == 4) ? 2 : 1;
            int qmax = (1 << bits) - 1;
            int half = (1 << (bits - 1));
            for (int i = 0; i < block_n; i++) {
                int byte_idx = i / vals_per_byte;
                int pos = i % vals_per_byte;
                uint8_t raw;
                if (bits == 2) raw = (blk->qs[byte_idx] >> (2 * pos)) & 0x3;
                else if (bits == 4) raw = (blk->qs[byte_idx] >> (4 * pos)) & 0xF;
                else raw = blk->qs[byte_idx];
                int sq = (int)raw;
                if (sq >= half) sq -= (qmax + 1);
                tmp[i] = (float)sq * scale;
            }
            /* Overwrite the new elements */
            for (int i = 0; i < bn; i++) tmp[blk_off + i] = buf[done + i];
            /* Re-quantize the whole block */
            float amax = 0.0f, mean = 0.0f;
            for (int i = 0; i < block_n; i++) {
                float ax = fabsf(tmp[i]);
                if (ax > amax) amax = ax;
                mean += tmp[i];
            }
            mean /= (float)block_n;
            float var = 0.0f;
            for (int i = 0; i < block_n; i++) { float d = tmp[i] - mean; var += d * d; }
            var /= (float)block_n;
            int new_bits = (var < 0.01f) ? 2 : (var < 0.1f) ? 4 : 8;
            int nhalf = (1 << (new_bits - 1));
            /* Symmetric signed: [-amax, +amax] → [-nhalf, nhalf-1] */
            float new_scale = (amax > 1e-8f) ? amax / (float)(nhalf - 1) : 1e-8f;
            blocks[blk_idx].width_bits = (uint8_t)new_bits;
            blocks[blk_idx].scale = new_scale;
            int nvpb = (new_bits == 2) ? 4 : (new_bits == 4) ? 2 : 1;
            memset(blocks[blk_idx].qs, 0, sizeof(blocks[blk_idx].qs));
            for (int i = 0; i < block_n; i++) {
                int q = (int)roundf(tmp[i] / new_scale);
                if (q >= nhalf) q = nhalf - 1;
                if (q < -nhalf) q = -nhalf;
                q += nhalf;
                int byte_idx = i / nvpb;
                int pos = i % nvpb;
                if (new_bits == 2) blocks[blk_idx].qs[byte_idx] |= (uint8_t)(q << (2 * pos));
                else if (new_bits == 4) blocks[blk_idx].qs[byte_idx] |= (uint8_t)(q << (4 * pos));
                else blocks[blk_idx].qs[byte_idx] = (uint8_t)q;
            }
        }
        done += bn;
    }
}

// KV cache allocation: returns number of bytes needed for n_elems (runtime dispatch).
// Uses model's actual head_dim via g_kv_head_dim (set by wubu_kv_autoselect).
static inline int64_t kv_cache_alloc_size(int64_t n_elems) {
    switch (g_kv_scheme) {
        case WUBU_KV_Q4_0: {
            int64_t n_blocks = (n_elems + QK4_CACHE - 1) / QK4_CACHE;
            return n_blocks * (int64_t)sizeof(block_q4_0_cache);
        }
        case WUBU_KV_Q8: {
            int64_t n_blocks = (n_elems + QK8_CACHE - 1) / QK8_CACHE;
            return n_blocks * (int64_t)sizeof(block_q8_0_cache);
        }
        case WUBU_KV_KIVI: {
            int64_t hd = g_kv_head_dim > 0 ? g_kv_head_dim : KV_KIVI_HEADDIM;
            int64_t tokens = (n_elems + hd - 1) / hd;
            return n_elems * (int64_t)sizeof(int8_t) + tokens * (int64_t)sizeof(float);
        }
        case WUBU_KV_ADAPTIVE: {
            int64_t n_blocks = (n_elems + ADAPTIVE_CACHE - 1) / ADAPTIVE_CACHE;
            return n_blocks * (int64_t)sizeof(block_adaptive_cache);
        }
        case WUBU_KV_F16:
            return n_elems * (int64_t)sizeof(uint16_t);
        case WUBU_KV_4KV:
            /* SAW-INT4: 0.5 bytes/elem (4-bit) + per-block-16 scales */
            return (n_elems / 2) + (n_elems / 16) * (int64_t)sizeof(float) + 8;
        case WUBU_KV_3BIT:
            /* TurboQuant INT3: ~0.375 bytes/elem (3-bit) + per-token scales */
            return (n_elems * 3 + 7) / 8 + (n_elems / 64) * (int64_t)sizeof(float) + 8;
        default:
            return n_elems * (int64_t)sizeof(float);
    }
}

// MTP (Multi-Token Prediction) head for speculative decode
// Architecture: h_39 → hnorm → concat(hnorm, enorm(embd)) → eh_proj → blk.40 → shared_head_norm → output
typedef struct mtp_head_t {
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
    float *k_cache;  // [GQA_MAX_CTX * kv_dim]
    float *v_cache;  // [GQA_MAX_CTX * kv_dim]
    int kv_dim;      // per-layer KV dim (kv_heads * head_dim)
    int cache_len;
} mtp_head_t;
typedef struct wubu_model_t {
    int n_layers;
    wubu_layer_t *layers;
    
    // Token embedding
    float *token_embd;       // [vocab_size, D_MODEL] or NULL if using embedding file
    float *output_weight;    // [D_MODEL, vocab_size] or NULL
    const uint8_t *output_weight_q;   // raw Q4_K quantized
    int output_weight_type;
    bool tied_output;                // true if output_weight_q points to token_embd
    const uint8_t *token_embd_q;     // raw Q4_K quantized token embeddings (large vocab, mmap'd)
    int token_embd_type;             // GGML type of token_embd (usually Q4_K)
    int rotate_P;                    // doc 013: Hadamard prefix fused into output_weight (>1 if WUBU_ROTATE_W)
    // Lazy, zero-copy embedding / lm_head (safetensors BF16 path). When set,
    // token_embd / output_weight are NOT copied to F32; the row is dequantized
    // on demand from the mmap'd shard. Saves ~10 GB for 27B-class models.
    const uint8_t *lazy_embd_raw;     // raw bytes of embed_tokens (row-major)
    int            lazy_embd_dtype;   // ST_DTYPE_*
    int64_t        lazy_embd_row;     // elems per embedding row (= D_MODEL)
    const uint8_t *lazy_lmhead_raw;   // raw bytes of lm_head.weight [vocab, D]
    int            lazy_lmhead_dtype;
    int64_t        lazy_lmhead_row;   // elems per lm_head row (= D_MODEL)

    /* Persistent shard context for lazy embed/lm_head mmap access.
     * Must stay alive for the lifetime of the model. */
    struct wubu_shard_ctx *shard_ctx;

    // Embedding file (from Phase 1)
    bool use_embedding_file;
    int vocab_size;
    
    // Norms
    float *norm_weight;  // final RMSNorm [D_MODEL]
    
    // State buffers (reused across calls)
    float *ssm_states;    // [max_layers, SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
    float *conv_states;   // [max_layers, B, CONV_KERNEL-1, CONV_DIM]
    size_t ssm_state_total;  // bytes allocated for ssm_states (incl. conv_states)    
    // GQA KV cache (10 GQA layers, max 256k context)
    void *gqa_k_cache;  // [n_gqa_layers * runtime_max_ctx * kv_dim] F32 or F16
    void *gqa_v_cache;  // [n_gqa_layers * runtime_max_ctx * kv_dim]
    int gqa_cache_len;   // how many tokens cached per layer (all layers same len)
    int gqa_max_ctx;     // runtime max ctx (GQA_MAX_CTX or WUBU_MAX_CTX env)

    // GGUF context (for per-layer MoE lazy loading)
    // Model state save/restore (for speculative decode rollback)
    float *ssm_states_saved;    // same size as ssm_states
    float *conv_states_saved;   // same size as conv_states
    int gqa_cache_len_saved;
    int mtp_cache_len_saved;

    // AirLLM layer streaming: when set, force chunked prefill in gen_text
    // even for single calls. The budget system sets this when max_ctx < 256
    // (KV cache too large for available RAM — must stream layers).
    int use_layer_stream;

    // GGUF context (for per-layer MoE lazy loading)
    gguf_ctx *gguf_ctx;
    size_t    data_blob_size;  // size of GGUF data blob (for budget calc)
    
    // Enable MoE during forward (default: false for memory reasons)
    bool enable_moe;
    
    // Skip output projection in forward (for GPU offload)
    bool skip_output_proj;
    
    // MoE test: only load MoE for first N layers (0 = all)
    int moe_max_layers;

    // ds4-ssd slot-bank: when non-NULL, routed MoE experts are paged from a
    // sidecar on disk (LRU) instead of held resident. Set by the bridge when
    // a sidecar directory is supplied. Forward uses wubu_moe_forward_ssd.
    wubu_ssd_moe_t *ssd_moe;
    
    // MTP (Multi-Token Prediction) head for speculative decode
    mtp_head_t mtp;
    
    // Last hidden state capture (for MTP: set to a [D_MODEL] buffer before forward)
    float *save_last_hidden;

    // GPU acceleration context (opaque pointer, managed by wubu_model_gpu.cu)
    void *gpu_ctx;

    // Active backend vtable (CPU or CUDA). Set at model init;
    // NULL when no backend is installed yet. Separated from gpu_ctx
    // because gpu_ctx is the GPU context (wubu_gpu_ctx_t*), while
    // backend points to the dispatch struct (wubu_backend_t*).
    struct wubu_backend_t *backend;

    // OOM-safe forward arena: allocated once at model init, reset per forward.
    // All temporary buffers (x, normed, attn_out, normed2, ffn_out, prev_experts)
    // come from here — no per-token malloc churn. Sized to B*T*d_model*4.
    wubu_arena_t fwd_arena;
    wubu_sub_arena_t fwd_sub;

    // Number of GQA layers (for KV cache sizing)
    int n_gqa_layers;

    // ---- HW-acceleration wiring (doc "tandem"/"rambus"/"gamebud") ----
    // hwcaps: detected SIMD ladder at model load (cached from wubu_hwcaps_get).
    int hw_simd_bits;     // 128/256/512
    int hw_simd_lanes;    // 4/8/16 floats per lane
    // rambus: KV cache laid out as interleaved banks (RDRAM-style) so decode
    // attention reads stream bank-by-bank with row-buffer hits. The flat
    // gqa_k_cache/gqa_v_cache are allocated THROUGH this arena.
    void *kv_rambus;      // wubu_rambus_t* (opaque)
    int   kv_rambus_banks;
    // gamebud: per-decode-step frame budget governor (NULL = disabled).
    void *gamebud;        // wubu_gamebud_t* (opaque)
    uint64_t frame_budget_us;  // 0 = disabled
    // tandem: N64 RCP two-stage pipeline (prefill=A, decode=B). NULL = inline.
    void *tandem;         // wubu_tandem_t* (opaque)
    // Dynamic model dimensions (extracted from GGUF, model-adapter aware)
    int d_model;          // hidden dimension (2048 for Qwen, 2816 for DiffusionGemma)
    int d_ff;             // expert intermediate dim
    int n_experts;        // total experts
    int n_active_experts; // top-k
    int shared_expert_ff; // shared expert intermediate dim (MoE); 0 if none
    int tensor_naming;    // 0=blk.Qwen 1=model.layers.Gemma 2=pure-GQA

    // Additional dynamic dimensions for GPU and other code
    int d_inner;          // SSM inner dimension / VALUE_DIM
    int key_dim;          // KEY_DIM = SSM_D_STATE * SSM_K_HEADS
    int conv_dim;         // CONV_DIM
    int conv_kernel;      // CONV_KERNEL (conv1d kernel size, default 4)
    int dt_rank;          // DT_RANK
    int ssm_k_heads;      // SSM_K_HEADS
    int ssm_v_heads;      // SSM_V_HEADS
    int ssm_d_state;      // SSM_D_STATE
    int gqa_q_heads;      // GQA_Q_HEADS
    int gqa_kv_heads;     // GQA_KV_HEADS
    int gqa_head_dim;     // GQA_HEAD_DIM
    int rotary_dim;       // ROTARY_DIM (RoPE rotation dim)

    // ---- ADR-003: KV cache is a file system ----
    // Path-addressable KV namespace (wubu_kvfs). Each GQA layer's KV
    // block is mounted at /kv/layer_XX; the speed kernel and external
    // 9P clients read/write KV data by path through this namespace.
    // NULL until wubu_model_init wires it (allocation failure is
    // non-fatal — the flat gqa_k_cache/gqa_v_cache tensors remain the
    // authoritative store).
    wubu_kvfs_t *kvfs;
    // Per-layer KV block size (floats per layer = gqa_max_ctx * kv_dim).
    // Uniform across GQA layers for the mounted namespace; 0 if no kvfs.
    size_t kvfs_block_floats;
    int    kvfs_n_layers;  // GQA layers mounted in the namespace
    // Resolve-once handles: one per GQA layer, created at init by
    // wubu_kvfs_open("/kv/layer_XX"). The speed kernel grabs the
    // handle and does bounds-checked memcpy I/O with ZERO string
    // ops per access. NULL for SSM layers; array NULL if no kvfs.
    wubu_kvfs_handle_t **kvfs_layer_handles;
    int kvfs_n_handles;    // allocated handle slots (== n_layers)
} wubu_model_t;

// Create model, load from GGUF
bool wubu_model_init(wubu_model_t *model, const char *gguf_path);

// Free model resources
void wubu_model_free(wubu_model_t *model);

// ---- ADR-003: KV cache is a file system ----
// Return the model's KV namespace (NULL until wubu_model_init wires it,
// or if namespace allocation failed — flat tensors remain authoritative).
wubu_kvfs_t *wubu_model_kvfs(const wubu_model_t *model);

/* Read KV data by namespace path into dst (floats). Routes through the
 * active backend (GPU-accelerable), falling back to the flat host KV
 * tensor. Returns 0 on success, -1 if the path is unmounted, the
 * namespace is missing, or the read would exceed the mount's range. */
int wubu_model_kvfs_read(wubu_model_t *model, const char *path,
                         float *dst, size_t n_floats);

/* Write KV data by namespace path from src (floats). Same routing as
 * wubu_model_kvfs_read. Returns 0 on success, -1 on failure. */
int wubu_model_kvfs_write(wubu_model_t *model, const char *path,
                          const float *src, size_t n_floats);

/* JSON snapshot of the mounted namespace (caller frees). NULL if the
 * namespace is missing. */
char *wubu_model_kvfs_snapshot_json(wubu_model_t *model, size_t *out_len);

/* ---- ADR-003 speed-kernel hot path: resolve once, use many ----
 * The per-layer handles are resolved at init. The speed kernel calls
 * wubu_model_kvfs_layer_handle(layer) once, then wubu_kvfs_handle_read/
 * write from wubu_kvfs.h — bounds-checked memcpy, zero string ops.
 * Returns NULL if the layer is an SSM layer or the namespace is
 * missing. The handle is owned by the model (do NOT close it). */
wubu_kvfs_handle_t *wubu_model_kvfs_layer_handle(const wubu_model_t *model,
                                                 int layer);

/* Resolve an arbitrary namespace path to a model-owned handle, or
 * NULL if unmounted/namespace missing. Same ownership as above. */
wubu_kvfs_handle_t *wubu_model_kvfs_open_handle(wubu_model_t *model,
                                                const char *path);

/* Handle-based read/write with backend routing: tries the active
 * backend first (device-resident KV), falls back to the flat host
 * tensor through the handle. 0 on success, -1 on failure. */
int wubu_model_kvfs_handle_read(wubu_model_t *model,
                                const wubu_kvfs_handle_t *h,
                                float *dst, size_t n_floats);
int wubu_model_kvfs_handle_write(wubu_model_t *model,
                                 const wubu_kvfs_handle_t *h,
                                 const float *src, size_t n_floats);

/* ---- HW-acceleration wiring (doc "tandem"/"rambus"/"gamebud"/hwcaps) ----
 * Wire the SIMD-ladder detect + RDRAM-interleaved KV + N64 tandem pipeline +
 * game frame-budget into an already-init'd model. Call after wubu_model_init.
 *   simd_autodetect : if true, detect CPU SIMD width (always on here).
 *   rambus_banks    : interleave factor for KV (0 = 8 default; 1 = disable).
 *   kv_dim          : per-layer KV dim (kv_heads * head_dim) for arena sizing.
 *   frame_budget_us : per-decode-step time budget (0 = disable gamebud).
 *   tandem_a/tandem_b: core lists for the two stages (NULL = OS default).
 * Returns 0 on success. Safe to call once; re-call frees prior wiring. */
int wubu_model_wire_hwaccel(wubu_model_t *model, int simd_autodetect,
                            int rambus_banks, int kv_dim, uint64_t frame_budget_us,
                            const char *tandem_a, const char *tandem_b);

/* Tear down HW-accel wiring (called by wubu_model_free). */
void wubu_model_unwire_hwaccel(wubu_model_t *model);

/* Report a one-line HW-accel status string into buf (static). */
const char *wubu_model_hwaccel_str(const wubu_model_t *model);

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

// Reset persistent SSM/conv recurrence state to zero (start a fresh sequence).
// Required before comparing two independent generations (e.g. plain vs
// speculative decode) so both start from the same zero state.
void wubu_model_reset_state(wubu_model_t *model);

// Chunked forward: process a long [B, T_total] sequence in time-chunks of
// <= chunk_sz tokens, carrying the model's persistent SSM/conv/KV-cache state
// across chunks. Mathematically identical to a single forward, but bounds peak
// memory: each chunk allocates intermediates for chunk_sz tokens only. The
// final chunk's logits (positions [T_total-chunk_sz, T_total)) are written to
// `logits` (sized B*chunk_sz*vocab_size). Use this to run the full 256K context
// on a memory-limited box.
void wubu_model_forward_chunked(wubu_model_t *model,
                                const int *token_ids, int B, int T_total,
                                int chunk_sz, float *logits);

// ================================================================
// GPU-Accelerated Forward Path — declared in wubu_model_gpu.h
// (Strangler Fig split: GPU-only consumers include that header alone.)
// ================================================================
#include "wubu_model_gpu.h"

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
                   gguf_ctx *main_ctx, const uint8_t *main_blob,
                   int gqa_max_ctx);

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
