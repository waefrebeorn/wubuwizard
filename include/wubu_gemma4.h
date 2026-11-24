#ifndef WUBU_GEMMA4_H
#define WUBU_GEMMA4_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Gemma 4 12B architecture — dual-head-dim ISWA design
 * 48 layers: 40 sliding-window (HEAD_DIM=256) + 8 full-attention (HEAD_DIM=512)
 * Full at indices: 5, 11, 17, 23, 29, 35, 41, 47 (every 6th from 5)
 * KV sharing: layers 40-47 reuse KV from layers 38-39
 */

/* ---- Global constants (same for all layers) ---- */
#define G4_HIDDEN       3840
#define G4_HEADS        16
#define G4_N_LAYERS     48
#define G4_VOCAB        262144
#define G4_MAX_CTX      262144
#define G4_SLIDING_WINDOW 1024
#define G4_RMS_EPS      1e-6f
#define G4_SOFTCAP      30.0f
#define G4_FFN          15360

/* ---- Per-layer-type constants ---- */
/* Sliding window layers (40 of 48) */
#define G4_SW_HEAD_DIM    256
#define G4_SW_Q_DIM       4096   /* 16 * 256 */
#define G4_SW_KV_DIM      2048   /*  8 * 256 */
#define G4_SW_KV_HEADS    8
#define G4_SW_N_ROT       256    /* 100% rotary */
#define G4_SW_ROPE_BASE   10000.0f

/* Full attention layers (8 of 48) */
#define G4_FULL_HEAD_DIM    512
#define G4_FULL_Q_DIM       8192   /* 16 * 512 */
#define G4_FULL_KV_DIM      512    /*  1 * 512 */
#define G4_FULL_KV_HEADS    1
#define G4_FULL_N_ROT       512    /* GGUF rope.dimension_count = full head dim; 25% active via factors */
#define G4_FULL_ROPE_BASE   1000000.0f

/* KV sharing zone: layers 40-47 reuse KV from layers 38 (SWA) / 39 (full) */
#define G4_KV_SHARE_START   40
#define G4_KV_SHARE_SWA_SRC 38
#define G4_KV_SHARE_FULL_SRC 35  /* last full-attn layer before share block; 39 is SWA */

/* ---- Helpers ---- */
/* Returns true if layer i is a full-attention layer */
static inline int g4_layer_is_full(int i) {
    return (i % 6 == 5);  /* indices 5,11,17,23,29,35,41,47 */
}

/* Returns true if layer i shares KV (no own K,V weights) */
static inline int g4_layer_shares_kv(int i) {
    /* This GGUF reports gemma4.attention.shared_kv_layers = 0: every layer
     * has its own attn_k/attn_v tensors. The old i>=40 rule skipped loading
     * those weights and fed the last 8 layers garbage K/V. */
    (void)i;
    return 0;
}

/* Returns the KV source layer index for a sharing layer */
static inline int g4_layer_kv_src(int i) {
    if (i < G4_KV_SHARE_START) return i;
    return g4_layer_is_full(i) ? G4_KV_SHARE_FULL_SRC : G4_KV_SHARE_SWA_SRC;
}

/* Per-layer dimensions */
static inline int g4_head_dim(int layer_idx) {
    return g4_layer_is_full(layer_idx) ? G4_FULL_HEAD_DIM : G4_SW_HEAD_DIM;
}
static inline int g4_q_dim(int layer_idx) {
    return g4_layer_is_full(layer_idx) ? G4_FULL_Q_DIM : G4_SW_Q_DIM;
}
static inline int g4_kv_dim(int layer_idx) {
    return g4_layer_is_full(layer_idx) ? G4_FULL_KV_DIM : G4_SW_KV_DIM;
}
static inline int g4_kv_heads(int layer_idx) {
    return g4_layer_is_full(layer_idx) ? G4_FULL_KV_HEADS : G4_SW_KV_HEADS;
}
static inline int g4_n_rot(int layer_idx) {
    return g4_layer_is_full(layer_idx) ? G4_FULL_N_ROT : G4_SW_N_ROT;
}
static inline float g4_rope_base(int layer_idx) {
    return g4_layer_is_full(layer_idx) ? G4_FULL_ROPE_BASE : G4_SW_ROPE_BASE;
}
static inline int g4_q_norm_size(int layer_idx) {
    return g4_layer_is_full(layer_idx) ? G4_FULL_HEAD_DIM : G4_SW_HEAD_DIM;
}

/* Max dimensions across all layers (for buffer sizing) */
#define G4_MAX_Q_DIM       G4_FULL_Q_DIM    /* 8192 */
#define G4_MAX_KV_DIM      G4_SW_KV_DIM     /* 2048 (full is smaller: 512) */
#define G4_MAX_HEAD_DIM    G4_FULL_HEAD_DIM /* 512 */
#define G4_MAX_KV_HEADS    G4_SW_KV_HEADS   /* 8 */
#define G4_MAX_KV_DIM_FULL G4_FULL_KV_DIM   /* 512 */

/* ---- Quantized weight descriptor ---- */
typedef struct {
    const uint8_t *data;
    int64_t n_elems;
    int ggml_type;
    int64_t raw_bytes;
    int data_owned;  /* 1 if data was malloc'd and needs free */
    void *stream_ctx;  /* gguf_ctx* when weight is read on demand from file */
    void *stream_info; /* gguf_tensor_info* for that weight */
    uint8_t *raw_cache; /* quantized bytes held in the RAM tier (LRU) */
    uint64_t last_use; /* LRU stamp */
    void *cache_prev, *cache_next;
    int pinned;        /* 1 = residency hot set: never evicted from the RAM tier */
    void *dev;         /* device copy of the quantized bytes (CUDA path), or NULL */
} g4_qweight_t;

/* ---- KV cache entry (per-layer) ---- */
typedef struct {
    float *k;   /* [size, kv_heads, head_dim] - heads/dim vary by layer type */
    float *v;
    int size;
    int max;
    int kv_heads;
    int head_dim;
} g4_kv_cache_t;

/* ---- Single layer ---- */
typedef struct {
    int layer_idx;
    int is_full;     /* 0=sliding, 1=full */
    int share_kv;    /* 1=this layer reuses KV from another layer */
    int kv_src_idx;  /* which layer's KV cache to use */
    
    int head_dim;
    int q_dim;
    int kv_dim;
    int kv_heads;
    int n_rot;
    float rope_base;
    
    /* Norms (F32 — tiny <15KB each) */
    float *attn_norm_weight;          /* [3840] */
    float *attn_q_norm_weight;        /* [head_dim per layer type] */
    float *attn_k_norm_weight;        /* [head_dim per layer type] */
    float *post_attn_norm_weight;     /* [3840] */
    float *ffn_norm_weight;           /* [3840] */
    float *post_ffn_norm_weight;      /* [3840] */
    
    /* Large weight matrices (quantized) */
    g4_qweight_t attn_q;              /* [3840, q_dim] */
    g4_qweight_t attn_k;              /* [3840, kv_dim] — may be empty if share_kv */
    g4_qweight_t attn_v;              /* [3840, kv_dim] — may be aliased to attn_k */
    g4_qweight_t attn_out;            /* [q_dim, 3840] */
    g4_qweight_t ffn_gate;            /* [3840, 15360] */
    g4_qweight_t ffn_up;              /* [3840, 15360] */
    g4_qweight_t ffn_down;            /* [15360, 3840] */
    
    /* Optional: layer output scale */
    float layer_out_scale;
    bool has_out_scale;
    
    /* RoPE frequencies for full attention (F32 [HEAD_DIM/2]) */
    float *rope_freqs;
    bool has_rope_freqs;
    
    /* Is V = K? (full-attention layers tie K=V) */
    bool kv_eq;
} g4_layer_t;

/* ---- Complete model ---- */
typedef struct {
    int n_layers;
    g4_layer_t *layers;
    
    /* Global tensors */
    g4_qweight_t token_embd;   /* [3840, 262144] */
    g4_qweight_t output;       /* [3840, 262144] — tied to token_embd if no separate tensor */
    bool tied_output;
    float *output_norm_weight; /* [3840] F32 */
    /* Opt-in LM-head cache: dequantize the tied embedding once so decode is a
     * plain matvec instead of re-dequantizing ~4 GB per token. */
    float *lm_head_f32;        /* [3840 * 262144] F32, or NULL */
    
    /* GGUF data blob */
    void *data_blob;
    size_t data_blob_size;
    int data_blob_is_mmap;
    void *data_blob_mmap_base;
    size_t data_blob_mmap_len;
    
    /* KV cache (per-layer) */
    g4_kv_cache_t *kv_cache;
    
    /* Context state */
    int max_ctx;
    int current_pos;
    
    /* Reusable buffers (sized for largest layer type) */
    float *buf_q;         /* [N, 8192] full-attn Q */
    float *buf_k;         /* [N, 2048] sliding K (larger) */
    float *buf_v;         /* [N, 2048] sliding V (larger) */
    float *buf_attn_out;  /* [N, 8192] full-attn output */
    float *buf_normed;    /* [N, 3840] */
    float *buf_embd;      /* [N, 3840] */
    float *buf_ffn_gate;  /* [N, 15360] */
    float *buf_ffn_up;    /* [N, 15360] */
    float *buf_ffn_out;   /* [N, 3840] */
    float *buf_scores;    /* [N, 16, 1024] sliding window max */
    float *buf_row;       /* [max(HIDDEN, FFN, 8192)] */
    float *buf_attn_full; /* [N, 16, 512] full-attn scores (N*16*MAX_CTX = huge — dynamic alloc) */
    int buf_size;

    /* GGUF file handle for streaming weight upload to GPU.
     * In G4_STREAM_WEIGHTS mode stream_ctx owns this ctx (and the file). */
    void *stream_ctx;
    FILE *gguf_file;           /* File handle, kept open for weight streaming */
    uint64_t gguf_data_offset; /* Offset in file where data blob starts */
    int gguf_n_tensors;        /* Number of tensors */
    void *gguf_tensor_info;    /* Array of gguf_tensor_info for all tensors */
    char gguf_path[1024];      /* For opening a second file handle for prefetch */
    void *prefetch;            /* g4_prefetch_t* (opaque, Windows only) */
} g4_model_t;

/* ---- API ---- */
bool g4_model_init(g4_model_t *model, const char *gguf_path);
void g4_model_destroy(g4_model_t *model);
void g4_model_reset(g4_model_t *model);

/* Forward passes */
void g4_model_forward(g4_model_t *model, const float *embeddings, int B, int T, float *logits);
void g4_model_forward_from_tokens(g4_model_t *model, const int *tokens, int B, int T, float *logits);
void g4_model_decode(g4_model_t *model, int token, float *logits);

/* Math utilities */
void g4_rms_norm(const float *x, const float *weight, int n, float eps, float *out);
float g4_gelu_tanh(float x);
void g4_softcap(float *logits, int n, float cap);

/* Quantized matmul: c[M,N] = x[M,K] @ dequant(W[K,N]) — uses quantized_matmul from quantized_matmul.c */
void g4_qmatmul_xw(const float *x, const g4_qweight_t *w, int M, int N, int K,
                   float *c, float *row_buf);

/* Internal forward functions — called from CPU and GPU paths */
void g4_batched_qmatmul(const float *x, const g4_qweight_t *w, int M, int N, int K, float *c);
void g4_layer_forward_cpu(g4_model_t *model, int il, float *x,
                          float *normed, float *attn_out,
                          float *q, float *k, float *v,
                          float *ffn_gate, float *ffn_up, float *ffn_down_out,
                          float *scores, float *row, int N, const int *positions);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_GEMMA4_H */
