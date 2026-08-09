/*
 * wubu.h -- WuBu-35M, our base model, in C11. THE MUSTARD SEED.
 *
 * Original WaefreBeorn work under the WaefreBeorn Umbrella License v3.0
 * (LICENSE at the repo root). WuBu-35M is a 35,072,768-parameter
 * decoder-only base language model, designed, implemented, and trained
 * in-house by the WaefreBeorn project (the "there is no third party"
 * doctrine). It grows via the AGI brain-cluster loop -- more tokens,
 * more parameters, more knowledge, all in-house.
 *
 * Architecture:
 *   - 12 layers, dim 448, 7 query heads, 1 KV head (GQA 7:1)
 *   - hybrid attention: 3 local (256-token window) layers then 1 full layer
 *   - 50% partial RoPE (rotary on the first rope_dim=32 of head_dim=64)
 *   - QK RMSNorm, gated attention outputs (sigmoid gate), bounded SwiGLU
 *     (activation clip 10), residual selectors every 4 layers
 *   - tied embeddings, vocab 16,384, context 2,048
 *   - Muon-optimizer training (the reference recipe: lr 1e-4, wd 0.1)
 *
 * Pure C11, opaque struct, no third-party deps. The forward pass runs on
 * CPU (hosted) and will run on metal via the same kernel-dispatch path.
 */
#ifndef WUBU_H
#define WUBU_H

#include <stdint.h>
#include <stddef.h>

/* The RUNTIME dims (the agnostic loader doctrine): the dimensional macros
 * are DATA — set by wubu_runtime_dims_probe/set at load time. wubu_runtime_dims.h
 * comes FIRST so its WUBU_RUNTIME_DIMS-backed macros win (WUBU_DIM =
 * WUBU_RUNTIME_DIMS.dim etc). The block below is the DEFAULT (wubu-35m seed)
 * geometry, used only when the runtime dims were not set yet. */
#include "wubu_runtime_dims.h"
#ifndef WUBU_RUNTIME_DIMS_H
#define WUBU_VOCAB       16384
#define WUBU_DIM         448
#define WUBU_LAYERS      12
#define WUBU_HEADS       7
#define WUBU_KV_HEADS    1
#define WUBU_HEAD_DIM    64
#define WUBU_ROPE_DIM    32
#define WUBU_FFN_DIM     1228
#define WUBU_MAX_SEQ     2048
#define WUBU_LOCAL_WIN   256
#define WUBU_FULL_EVERY  4
#define WUBU_SELECT_EVERY 4
#define WUBU_CLIP        10.0f
#define WUBU_EPS         1e-6f
#define WUBU_SELECTORS   3   /* 12 / 4 */
#endif
/* struct pointer-array cap (runtime L <= 64; the struct holds POINTERS) */
#ifndef WUBU_MAX_LAYERS
#define WUBU_MAX_LAYERS 64
#endif

/* the exact released parameter count */
#define WUBU_PARAMS      35072768

/* A single transformer block's weights (local OR full attention). */
typedef struct {
    /* attention */
    float *q_proj;   /* [dim, heads*head_dim] = [448, 448] */
    float *k_proj;   /* [dim, kv_heads*head_dim] = [448, 64] */
    float *v_proj;   /* [448, 64] */
    float *o_proj;   /* [448, 448] */
    float *g_proj;   /* [448, 448] (attention gate) */
    float *q_norm;   /* [64] */
    float *k_norm;   /* [64] */
    float *attn_norm;/* [448] */
    /* ffn (bounded SwiGLU) */
    float *gate_up;  /* [448, 2*1228] = [448, 2456] */
    float *down;     /* [1228, 448] */
    float *ffn_norm; /* [448] */
} wubu_block_t;

/* The full model. */
typedef struct {
    float *embedding;       /* [16384, 448] (tied with lm_head) */
    float *final_norm;      /* [448] */
    wubu_block_t blocks[WUBU_MAX_LAYERS];
    float *selectors[8];  /* [448] each (score weight) — runtime count */
    int    is_full[WUBU_MAX_LAYERS];       /* attention rhythm */
    int    fire_sel[WUBU_MAX_LAYERS];     /* the residual-selector rhythm
                                          (per-block like is_full: the
                                          growth operator shifts it) */
    int    n_layers;                   /* the ACTIVE layer count (the
                                          growth operator's contract; the
                                          released model = WUBU_LAYERS) */
    /* the WuBu mode (the blueprint phases 1-2): 0 = the released WuBu
     * path (exact parity); 1 = hyperbolic lift/rotation + mixed agents.
     * Set with wubu_set_wubu_mode(). */
    int    wubu_mode;
    /* the mixed-agents router weights (wubu_moe2), owned by the caller
     * when wubu_mode is on; one per block, shared structure reused. */
    void  *wubu_moe;
} wubu_model_t;

/* An inference buffer (all the working memory, allocated once). */
typedef struct {
    float *x;        /* [seq, 448] the hidden stream */
    float *x2;       /* [seq, 448] scratch */
    float *q;        /* [seq, heads*head_dim] */
    float *k;        /* [seq, kv*head_dim] */
    float *v;        /* [seq, kv*head_dim] */
    float *attn_out; /* [seq, 448] */
    float *gate;     /* [seq, 448] scratch: rmsnorm out / ffn input */
    float *g_out;    /* [seq, 448] the attention gate (g_proj output).
                        Kept separate: matmul must never write into its
                        own input (the in-place aliasing bug) */
    float *ffn_gate; /* [seq, ffn_dim] */
    float *ffn_up;   /* [seq, ffn_dim] */
    float *ffn_out;  /* [seq, 448] */
    float *logits;   /* [seq, vocab] */
    float *checkpoint; /* [max_seq, 448] the group-input checkpoint
                          (the released path uses x2; the wubu path
                          needs its own so the o_proj output survives) */
    float *cos_tbl;  /* [max_seq, rope_dim] */
    float *sin_tbl;  /* [max_seq, rope_dim] */
    float *cache_k;  /* [layers][max_seq, 64] */
    float *cache_v;  /* [layers][max_seq, 64] */
    size_t seq_alloc;
} wubu_buf_t;

/* B1: init the model from raw weight buffers (the safetensors loader
 * fills them; the model takes ownership of the pointers). */
int wubu_model_init(wubu_model_t *m, float *embedding, float *final_norm,
                    wubu_block_t *blocks, float **selectors);
/* The from-scratch random-init builder: allocates the FULL model at the
 * runtime WUBU_RUNTIME_DIMS geometry with zero pretrained weights (the amoeba
 * doctrine — 'delete the old model, make a new model'). */
int wubu_model_random_init(wubu_model_t *m);

/* B2: load the released checkpoint from a safetensors file. */
int wubu_load(wubu_model_t *m, const char *safetensors_path);

/* B3: allocate the inference buffer for a given max sequence. */
int wubu_buf_alloc(wubu_buf_t *b, size_t max_seq);

/* B4: the forward pass -- full sequence, causal. */
int wubu_forward(wubu_model_t *m, wubu_buf_t *b,
                  const uint16_t *tokens, size_t n_tokens);

/* B5: greedy + temperature generation. */
size_t wubu_generate(wubu_model_t *m, wubu_buf_t *b,
                      uint16_t *tokens, size_t n_prompt, size_t max_new,
                      float temperature, uint32_t seed);

/* B6: the logits for the last token (the caller samples). */
float *wubu_last_logits(wubu_buf_t *b);

/* B7: the cross-entropy loss + the Muon optimizer step (training). */
float wubu_loss(wubu_buf_t *b, const uint16_t *tokens, size_t n_tokens);
int  wubu_muon_step(wubu_model_t *m, float lr, float weight_decay);

/* B8: the WuBu mode (the blueprint). 0 = released-path parity (default);
 * 1 = hyperbolic lift/rotation + mixed-agents FFN. The mixed-agents
 * weights are a wubu_moe2_t* (caller-owned) or NULL (the router runs
 * on the block's own gate_up as the shared expert only). */
int wubu_set_wubu_mode(wubu_model_t *m, int mode, void *moe);

/* B9: the parameter count sanity check (must be 35,072,768). */
long wubu_parameter_count(const wubu_model_t *m);

/* B10: free everything. */
void wubu_free(wubu_model_t *m, wubu_buf_t *b);

#endif