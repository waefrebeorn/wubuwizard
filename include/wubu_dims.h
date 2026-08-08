/*
 * wubu_dims.h -- runtime model dimensions for wubuwizard.
 *
 * wubuwizard's SSM/GQA/MoE forward passes were originally hardcoded to a
 * single model via compile-time macros (D_MODEL=2048, CONV_DIM=8192,
 * ...). That is broken: it cannot load the real Colonel models
 * (Qwen3.6-27B hidden=5120, Agents-A1-4B hidden=2560, ...).
 *
 * This module makes those macros read from a single runtime global
 * WUBU_DIMS, which the loader sets from the ACTUAL tensor shapes of the
 * checkpoint being loaded. Every forward function then runs at the
 * model's real dimensions -- no edits inside the 2900-line forward
 * bodies, just a header redefinition. One source of truth, split out so
 * no other file is a "god header".
 *
 * C11, opaque-ish: only the struct + the macro aliases live here.
 */
#ifndef WUBU_DIMS_H
#define WUBU_DIMS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int d_model;       /* hidden size                       */
    int conv_dim;     /* SSM in_proj_qkv out dim (Q+K+V)   */
    int value_dim;    /* SSM in_proj_z out (VALUE_DIM)     */
    int key_dim;      /* SSM Q/K dim (K heads)             */
    int dt_rank;      /* ssm_time_step_rank                */
    int ssm_d_state;  /* SSM state dim (head_k = head_v)   */
    int ssm_k_heads;  /* SSM num key heads                 */
    int ssm_v_heads;  /* SSM num value heads               */
    int conv_kernel;  /* conv1d kernel size                */
    int gqa_q_heads;  /* attention query heads             */
    int gqa_kv_heads; /* attention kv heads (GQA)          */
    int gqa_head_dim; /* attention head dim                */
    int gqa_kv_dim;   /* kv_heads * head_dim               */
    /* MLA (Multi-Latent Attention, DeepSeek-V2/V4): latent compression */
    int q_lora_rank;   /* Q down-projection rank          */
    int kv_lora_rank;  /* KV down-projection rank         */
    int rope_head_dim; /* RoPE dimension per head         */
    int head_dim_full; /* head_dim + rope_head_dim (MLA K) */
} wubu_dims_t;

/* The single active dimension set. The loader calls wubu_dims_set() right
 * before loading/running a model. */
extern wubu_dims_t WUBU_DIMS;

/* Host-callable: push WUBU_DIMS into CUDA __constant__ memory. Defined in
 * wubu_dims_gpu.cu (always built, nvcc present). Called from wubu_dims_set
 * so device kernels always see the active dims. No-op safe if no GPU. */
void wubu_dims_sync_gpu(void);

/* CUDA device code cannot read the host 'extern WUBU_DIMS' global. For
 * device kernels we mirror it into a __constant__ symbol and redefine the
 * macro so the 330+ existing device references compile unchanged. */
#ifdef __CUDACC__
extern __constant__ wubu_dims_t WUBU_DIMS_DEV;
#define WUBU_DIMS WUBU_DIMS_DEV
#endif

/* Set dims explicitly (loader computes them from real tensor shapes). */
void wubu_dims_set(const wubu_dims_t *d);

/* Recompute derived fields (conv_dim, key_dim, gqa_kv_dim) from the
 * primary fields already present in *d. Safe to call after a partial set. */
void wubu_dims_finalize(wubu_dims_t *d);

/* Defaults matching wubuwizard's original 2048-hidden build (GGUF path
 * stays byte-compatible). Call before any legacy GGUF load. */
void wubu_dims_default(void);

#ifdef __cplusplus
}
#endif

/* ---- Macro aliases ----
 * Principle: only dims that ACTUALLY VARY between the supported Colonel
 * models are routed through the runtime global WUBU_DIMS. Dims that are
 * invariant across every target model (Qwen3.6-27B, Agents-A1-4B,
 * KAT-Coder, BTL-3) stay as compile-time constants so they remain legal
 * in CUDA __shared__ arrays and template parameters.
 *
 * Invariant (verified from the 4 real config.json files):
 *   SSM_D_STATE = 128   SSM_K_HEADS = 16   DT_RANK = 32   CONV_KERNEL = 4
 *   KEY_DIM = SSM_D_STATE*SSM_K_HEADS = 2048   (same for all 4)
 *
 * Varying (routed through WUBU_DIMS):
 *   D_MODEL, VALUE_DIM, SSM_V_HEADS, CONV_DIM(=2*KEY_DIM+VALUE_DIM),
 *   GQA_Q_HEADS, GQA_KV_HEADS, GQA_HEAD_DIM, GQA_KV_DIM.
 */
#define SSM_D_STATE   128
#define SSM_K_HEADS   16
#define DT_RANK       WUBU_DIMS.dt_rank   /* runtime: Qwen3.5=16, Qwen3.6=32 */
#define CONV_KERNEL   4
#define KEY_DIM       (SSM_D_STATE * SSM_K_HEADS)   /* 2048, invariant */

#define D_MODEL       WUBU_DIMS.d_model
#define VALUE_DIM     WUBU_DIMS.value_dim
#define SSM_V_HEADS   WUBU_DIMS.ssm_v_heads
#define CONV_DIM      (KEY_DIM * 2 + VALUE_DIM)     /* varies via VALUE_DIM */
#define GQA_Q_HEADS   WUBU_DIMS.gqa_q_heads
#define GQA_KV_HEADS  WUBU_DIMS.gqa_kv_heads
#define GQA_HEAD_DIM  WUBU_DIMS.gqa_head_dim
#define GQA_KV_DIM    WUBU_DIMS.gqa_kv_dim
/* MLA (Multi-Latent Attention) dims */
#define Q_LORA_RANK   WUBU_DIMS.q_lora_rank
#define KV_LORA_RANK  WUBU_DIMS.kv_lora_rank
#define ROPE_HEAD_DIM WUBU_DIMS.rope_head_dim
#define HEAD_DIM_FULL (WUBU_DIMS.head_dim_full)

#endif /* WUBU_DIMS_H */
