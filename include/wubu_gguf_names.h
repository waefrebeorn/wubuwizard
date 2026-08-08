/* wubu_gguf_names.h — role-based GGUF tensor-name resolver.
 *
 * The problem: wubu_model.c historically hardcoded Qwen-style tensor names
 * ("blk.%d.attn_q.weight") in 160+ places, which silently broke every other
 * GGUF naming convention (Gemma "model.layers.N.*", HF-style "layers.N.*"
 * with attn.q_proj / ffn.gate_up fused layouts, full HF
 * "model.language_model.layers.N.*").
 *
 * The fix — better than llama.cpp's per-architecture tensor maps: resolve
 * EVERY weight by ROLE. Each role carries the full candidate list across all
 * known conventions; the resolver scans the file's actual tensor names and
 * returns the first hit. No architecture metadata required (our own
 * WuBu-35M GGUF has none), no hardcoded per-arch tables, works on any GGUF.
 *
 * C11, opaque-free (operates on gguf_ctx directly), minimal includes.
 */

#ifndef WUBU_GGUF_NAMES_H
#define WUBU_GGUF_NAMES_H

#include "gguf_reader.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Every weight the engine can load, expressed as a role. */
typedef enum {
    WUBU_T_ATTN_NORM,          /* pre-attention RMSNorm (attn_norm.weight /
                                  input_layernorm.weight) */
    WUBU_T_POST_ATTN_NORM,     /* post-attention norm (post_attention_norm.weight /
                                  post_attention_layernorm.weight / ffn_norm.weight) */
    WUBU_T_FFN_NORM,           /* pre-FFN norm (ffn_norm.weight /
                                  post_attention_layernorm.weight) */
    WUBU_T_ATTN_Q,             /* Q projection (attn_q.weight / attn.q_proj.weight /
                                  self_attn.q_proj.weight) */
    WUBU_T_ATTN_K,             /* K projection */
    WUBU_T_ATTN_V,             /* V projection */
    WUBU_T_ATTN_O,             /* O projection (attn_output.weight / attn.o_proj.weight) */
    WUBU_T_ATTN_QKV,           /* fused Q+K+V (attn_qkv.weight / attn.qkv_proj.weight) */
    WUBU_T_ATTN_GATE,          /* gated-attention gate (attn_gate.weight /
                                  attn.g_proj.weight / self_attn.g_proj.weight) */
    WUBU_T_ATTN_Q_NORM,        /* per-head Q RMSNorm (attn_q_norm.weight /
                                  attn.q_norm.weight) */
    WUBU_T_ATTN_K_NORM,        /* per-head K RMSNorm */
    WUBU_T_SSM_QKV,            /* SSM fused QKV (attn_qkv.weight) */
    WUBU_T_SSM_GATE,           /* SSM gate (attn_gate.weight) */
    WUBU_T_SSM_BETA,           /* SSM beta */
    WUBU_T_SSM_ALPHA,          /* SSM alpha */
    WUBU_T_SSM_DT,             /* SSM dt bias (ssm_dt.bias) */
    WUBU_T_SSM_A,              /* SSM a matrix (ssm_a / ssm_a.weight) */
    WUBU_T_SSM_CONV1D,         /* SSM conv1d */
    WUBU_T_SSM_NORM,           /* SSM norm */
    WUBU_T_SSM_OUT,            /* SSM output projection (ssm_out.weight) */
    WUBU_T_FFN_GATE,           /* dense FFN gate (ffn_gate.weight / mlp.gate_proj.weight) */
    WUBU_T_FFN_UP,             /* dense FFN up (ffn_up.weight / mlp.up_proj.weight) */
    WUBU_T_FFN_DOWN,           /* dense FFN down (ffn_down.weight / mlp.down_proj.weight) */
    WUBU_T_FFN_GATE_UP,        /* FUSED gate+up (ffn_gate_up.weight / ffn.gate_up.weight /
                                  mlp.gate_up_proj.weight) — SwiGLU fused */
    WUBU_T_MOE_GATE_INP,       /* MoE router (ffn_gate_inp.weight / ffn_gate.weight) */
    WUBU_T_MOE_GATE_SHEXP,     /* MoE shared-expert gate */
    WUBU_T_MOE_GATE_EXPS,      /* MoE routed gate weights */
    WUBU_T_MOE_UP_EXPS,        /* MoE routed up weights */
    WUBU_T_MOE_DOWN_EXPS,      /* MoE routed down weights */
    WUBU_T_MOE_UP_SHEXP,       /* MoE shared-expert up */
    WUBU_T_MOE_DOWN_SHEXP,     /* MoE shared-expert down */
    WUBU_T_CONV_IN,            /* LFM shortconv in_proj (shortconv.in_proj.weight) */
    WUBU_T_CONV_W,             /* LFM shortconv depthwise causal conv (shortconv.conv.weight) */
    WUBU_T_CONV_OUT,           /* LFM shortconv out_proj (shortconv.out_proj.weight) */
    WUBU_T_TOKEN_EMBD,         /* token embeddings (token_embd.weight / embedding.weight) */
    WUBU_T_OUTPUT,             /* logit head (output.weight / lm_head.weight) */
    WUBU_T_OUTPUT_NORM,        /* final norm (output_norm.weight / final_norm.weight /
                                  model.norm.weight) */
    WUBU_T_COUNT
} wubu_gguf_role_t;

/* Naming conventions detected from the actual tensor names. */
typedef enum {
    WUBU_CONV_UNKNOWN = 0,
    WUBU_CONV_QWEN,            /* blk.N.* (llama.cpp / Qwen-family GGUFs) */
    WUBU_CONV_GEMMA,           /* model.layers.N.* (Gemma-family) */
    WUBU_CONV_HF_BARE,         /* layers.N.* with attn.q_proj / ffn.gate_up (HF-style
                                  hierarchical, WuBu-35M GGUF) */
    WUBU_CONV_HF_FULL          /* model.language_model.layers.N.* (full HF path) */
} wubu_gguf_convention_t;

/* Result of scanning a GGUF's tensor table. */
typedef struct {
    wubu_gguf_convention_t convention;
    int max_layer;             /* highest layer index seen (any prefix) */
    int n_layers;              /* max_layer + 1 (0 if no layer tensors) */
    int has_ssm;               /* any ssm_* / linear_attn tensor present */
    int has_moe;               /* any ffn_*_exps / expert tensor present */
    int has_dense_ffn;         /* ffn_gate/up/down or gate_up present */
    int has_gqa;               /* attn_q / q_proj present */
} wubu_gguf_names_t;

/* Scan ctx->tensors, detect convention + layer count + feature flags. */
void wubu_gguf_names_detect(gguf_ctx *ctx, wubu_gguf_names_t *out);

/* Resolve a layer weight by role. Returns NULL if no candidate matches.
 * Pass layer < 0 for non-layer tensors (embeddings, output, final norm). */
gguf_tensor_info *wubu_gguf_find(gguf_ctx *ctx, int layer,
                                 wubu_gguf_role_t role);

/* Non-layer roles convenience (layer arg ignored). */
static inline gguf_tensor_info *wubu_gguf_find_global(gguf_ctx *ctx,
                                                      wubu_gguf_role_t role) {
    return wubu_gguf_find(ctx, -1, role);
}

#ifdef __cplusplus
}
#endif

#endif /* WUBU_GGUF_NAMES_H */
