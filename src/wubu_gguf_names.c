/* wubu_gguf_names.c — role-based GGUF tensor-name resolver.
 *
 * Every weight is a ROLE; every role has a candidate list covering all
 * naming conventions seen in the wild. Detection scans the file's actual
 * tensor names (our WuBu-35M GGUF has no general.architecture metadata,
 * so we never trust metadata). Layer counting accepts every known prefix.
 *
 * C11, no third-party, single pass over the tensor table at detect.
 */

#include "wubu_gguf_names.h"

#include <stdio.h>
#include <string.h>

/* ---------------------------------------------------------------- */
/* Candidate templates per role.                                     */
/*                                                                    */
/* Each entry: { prefix template, suffix }. prefix is snprintf'd with  */
/* the layer number; "blk.%d." / "model.layers.%d." / "layers.%d." /  */
/* "model.language_model.layers.%d.". Global roles use prefix "".     */
/* ---------------------------------------------------------------- */

typedef struct {
    const char *prefix;   /* e.g. "blk.%d." — snprintf with layer; "" = global */
    const char *suffix;   /* e.g. "attn_q.weight" */
} wubu_gguf_tmpl_t;

#define QWEN    "blk.%d."
#define GEMMA   "model.layers.%d."
#define HFBARE  "layers.%d."
#define HFFULL  "model.language_model.layers.%d."

/* Role -> candidate list. Ordered: most common first. */
static const wubu_gguf_tmpl_t g_role_templates[WUBU_T_COUNT][12] = {
    /* WUBU_T_ATTN_NORM */
    { { QWEN,   "attn_norm.weight" },
      { GEMMA,  "attn_norm.weight" },
      { HFBARE, "attn_norm.weight" },
      { HFBARE, "input_layernorm.weight" },
      { HFFULL, "input_layernorm.weight" },
      { HFFULL, "attn_norm.weight" },
      { "", "" } },
    /* WUBU_T_POST_ATTN_NORM */
    { { QWEN,   "post_attention_norm.weight" },
      { GEMMA,  "post_attention_norm.weight" },
      { HFBARE, "post_attention_layernorm.weight" },
      { HFBARE, "post_attention_norm.weight" },
      { HFFULL, "post_attention_layernorm.weight" },
      { QWEN,   "ffn_norm.weight" },
      { HFBARE, "ffn_norm.weight" },
      { "", "" } },
    /* WUBU_T_FFN_NORM */
    { { QWEN,   "ffn_norm.weight" },
      { GEMMA,  "ffn_norm.weight" },
      { HFBARE, "ffn_norm.weight" },
      { HFBARE, "post_attention_layernorm.weight" },
      { HFFULL, "post_attention_layernorm.weight" },
      { "", "" } },
    /* WUBU_T_ATTN_Q */
    { { QWEN,   "attn_q.weight" },
      { GEMMA,  "attn_q.weight" },
      { HFBARE, "attn_q.weight" },
      { HFBARE, "attn.q_proj.weight" },
      { HFFULL, "self_attn.q_proj.weight" },
      { HFBARE, "self_attn.q_proj.weight" },
      { "", "" } },
    /* WUBU_T_ATTN_K */
    { { QWEN,   "attn_k.weight" },
      { GEMMA,  "attn_k.weight" },
      { HFBARE, "attn_k.weight" },
      { HFBARE, "attn.k_proj.weight" },
      { HFFULL, "self_attn.k_proj.weight" },
      { HFBARE, "self_attn.k_proj.weight" },
      { "", "" } },
    /* WUBU_T_ATTN_V */
    { { QWEN,   "attn_v.weight" },
      { GEMMA,  "attn_v.weight" },
      { HFBARE, "attn_v.weight" },
      { HFBARE, "attn.v_proj.weight" },
      { HFFULL, "self_attn.v_proj.weight" },
      { HFBARE, "self_attn.v_proj.weight" },
      { "", "" } },
    /* WUBU_T_ATTN_O */
    { { QWEN,   "attn_output.weight" },
      { GEMMA,  "attn_output.weight" },
      { HFBARE, "attn_output.weight" },
      { HFBARE, "attn.o_proj.weight" },
      { HFFULL, "self_attn.o_proj.weight" },
      { HFBARE, "self_attn.o_proj.weight" },
      { "", "" } },
    /* WUBU_T_ATTN_QKV (fused) */
    { { QWEN,   "attn_qkv.weight" },
      { GEMMA,  "attn_qkv.weight" },
      { HFBARE, "attn_qkv.weight" },
      { HFBARE, "attn.qkv_proj.weight" },
      { HFFULL, "self_attn.qkv_proj.weight" },
      { "", "" } },
    /* WUBU_T_ATTN_GATE */
    { { QWEN,   "attn_gate.weight" },
      { GEMMA,  "attn_gate.weight" },
      { HFBARE, "attn_gate.weight" },
      { HFBARE, "attn.g_proj.weight" },
      { HFFULL, "self_attn.g_proj.weight" },
      { HFBARE, "self_attn.g_proj.weight" },
      { "", "" } },
    /* WUBU_T_ATTN_Q_NORM */
    { { QWEN,   "attn_q_norm.weight" },
      { GEMMA,  "attn_q_norm.weight" },
      { HFBARE, "attn_q_norm.weight" },
      { HFBARE, "attn.q_norm.weight" },
      { HFFULL, "self_attn.q_norm.weight" },
      { "", "" } },
    /* WUBU_T_ATTN_K_NORM */
    { { QWEN,   "attn_k_norm.weight" },
      { GEMMA,  "attn_k_norm.weight" },
      { HFBARE, "attn_k_norm.weight" },
      { HFBARE, "attn.k_norm.weight" },
      { HFFULL, "self_attn.k_norm.weight" },
      { "", "" } },
    /* WUBU_T_SSM_QKV */
    { { QWEN,   "attn_qkv.weight" },
      { HFBARE, "attn_qkv.weight" },
      { HFBARE, "linear_attn.in_proj_qkv.weight" },
      { HFFULL, "linear_attn.in_proj_qkv.weight" },
      { "", "" } },
    /* WUBU_T_SSM_GATE */
    { { QWEN,   "attn_gate.weight" },
      { HFBARE, "attn_gate.weight" },
      { HFBARE, "linear_attn.in_proj_z.weight" },
      { HFFULL, "linear_attn.in_proj_z.weight" },
      { "", "" } },
    /* WUBU_T_SSM_BETA */
    { { QWEN,   "ssm_beta.weight" },
      { GEMMA,  "ssm_beta.weight" },
      { HFBARE, "ssm_beta.weight" },
      { HFBARE, "linear_attn.beta.weight" },
      { HFFULL, "linear_attn.beta.weight" },
      { "", "" } },
    /* WUBU_T_SSM_ALPHA */
    { { QWEN,   "ssm_alpha.weight" },
      { GEMMA,  "ssm_alpha.weight" },
      { HFBARE, "ssm_alpha.weight" },
      { HFBARE, "linear_attn.alpha.weight" },
      { HFFULL, "linear_attn.alpha.weight" },
      { "", "" } },
    /* WUBU_T_SSM_DT */
    { { QWEN,   "ssm_dt.bias" },
      { GEMMA,  "ssm_dt.bias" },
      { HFBARE, "ssm_dt.bias" },
      { HFBARE, "linear_attn.dt.bias" },
      { HFFULL, "linear_attn.dt.bias" },
      { "", "" } },
    /* WUBU_T_SSM_A */
    { { QWEN,   "ssm_a" },
      { GEMMA,  "ssm_a" },
      { HFBARE, "ssm_a" },
      { QWEN,   "ssm_a.weight" },
      { HFBARE, "ssm_a.weight" },
      { HFBARE, "linear_attn.a.weight" },
      { HFFULL, "linear_attn.a.weight" },
      { "", "" } },
    /* WUBU_T_SSM_CONV1D */
    { { QWEN,   "ssm_conv1d.weight" },
      { GEMMA,  "ssm_conv1d.weight" },
      { HFBARE, "ssm_conv1d.weight" },
      { HFBARE, "linear_attn.conv1d.weight" },
      { HFFULL, "linear_attn.conv1d.weight" },
      { "", "" } },
    /* WUBU_T_SSM_NORM */
    { { QWEN,   "ssm_norm.weight" },
      { GEMMA,  "ssm_norm.weight" },
      { HFBARE, "ssm_norm.weight" },
      { HFBARE, "linear_attn.norm.weight" },
      { HFFULL, "linear_attn.norm.weight" },
      { "", "" } },
    /* WUBU_T_SSM_OUT */
    { { QWEN,   "ssm_out.weight" },
      { GEMMA,  "ssm_out.weight" },
      { HFBARE, "ssm_out.weight" },
      { HFBARE, "linear_attn.out_proj.weight" },
      { HFFULL, "linear_attn.out_proj.weight" },
      { "", "" } },
    /* WUBU_T_FFN_GATE */
    { { QWEN,   "ffn_gate.weight" },
      { GEMMA,  "ffn_gate.weight" },
      { HFBARE, "ffn.gate_proj.weight" },
      { HFFULL, "mlp.gate_proj.weight" },
      { HFBARE, "mlp.gate_proj.weight" },
      { "", "" } },
    /* WUBU_T_FFN_UP */
    { { QWEN,   "ffn_up.weight" },
      { GEMMA,  "ffn_up.weight" },
      { HFBARE, "ffn.up_proj.weight" },
      { HFFULL, "mlp.up_proj.weight" },
      { HFBARE, "mlp.up_proj.weight" },
      { "", "" } },
    /* WUBU_T_FFN_DOWN */
    { { QWEN,   "ffn_down.weight" },
      { GEMMA,  "ffn_down.weight" },
      { HFBARE, "ffn.down.weight" },
      { HFFULL, "mlp.down_proj.weight" },
      { HFBARE, "mlp.down_proj.weight" },
      { "", "" } },
    /* WUBU_T_FFN_GATE_UP (fused SwiGLU) */
    { { HFBARE, "ffn.gate_up.weight" },
      { HFFULL, "mlp.gate_up_proj.weight" },
      { HFBARE, "mlp.gate_up_proj.weight" },
      { QWEN,   "ffn_gate_up.weight" },
      { "", "" } },
    /* WUBU_T_MOE_GATE_INP */
    { { QWEN,   "ffn_gate_inp.weight" },
      { HFBARE, "ffn.gate.weight" },
      { HFFULL, "mlp.gate.weight" },
      { HFBARE, "mlp.gate.weight" },
      { QWEN,   "ffn_gate.weight" },
      { "", "" } },
    /* WUBU_T_MOE_GATE_SHEXP */
    { { QWEN,   "ffn_gate_inp_shexp.weight" },
      { QWEN,   "ffn_gate_shexp.weight" },
      { HFBARE, "ffn.gate_shexp.weight" },
      { "", "" } },
    /* WUBU_T_MOE_GATE_EXPS */
    { { QWEN,   "ffn_gate_exps.weight" },
      { HFBARE, "ffn.gate_exps.weight" },
      { HFFULL, "mlp.experts.gate_proj.weight" },
      { "", "" } },
    /* WUBU_T_MOE_UP_EXPS */
    { { QWEN,   "ffn_up_exps.weight" },
      { HFBARE, "ffn.up_exps.weight" },
      { HFFULL, "mlp.experts.up_proj.weight" },
      { "", "" } },
    /* WUBU_T_MOE_DOWN_EXPS */
    { { QWEN,   "ffn_down_exps.weight" },
      { HFBARE, "ffn.down_exps.weight" },
      { HFFULL, "mlp.experts.down_proj.weight" },
      { "", "" } },
    /* WUBU_T_MOE_UP_SHEXP */
    { { QWEN,   "ffn_up_shexp.weight" },
      { HFBARE, "ffn.up_shexp.weight" },
      { "", "" } },
    /* WUBU_T_MOE_DOWN_SHEXP */
    { { QWEN,   "ffn_down_shexp.weight" },
      { HFBARE, "ffn.down_shexp.weight" },
      { "", "" } },
    /* WUBU_T_CONV_IN (LFM shortconv in_proj) */
    { { QWEN,   "shortconv.in_proj.weight" },
      { HFBARE, "shortconv.in_proj.weight" },
      { HFFULL, "conv.in_proj.weight" },
      { "", "" } },
    /* WUBU_T_CONV_W (LFM shortconv depthwise causal conv) */
    { { QWEN,   "shortconv.conv.weight" },
      { HFBARE, "shortconv.conv.weight" },
      { HFFULL, "conv.conv.weight" },
      { "", "" } },
    /* WUBU_T_CONV_OUT (LFM shortconv out_proj) */
    { { QWEN,   "shortconv.out_proj.weight" },
      { HFBARE, "shortconv.out_proj.weight" },
      { HFFULL, "conv.out_proj.weight" },
      { "", "" } },
    /* WUBU_T_TOKEN_EMBD */
    { { "", "token_embd.weight" },
      { "", "embedding.weight" },
      { "", "model.embed_tokens.weight" },
      { "", "model.language_model.embed_tokens.weight" },
      { "", "" } },
    /* WUBU_T_OUTPUT */
    { { "", "output.weight" },
      { "", "lm_head.weight" },
      { "", "model.language_model.lm_head.weight" },
      { "", "" } },
    /* WUBU_T_OUTPUT_NORM */
    { { "", "output_norm.weight" },
      { "", "final_norm.weight" },
      { "", "model.norm.weight" },
      { "", "model.language_model.final_norm.weight" },
      { "", "token_embd_norm.weight" },   /* LFM2.5: embed_norm applied after all layers */
      { "", "" } },
};

/* ---------------------------------------------------------------- */
/* Detection                                                          */
/* ---------------------------------------------------------------- */

static int tensor_matches_prefix(const char *name, const char *prefix_tmpl,
                                 int *layer_out) {
    /* Compare the fixed prefix part (before "%d.") and parse the layer. */
    const char *pct = strstr(prefix_tmpl, "%d.");
    if (!pct) return 0;
    size_t pre_len = (size_t)(pct - prefix_tmpl);
    if (strncmp(name, prefix_tmpl, pre_len) != 0) return 0;
    int layer = atoi(name + pre_len);
    if (layer < 0) return 0;
    if (layer_out) *layer_out = layer;
    return 1;
}

static int looks_like_layer_name(const char *name) {
    static const char *const prefixes[] = {
        "blk.", "model.layers.", "layers.", "model.language_model.layers."
    };
    for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); i++) {
        size_t n = strlen(prefixes[i]);
        if (strncmp(name, prefixes[i], n) == 0) {
            /* must be followed by digits */
            if (name[n] >= '0' && name[n] <= '9') return 1;
        }
    }
    return 0;
}

static int has_any_layer_tensor(gguf_ctx *ctx, const char *prefix, int *max_layer) {
    int found = 0, ml = -1;
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        int layer = -1;
        if (tensor_matches_prefix(ctx->tensors[i].name, prefix, &layer)) {
            found = 1;
            if (layer > ml) ml = layer;
        }
    }
    if (max_layer) *max_layer = ml;
    return found;
}

void wubu_gguf_names_detect(gguf_ctx *ctx, wubu_gguf_names_t *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!ctx) return;

    int ml = -1;
    /* Prefer the most specific convention that actually matches. */
    if (has_any_layer_tensor(ctx, HFFULL, &ml)) {
        out->convention = WUBU_CONV_HF_FULL;
    } else if (has_any_layer_tensor(ctx, HFBARE, &ml)) {
        out->convention = WUBU_CONV_HF_BARE;
    } else if (has_any_layer_tensor(ctx, GEMMA, &ml)) {
        out->convention = WUBU_CONV_GEMMA;
    } else if (has_any_layer_tensor(ctx, QWEN, &ml)) {
        out->convention = WUBU_CONV_QWEN;
    }
    if (ml >= 0) {
        out->max_layer = ml;
        out->n_layers = ml + 1;
    }

    /* Feature flags from any tensor name (any convention). */
    for (int64_t i = 0; i < ctx->n_tensors && ctx->tensors; i++) {
        const char *n = ctx->tensors[i].name;
        if (!n) continue;
        if (strstr(n, "ssm_") || strstr(n, "linear_attn")) out->has_ssm = 1;
        if (strstr(n, "_exps") || strstr(n, "experts.") ||
            strstr(n, "ffn_gate_inp") || strstr(n, "mlp.gate.weight")) out->has_moe = 1;
        if (strstr(n, "ffn_gate") || strstr(n, "ffn_up") ||
            strstr(n, "ffn_down") || strstr(n, "gate_up") ||
            strstr(n, "gate_proj") || strstr(n, "up_proj") ||
            strstr(n, "down_proj")) out->has_dense_ffn = 1;
        if (strstr(n, "attn_q") || strstr(n, "q_proj") ||
            strstr(n, "attn_qkv")) out->has_gqa = 1;
    }
}

/* ---------------------------------------------------------------- */
/* Role resolution                                                    */
/* ---------------------------------------------------------------- */

gguf_tensor_info *wubu_gguf_find(gguf_ctx *ctx, int layer,
                                 wubu_gguf_role_t role) {
    if (!ctx || role < 0 || role >= WUBU_T_COUNT) return NULL;
    char full[320];

    for (int i = 0; i < 12; i++) {
        const wubu_gguf_tmpl_t *t = &g_role_templates[role][i];
        if (!t->suffix[0]) break; /* end of list */
        if (t->prefix[0]) {
            if (layer < 0) continue; /* layer roles need a layer */
            /* prefix template contains %d — substitute the layer */
            char pref[128];
            snprintf(pref, sizeof(pref), t->prefix, layer);
            snprintf(full, sizeof(full), "%s%s", pref, t->suffix);
        } else {
            snprintf(full, sizeof(full), "%s", t->suffix);
        }
        gguf_tensor_info *hit = gguf_find_tensor(ctx, full);
        if (hit) return hit;
    }
    return NULL;
}
