#ifndef WUBU_LFM2_H
#define WUBU_LFM2_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * LFM2.5 (Liquid AI) hybrid loader + forward -- self-contained,
 * C11, opaque structs. Reuses wubuwizard's quantized_matmul /
 * rmsnorm helpers but owns its own gated-conv + GQA dispatch.
 *
 * Conv block (Liquid LFM2 technical report, arxiv 2511.23404):
 *   (B, C, h_tilde) = in_proj(h)        // [3*conv_dim, d_model]
 *   y = B * h_tilde                    // input gate
 *   z = depthwise_causal_conv_k(y)     // kernel k (3 for LFM2.5)
 *   o = out_proj(C * z)                // output gate + linear
 * No recurrent state -- pure conv.
 * ============================================================ */

typedef struct {
    /* Conv (SSM) block -- all F32 (dequantized from BF16) */
    float *in_proj;     /* [3*conv_dim, d_model] */
    float *conv_w;      /* [conv_dim, k] depthwise causal conv weights */
    float *out_proj;    /* [d_model, conv_dim] */
    int    conv_k;      /* conv kernel (3) */

    /* GQA attention block */
    float *q_proj;      /* [d_model, d_model] (n_q*hd) */
    float *k_proj;      /* [kv_dim, d_model]   (n_kv*hd) */
    float *v_proj;      /* [kv_dim, d_model] */
    float *o_proj;      /* [d_model, d_model] */
    float *q_ln;        /* [hd] layernorm gamma */
    float *k_ln;        /* [hd] layernorm gamma */

    /* SwiGLU FFN */
    float *w1;          /* [ff, d_model] gate */
    float *w2;          /* [d_model, ff] down */
    float *w3;          /* [ff, d_model] up */

    /* Norms */
    float *ffn_norm;    /* [d_model] */
    float *op_norm;     /* [d_model] operator (attn/conv) norm */

    /* LAZY quantized sources (GGUF blob, zero-copy). When set, the F32
     * fields above are NULL until lfm2_layer_materialize() dequantizes
     * THIS layer only; lfm2_layer_release() frees after the forward
     * pass. Keeps peak RAM ~= one layer, not the whole model. */
    const uint8_t *q_in_proj;  int q_in_proj_t;  int64_t q_in_proj_ne;
    const uint8_t *q_conv_w;   int q_conv_w_t;   int64_t q_conv_w_ne;
    const uint8_t *q_out_proj; int q_out_proj_t; int64_t q_out_proj_ne;
    const uint8_t *q_q_proj;   int q_q_proj_t;   int64_t q_q_proj_ne;
    const uint8_t *q_k_proj;   int q_k_proj_t;   int64_t q_k_proj_ne;
    const uint8_t *q_v_proj;   int q_v_proj_t;   int64_t q_v_proj_ne;
    const uint8_t *q_o_proj;   int q_o_proj_t;   int64_t q_o_proj_ne;
    const uint8_t *q_w1;       int q_w1_t;       int64_t q_w1_ne;
    const uint8_t *q_w2;       int q_w2_t;       int64_t q_w2_ne;
    const uint8_t *q_w3;       int q_w3_t;       int64_t q_w3_ne;
} lfm2_layer_t;

typedef struct {
    int n_layers;
    int d_model;
    int conv_dim;
    int n_q_heads;
    int n_kv_heads;
    int head_dim;
    int ff_dim;
    int vocab_size;
    float rope_theta;
    float rms_eps;      /* layer_norm_rms_epsilon (MiniCPM5: 1e-6, LFM2.5: 1e-5) */
    int conv_k;         /* shortconv kernel (l_cache), default 3 */
    bool *is_conv;      /* per-layer: true=conv block, false=GQA */
    int is_conv_from_kv;    /* is_conv came from head_count_kv KV array */
    bool is_conv_kv[128];   /* per-layer conv flag from KV (authoritative) */
    lfm2_layer_t *layers;
    float *embed;       /* [vocab, d_model] (tied with lm_head) */
    float *embed_norm;  /* [d_model] applied to hidden ONCE after all layers (HF Lfm2Model) */
    /* quantized embed source (GGUF blob) — embed stays NULL until a row
     * is dequantized on demand (1GB F32 vs ~350MB quantized) */
    const uint8_t *q_embed;
    int q_embed_type;
    int embed_bytes_per_row;
    /* lm_head: many GGUFs have a SEPARATE untied output.weight (MiniCPM5).
     * When present it wins over the tied embed (LFM2.5 uses the tied path). */
    const uint8_t *q_lm_head;
    int q_lm_head_type;
    int lm_head_bytes_per_row;
    /* KV cache for attention layers: [n_layers][2][n_kv_heads*head_dim*maxT] */
    float *kv_cache;
    int    kv_max_t;
} lfm2_model_t;

/* Load a LFM2.5 safetensors checkpoint directory into an lfm2_model_t.
 * Returns true on success. Self-contained: maps model.layers.N.* names. */
bool lfm2_load(const char *model_dir, lfm2_model_t *m);

/* Free all owned buffers. */
void lfm2_free(lfm2_model_t *m);

/* Forward one sequence of token embeddings. emb[B*T*d_model] in,
 * logits[vocab] out (last token). Allocates scratch internally. */
bool lfm2_forward(const lfm2_model_t *m, const float *emb, int B, int T,
                  float *logits);

/* Materialize ONE layer's quantized weights to F32 (fills the float*
 * fields; no-op if already materialized). Returns true on success.
 * Call before the layer's forward, lfm2_layer_release() after. */
bool lfm2_layer_materialize(lfm2_layer_t *L, int d_model, int conv_dim,
                            int ff_dim);

/* Free the F32 fields of one layer (keeps the quantized blob pointers). */
void lfm2_layer_release(lfm2_layer_t *L);

/* Quantized tied lm_head: dequantize each vocab row on the fly. */
void lfm2_lmhead_q_from(const float *h, const lfm2_model_t *m, float *logits,
                        const uint8_t *src, int src_type, int bytes_per_row);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_LFM2_H */
