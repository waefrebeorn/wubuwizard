/* wubu1_block.h — THE canonical block (WuBu1 base model design §2).
 *
 * One struct, end to end: the training struct IS the inference struct.
 * No parallel formats, no bridge hacks. The block carries ROLES — the
 * loader asks for WUBU_T_ATTN_Q and gets the tensor, whatever the source
 * format (GGUF import via wubu_gguf_names, native .st, safetensors).
 *
 * Differences from the old parallel structs (wubu.h wubu_block_t /
 * wubu_model.h wubu_layer_t + wubu_ssm.h ssm_layer_weights):
 *   - dense FFN is FIRST-CLASS (never "1-expert MoE")
 *   - geometry (curvature / level descriptor / spread) is native
 *   - hive tissue (slot / is_full / fire_sel) is native
 *   - KV metadata (curvature + precision per block) is native
 *   - config is DATA (wubu1_header_t), not #defines
 *
 * The bridge wubu1_block_fill() converts a loaded wubu_layer_t into this
 * canonical form (lossless for the shared weight pointers). The forward
 * pass migrates to consume wubu1_block_t directly in the next slice.
 *
 * C11, opaque-free (the block is the seam), minimal includes.
 */
#ifndef WUBU1_BLOCK_H
#define WUBU1_BLOCK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_layer_t wubu_layer_t;   /* forward decl — the bridge takes the loader's layer */

/* ---- The checkpoint header: config is DATA, not code ---- */
typedef struct wubu1_header {
    int      d_model;        /* hidden dim (448 for WuBu-35M) */
    int      n_layers;       /* active layer count */
    int      min_layers;     /* amoeba growth floor */
    int      max_layers;     /* amoeba growth ceiling */
    int      q_heads;        /* query heads */
    int      kv_heads;       /* KV heads (GQA 7:1 → 1) */
    int      head_dim;       /* per-head dim (64) */
    int      rope_dim;       /* partial RoPE dim (32 = 50%) */
    int      ffn_dim;        /* dense FFN hidden (1228) */
    int      vocab;          /* byte-level BPE vocab (16384) */
    int      max_ctx;        /* max cached positions */
    int      tied_embeddings;/* 1 = lm_head == embedding */
    int      selectors_every;/* residual-selector rhythm (4) */
    int      full_every;     /* local/full attention rhythm (4) */
    int      n_curvatures;   /* K in the product of Poincaré balls */
    float    curvatures[16]; /* learnable per-level curvatures c_i */
    uint32_t layout_flags;   /* fused_gate_up / split gate+up, ... */
    uint32_t magic;          /* WUBU1_HEADER_MAGIC */
    uint32_t version;        /* 1 */
} wubu1_header_t;

#define WUBU1_HEADER_MAGIC 0x31425557u  /* "WUB1" */
#define WUBU1_HEADER_VERSION 1u

/* Layout flags (bit 0). */
#define WUBU1_FLAG_FUSED_GATE_UP  (1u << 0)
#define WUBU1_FLAG_TIED_EMBED     (1u << 1)

/* ---- The canonical block ---- */
typedef struct wubu1_block {
    /* attention (GQA native; any ratio from header) */
    float *q_proj;      /* [D, q_heads*head_dim]            */
    float *k_proj;      /* [D, kv_heads*head_dim]           */
    float *v_proj;      /* [D, kv_heads*head_dim]           */
    float *o_proj;      /* [q_heads*head_dim, D]            */
    float *g_proj;      /* [D, q_heads*head_dim] gate       */
    float *q_norm;      /* [head_dim] per-head Q RMSNorm    */
    float *k_norm;      /* [head_dim] per-head K RMSNorm    */
    float *attn_norm;   /* [D] pre-attention                */
    float *post_attn_norm; /* [D] post-attention (optional) */

    /* dense FFN — first-class, never "1-expert MoE" */
    float *gate_up;     /* fused SwiGLU [D, 2*ffn] (or gate+up split) */
    float *up;          /* [D, ffn] when not fused           */
    float *gate;        /* [D, ffn] when not fused           */
    float *down;        /* [ffn, D]                          */
    float *ffn_norm;    /* [D]                               */

    /* quantized weight pointers (into GGUF blob, don't free) */
    const uint8_t *q_proj_q;  int q_proj_q_type;
    const uint8_t *k_proj_q;  int k_proj_q_type;
    const uint8_t *v_proj_q;  int v_proj_q_type;
    const uint8_t *o_proj_q;  int o_proj_q_type;
    const uint8_t *g_proj_q;  int g_proj_q_type;
    const uint8_t *gate_up_q; int gate_up_q_type;
    const uint8_t *down_q;    int down_q_type;

    /* geometry (nested spheres) */
    float  curvature;   /* this block's Poincaré ball curvature c_i */
    float *ld;          /* level descriptor [D]              */
    float  spread;      /* level spread σ_i                  */

    /* KV metadata (the KV cache IS a filesystem — AN16) */
    float  kv_curvature;  /* Poincaré curvature of this block's KV region */
    int    kv_precision;  /* per-role precision for KV writes (Escha) */

    /* hive tissue (the amoeba body — WB04/WB05) */
    int    slot;        /* hive slot id (stable pointer)    */
    int    is_full;     /* attention rhythm (local/full)    */
    int    fire_sel;    /* residual-selector rhythm         */

    /* per-layer dynamic dims (from tensor shapes via roles) */
    int    q_dim;       /* q_heads * head_dim               */
    int    kv_dim;      /* kv_heads * head_dim              */
    int    out_dim;     /* o_proj input dim                 */
} wubu1_block_t;

/* ---- Header helpers ---- */
static inline int wubu1_header_valid(const wubu1_header_t *h) {
    return h && h->magic == WUBU1_HEADER_MAGIC &&
           h->version == WUBU1_HEADER_VERSION &&
           h->d_model > 0 && h->n_layers > 0 && h->vocab > 0 &&
           h->head_dim > 0 && h->q_heads > 0 && h->kv_heads > 0 &&
           h->kv_heads <= h->q_heads;
}

/* Lossless bridge: loaded wubu_layer_t → canonical block (wubu1_block_bridge.c).
 * Returns 1 on success, 0 if out/layer is NULL. */
int wubu1_block_fill(wubu1_block_t *out, const wubu_layer_t *layer);

#ifdef __cplusplus
}
#endif

#endif /* WUBU1_BLOCK_H */
