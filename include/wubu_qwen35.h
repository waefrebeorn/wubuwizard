/*
 * wubu_qwen35.h — the QWEN3.5 HYBRID ADAPTER's LOADER ROLE SPLIT
 * (AN28 "build next" — RESOLVED 2026-08-09, CORRECTED by the real
 * GGUF the same day). C11.
 *
 * The fused-tensor role math for the Qwen3.5 hybrid family, driven by
 * the config AND the GGUF tensor inventory (the GGUF is the source of
 * truth — the config-only inference attributed the fused 6144 to the
 * gated-attn, but the on-disk tensors show it belongs to the GDN):
 *
 *   - the GDN layers (layer_types[i] == "linear_attention", 18 of 24):
 *       attn_qkv  [d, 6144] = qk (2*key_dim) + v (value_dim)
 *                             (key_dim = qk_heads * key_hd = 2048;
 *                              value_dim = v_heads * value_hd = 2048)
 *       attn_gate [d, 2048] = the Z gate to value space (2048 =
 *                             value_dim) — the config's in_proj_qkvz
 *                             total (8192 = 6144 + 2048) is split
 *                             across TWO tensors in the GGUF
 *       ssm_conv1d [4, 6144]  = the causal conv over the qk+v width
 *       ssm_out [2048, 1024]  = value_dim -> d (the delta-rule output)
 *       ssm_a [16], ssm_alpha [d,16], ssm_beta [d,16], ssm_dt.bias [16],
 *       ssm_norm [128] (per-value-head RMSNorm)
 *       NO q/k norms (the GDN has no per-head QK RMSNorm)
 *
 *   - the gated-attn layers ("full_attention", 6 of 24 — indices
 *       3,7,11,15,19,23) are UNFUSED in the GGUF:
 *       attn_q [d, 4096] = q_and_gate (q 2048 + gate 2048)
 *       attn_k [d, 512]  = kv*hd (2 heads x 256)
 *       attn_v [d, 512]
 *       attn_q_norm [256], attn_k_norm [256]  (SHARED per-head norms)
 *       attn_output [2048, 1024] = W_o (d_out -> d)
 *       NO attn_qkv, NO attn_gate, NO ssm_* on these layers
 *
 * The layer kinds come from the config's layer_types (NOT from which
 * tensors exist): 24 layers = 6 x [3 GDN + 1 gated-attn].
 *
 * The 0.8B reference (verified against the real GGUF, 2026-08-09):
 *   hidden=1024, heads=8, kv=2, head_dim=256, full_attention_interval=4,
 *   GDN: qk_heads=16, v_heads=16, key_hd=128, value_hd=128, conv=4.
 */
#ifndef WUBU_QWEN35_H
#define WUBU_QWEN35_H

#include <stdint.h>
#include <stddef.h>

/* the layer kinds (from the config's layer_types) */
typedef enum {
    WUBU_Q35_LAYER_GDN = 0,        /* linear_attention: Gated DeltaNet */
    WUBU_Q35_LAYER_GATED_ATTN = 1  /* full_attention: gated attention */
} wubu_q35_layer_kind_t;

/* the config (the loader fills these from the GGUF KV) */
typedef struct {
    uint32_t hidden_size;        /* d (1024 on the 0.8B) */
    uint32_t num_attention_heads;/* gated-attn q heads (8) */
    uint32_t num_key_value_heads;/* gated-attn kv heads (2) */
    uint32_t head_dim;           /* gated-attn head dim (256) */
    uint32_t num_hidden_layers;  /* 24 */
    uint32_t full_attention_interval; /* 4 */
    /* the GDN (linear-attention) config */
    uint32_t lin_num_key_heads;  /* 16 */
    uint32_t lin_num_value_heads;/* 16 */
    uint32_t lin_key_head_dim;   /* 128 */
    uint32_t lin_value_head_dim; /* 128 */
    uint32_t lin_conv_kernel;    /* 4 */
    const uint8_t *layer_types;  /* per-layer kind (or NULL to derive) */
} wubu_q35_cfg_t;

/* the computed role math (the GGUF-corrected layout) */
typedef struct {
    /* the GDN attn_qkv [d, gdn_qkv_total]: qk + v */
    uint32_t gdn_qk_off;         /* 0 */
    uint32_t gdn_qk_len;         /* key_dim*2 */
    uint32_t gdn_v_off;          /* after qk */
    uint32_t gdn_v_len;          /* value_dim */
    uint32_t gdn_qkv_total;      /* key_dim*2 + value_dim (6144) */
    /* the GDN attn_gate (the z) */
    uint32_t gdn_z_len;          /* value_dim (2048) */
    /* the GDN ssm_conv1d width == gdn_qkv_total */
    uint32_t gdn_conv_total;
    /* the GDN ssm_out rows == value_dim */
    uint32_t gdn_out_rows;       /* value_dim (2048), NOT 2x */
    /* the gated-attn UNFUSED tensors */
    uint32_t ga_q_and_gate_len;  /* 2*d_out (4096) */
    uint32_t ga_q_len;           /* d_out (2048) */
    uint32_t ga_gate_len;        /* d_out (2048) */
    uint32_t ga_k_len;           /* kv*hd (512) */
    uint32_t ga_v_len;           /* kv*hd (512) */
    uint32_t ga_q_norm_len;      /* head_dim (256, shared per-head) */
    uint32_t ga_k_norm_len;      /* head_dim (256) */
} wubu_q35_split_t;

/* Q1: compute the role math from the config. Returns 0 on success,
 * -1 on a degenerate config. */
int wubu_q35_compute_split(const wubu_q35_cfg_t *cfg, wubu_q35_split_t *out);

/* Q2: the layer kind by index (the config's layer_types, falling back
 * to the interval derivation: layer i is gated-attn when
 * (i+1) % interval == 0 — the 0.8B pattern: 3 GDN then 1 gated). */
wubu_q35_layer_kind_t wubu_q35_layer_kind(const wubu_q35_cfg_t *cfg,
                                          uint32_t layer_idx);

/* Q3: the human-readable summary (for the loader's log). */
void wubu_q35_split_str(const wubu_q35_split_t *s, char *buf, size_t cap);

#endif
