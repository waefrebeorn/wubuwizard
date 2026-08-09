/*
 * wubu_qwen35.h — the QWEN3.5 HYBRID ADAPTER's LOADER ROLE SPLIT
 * (AN28 "build next" — resolved 2026-08-09). C11.
 *
 * The fused-tensor role math for the Qwen3.5 hybrid family, driven by
 * the config (NOT hardcoded): given the config, compute the EXACT
 * split offsets for every fused tensor:
 *
 *   - the GATED-ATTN fused attn_qkv [d, 6144]:
 *       6144 = q_and_gate (2*d_out) + k (kv*hd) + v (kv*hd) + z (d)
 *       where d_out = num_attention_heads * head_dim,
 *             kv = num_key_value_heads, hd = head_dim.
 *       The EXTRA d (1024 on the 0.8B) is the gated-attn's own output
 *       gate projection (attn_output_gate: true in the config) fused
 *       at the tail — the AN28 "open question" is RESOLVED by the
 *       config: attn_output_gate + the separate attn_gate [d, 2d]
 *       tensor (the sigmoid gate over the post-W_o output).
 *
 *   - the GDN layer (layer_type == "linear_attention"):
 *       ssm_in_proj [d, key_dim*2 + value_dim*2]  (= 8192 on the 0.8B:
 *       qk 4096 + v 2048 + z 2048 — 16 QK heads x 128 + 16 V heads x 128)
 *       ssm_conv1d [4, key_dim*2 + value_dim]     (= 6144: qk 4096 + v 2048)
 *       ssm_out [value_dim*2, d]                   (= 2048 -> 1024)
 *
 *   - the per-layer kind: layer_types[i] from the config
 *     ("linear_attention" = GDN, "full_attention" = gated-attn), NOT
 *     inferred from which tensors exist (the AN28 doc's heuristic).
 *
 * The 0.8B reference (from the HF config, 2026-08-09):
 *   hidden=1024, heads=8, kv=2, head_dim=256, full_attention_interval=4,
 *   24 layers = 6 x [3 GDN + 1 gated-attn]; GDN: qk_heads=16, v_heads=16,
 *   key_hd=128, value_hd=128, conv_kernel=4.
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
    int      attn_output_gate;   /* 1 = the fused z tail exists */
    const uint8_t *layer_types;  /* per-layer kind (or NULL to derive) */
} wubu_q35_cfg_t;

/* the computed fused-tensor split offsets */
typedef struct {
    /* the gated-attn attn_qkv [d, total] */
    uint32_t ga_q_and_gate_off;  /* 0 */
    uint32_t ga_q_and_gate_len;  /* 2*d_out */
    uint32_t ga_k_off;           /* after q_and_gate */
    uint32_t ga_k_len;           /* kv*hd */
    uint32_t ga_v_off;
    uint32_t ga_v_len;           /* kv*hd */
    uint32_t ga_z_off;           /* after v (0 when no output gate) */
    uint32_t ga_z_len;           /* d when attn_output_gate */
    uint32_t ga_total;           /* the full attn_qkv width */
    /* the GDN ssm_in_proj [d, total] */
    uint32_t gdn_qk_off;         /* 0 */
    uint32_t gdn_qk_len;         /* key_dim*2 */
    uint32_t gdn_v_off;
    uint32_t gdn_v_len;          /* value_dim */
    uint32_t gdn_z_off;          /* after v */
    uint32_t gdn_z_len;          /* value_dim */
    uint32_t gdn_total;          /* key_dim*2 + value_dim*2 */
    /* the GDN ssm_conv1d width (key_dim*2 + value_dim) */
    uint32_t gdn_conv_total;
    /* the GDN ssm_out [value_dim*2, d] */
    uint32_t gdn_out_rows;       /* value_dim*2 */
} wubu_q35_split_t;

/* Q1: compute the split offsets from the config. Returns 0 on
 * success (the config is consistent), -1 on a degenerate config. */
int wubu_q35_compute_split(const wubu_q35_cfg_t *cfg, wubu_q35_split_t *out);

/* Q2: the layer kind by index (the config's layer_types, falling back
 * to the interval derivation: layer i is gated-attn when
 * (i+1) % interval == 0 — the 0.8B pattern: 3 GDN then 1 gated). */
wubu_q35_layer_kind_t wubu_q35_layer_kind(const wubu_q35_cfg_t *cfg,
                                          uint32_t layer_idx);

/* Q3: the human-readable split summary (for the loader's log). */
void wubu_q35_split_str(const wubu_q35_split_t *s, char *buf, size_t cap);

#endif
