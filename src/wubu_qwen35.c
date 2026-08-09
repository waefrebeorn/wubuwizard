/*
 * wubu_qwen35.c — the QWEN3.5 HYBRID loader role split (see header).
 */
#include "wubu_qwen35.h"

#include <stdio.h>
#include <string.h>

int wubu_q35_compute_split(const wubu_q35_cfg_t *cfg, wubu_q35_split_t *out)
{
    if (!cfg || !out) return -1;
    if (cfg->hidden_size == 0 || cfg->num_attention_heads == 0 ||
        cfg->num_key_value_heads == 0 || cfg->head_dim == 0 ||
        cfg->lin_num_key_heads == 0 || cfg->lin_num_value_heads == 0 ||
        cfg->lin_key_head_dim == 0 || cfg->lin_value_head_dim == 0)
        return -1;
    memset(out, 0, sizeof(*out));

    /* the gated-attn: d_out = q_heads * hd; the fused is
     * [q_and_gate (2*d_out) | k (kv*hd) | v (kv*hd) | z (d if gate)] */
    uint32_t d_out = cfg->num_attention_heads * cfg->head_dim;
    uint32_t kv_len = cfg->num_key_value_heads * cfg->head_dim;
    out->ga_q_and_gate_off = 0;
    out->ga_q_and_gate_len = 2 * d_out;
    out->ga_k_off = out->ga_q_and_gate_len;
    out->ga_k_len = kv_len;
    out->ga_v_off = out->ga_k_off + kv_len;
    out->ga_v_len = kv_len;
    if (cfg->attn_output_gate) {
        out->ga_z_off = out->ga_v_off + kv_len;
        out->ga_z_len = cfg->hidden_size;
    }
    out->ga_total = out->ga_q_and_gate_len + 2 * kv_len +
                    (cfg->attn_output_gate ? cfg->hidden_size : 0);

    /* the GDN: key_dim = qk_heads * key_hd, value_dim = v_heads * v_hd */
    uint32_t key_dim = cfg->lin_num_key_heads * cfg->lin_key_head_dim;
    uint32_t value_dim = cfg->lin_num_value_heads * cfg->lin_value_head_dim;
    out->gdn_qk_off = 0;
    out->gdn_qk_len = 2 * key_dim;      /* q + k */
    out->gdn_v_off = 2 * key_dim;
    out->gdn_v_len = value_dim;
    out->gdn_z_off = 2 * key_dim + value_dim;
    out->gdn_z_len = value_dim;         /* the z gate (same width as v) */
    out->gdn_total = 2 * key_dim + 2 * value_dim;
    out->gdn_conv_total = 2 * key_dim + value_dim;
    out->gdn_out_rows = 2 * value_dim;  /* the ssm_out [2*value_dim, d] */
    return 0;
}

wubu_q35_layer_kind_t wubu_q35_layer_kind(const wubu_q35_cfg_t *cfg,
                                          uint32_t layer_idx)
{
    if (!cfg) return WUBU_Q35_LAYER_GDN;
    if (cfg->layer_types) {
        return cfg->layer_types[layer_idx] ? WUBU_Q35_LAYER_GATED_ATTN
                                           : WUBU_Q35_LAYER_GDN;
    }
    /* the interval derivation: the (interval-1)-in-N pattern — the
     * 0.8B is 6 x [3 GDN + 1 gated]: gated at indices 3,7,11,15,19,23 */
    if (cfg->full_attention_interval > 0 &&
        (layer_idx + 1) % cfg->full_attention_interval == 0)
        return WUBU_Q35_LAYER_GATED_ATTN;
    return WUBU_Q35_LAYER_GDN;
}

void wubu_q35_split_str(const wubu_q35_split_t *s, char *buf, size_t cap)
{
    if (!s || !buf || cap == 0) return;
    snprintf(buf, cap,
             "gated-attn qkv[%u] = q_and_gate[%u] k[%u] v[%u] z[%u]; "
             "gdn in_proj[%u] = qk[%u] v[%u] z[%u]; conv[%u]; out[%u]",
             s->ga_total, s->ga_q_and_gate_len, s->ga_k_len, s->ga_v_len,
             s->ga_z_len, s->gdn_total, s->gdn_qk_len, s->gdn_v_len,
             s->gdn_z_len, s->gdn_conv_total, s->gdn_out_rows);
}
