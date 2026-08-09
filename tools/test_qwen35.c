/*
 * test_qwen35.c — the QWEN3.5 HYBRID loader role split gate (AN28
 * "build next": the fused-tensor role math resolved from the config).
 *
 * Asserts (against the REAL Qwen3.5-0.8B GGUF, 2026-08-09):
 *   1. the GDN attn_qkv 6144 = qk(4096) + v(2048), attn_gate (z) 2048,
 *      conv 6144, ssm_out rows 2048
 *   2. the gated-attn UNFUSED layout: attn_q 4096 = q(2048) + gate(2048),
 *      attn_k/v 512, shared per-head q/k norms 256
 *   3. the layer kinds: 24 layers = 6 x [3 GDN + 1 gated-attn] from
 *      the config's layer_types (gated at 3,7,11,15,19,23)
 * NOTE: the AN61 config-only inference (the fused 6144 attributed to
 * the gated-attn with an output-gate z tail) was CORRECTED the same
 * day by the on-disk GGUF — the fused 6144 is the GDN's, the gated
 * attention is unfused.
 */
#include <stdio.h>
#include <string.h>

#include "wubu_qwen35.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_qwen35 (the hybrid loader role split) ===\n");

    /* the REAL Qwen3.5-0.8B config (HF config.json, 2026-08-09) */
    wubu_q35_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.hidden_size = 1024;
    cfg.num_attention_heads = 8;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 256;
    cfg.num_hidden_layers = 24;
    cfg.full_attention_interval = 4;
    cfg.lin_num_key_heads = 16;
    cfg.lin_num_value_heads = 16;
    cfg.lin_key_head_dim = 128;
    cfg.lin_value_head_dim = 128;
    cfg.lin_conv_kernel = 4;

    wubu_q35_split_t s;
    if (wubu_q35_compute_split(&cfg, &s) != 0) FAIL("split compute");
    char sb[256];
    wubu_q35_split_str(&s, sb, sizeof(sb));
    printf("  0.8B split: %s\n", sb);

    /* 1. the GDN attn_qkv 6144 = qk(4096) + v(2048) — the GGUF-corrected
     * layout (the fused 6144 belongs to the GDN, NOT the gated-attn) */
    printf("  GDN attn_qkv total %u (expect 6144)\n", s.gdn_qkv_total);
    if (s.gdn_qkv_total != 6144) FAIL("the GDN attn_qkv %u, want 6144",
                                      s.gdn_qkv_total);
    if (s.gdn_qk_len != 4096 || s.gdn_v_len != 2048)
        FAIL("the GDN qk/v %u/%u, want 4096/2048",
             s.gdn_qk_len, s.gdn_v_len);
    if (s.gdn_z_len != 2048) FAIL("the GDN attn_gate (z) %u, want 2048",
                                  s.gdn_z_len);
    if (s.gdn_conv_total != 6144 || s.gdn_out_rows != 2048)
        FAIL("the GDN conv/out %u/%u, want 6144/2048",
             s.gdn_conv_total, s.gdn_out_rows);

    /* 2. the gated-attn UNFUSED layout (the GGUF shows separate q/k/v) */
    printf("  gated-attn attn_q %u = q %u + gate %u (unfused)\n",
           s.ga_q_and_gate_len, s.ga_q_len, s.ga_gate_len);
    if (s.ga_q_and_gate_len != 4096) FAIL("the gated-attn q %u, want 4096",
                                          s.ga_q_and_gate_len);
    if (s.ga_q_len != 2048 || s.ga_gate_len != 2048)
        FAIL("the gated q/gate %u/%u, want 2048/2048",
             s.ga_q_len, s.ga_gate_len);
    if (s.ga_k_len != 512 || s.ga_v_len != 512)
        FAIL("the gated k/v %u/%u, want 512/512", s.ga_k_len, s.ga_v_len);
    if (s.ga_q_norm_len != 256 || s.ga_k_norm_len != 256)
        FAIL("the shared norms %u/%u, want 256/256",
             s.ga_q_norm_len, s.ga_k_norm_len);

    /* 4. the layer kinds: 6 x [3 GDN + 1 gated] = gated at 3,7,...,23 */
    int gated_count = 0;
    for (int i = 0; i < 24; i++)
        if (wubu_q35_layer_kind(&cfg, (uint32_t)i) == WUBU_Q35_LAYER_GATED_ATTN) {
            gated_count++;
            printf("  layer %2d: gated-attn\n", i);
        }
    printf("  gated-attn layers: %d (expect 6)\n", gated_count);
    if (gated_count != 6) FAIL("gated layers %d, want 6", gated_count);

    printf("=== ALL QWEN35 TESTS PASSED (the fused split is resolved) ===\n");
    return 0;
}
