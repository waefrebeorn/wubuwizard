/*
 * test_qwen35.c — the QWEN3.5 HYBRID loader role split gate (AN28
 * "build next": the fused-tensor role math resolved from the config).
 *
 * Asserts (against the REAL 0.8B config from HF, 2026-08-09):
 *   1. the gated-attn fused 6144 = q_and_gate(4096) + k(512) + v(512)
 *      + z(1024) — the AN28 open question (the extra 1024) RESOLVED
 *      as the output-gate tail (attn_output_gate: true)
 *   2. the GDN ssm_in_proj 8192 = qk(4096) + v(2048) + z(2048)
 *   3. the GDN conv 6144 = qk(4096) + v(2048)
 *   4. the layer kinds: 24 layers = 6 x [3 GDN + 1 gated-attn] from
 *      the config's layer_types (gated at 3,7,11,15,19,23)
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
    cfg.attn_output_gate = 1;

    wubu_q35_split_t s;
    if (wubu_q35_compute_split(&cfg, &s) != 0) FAIL("split compute");
    char sb[256];
    wubu_q35_split_str(&s, sb, sizeof(sb));
    printf("  0.8B split: %s\n", sb);

    /* 1. the gated-attn fused 6144 (the AN28 open question) */
    printf("  gated-attn attn_qkv total %u (expect 6144)\n", s.ga_total);
    if (s.ga_total != 6144) FAIL("the fused attn_qkv total is %u, want 6144",
                                 s.ga_total);
    if (s.ga_q_and_gate_len != 4096) FAIL("q_and_gate %u, want 4096",
                                          s.ga_q_and_gate_len);
    if (s.ga_k_len != 512 || s.ga_v_len != 512)
        FAIL("kv len %u/%u, want 512/512", s.ga_k_len, s.ga_v_len);
    if (s.ga_z_len != 1024) FAIL("the output-gate z %u, want 1024",
                                 s.ga_z_len);
    if (s.ga_v_off != 4608 || s.ga_z_off != 5120)
        FAIL("the fused offsets wrong (%u/%u)", s.ga_v_off, s.ga_z_off);

    /* 2. the GDN ssm_in_proj 8192 */
    printf("  GDN ssm_in_proj total %u (expect 8192)\n", s.gdn_total);
    if (s.gdn_total != 8192) FAIL("the GDN in_proj %u, want 8192",
                                  s.gdn_total);
    if (s.gdn_qk_len != 4096 || s.gdn_v_len != 2048 || s.gdn_z_len != 2048)
        FAIL("the GDN qk/v/z %u/%u/%u, want 4096/2048/2048",
             s.gdn_qk_len, s.gdn_v_len, s.gdn_z_len);

    /* 3. the GDN conv 6144 */
    printf("  GDN ssm_conv1d width %u (expect 6144)\n", s.gdn_conv_total);
    if (s.gdn_conv_total != 6144) FAIL("the GDN conv %u, want 6144",
                                       s.gdn_conv_total);
    if (s.gdn_out_rows != 4096) FAIL("the ssm_out rows %u, want 4096",
                                     s.gdn_out_rows);

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
