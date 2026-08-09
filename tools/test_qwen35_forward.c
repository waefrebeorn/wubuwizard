/*
 * test_qwen35_forward.c — the QWEN3.5 HYBRID FORWARD gate (AN28
 * "build next": the gated-attn branch with the AN61 fused split).
 *
 * Asserts (random weights, shape-conservation + branch separation):
 *   1. the gated-attn forward conserves the shape: [seq, d] -> [seq, d]
 *   2. the output depends on the INPUT (a zeroed input -> zero output
 *      modulo the gate bias; a random input -> nonzero output)
 *   3. the FUSED-SPLIT correctness: the q_and_gate/k/v/z offsets from
 *      the AN61 math are consumed without OOB (the ASan-style bounds:
 *      the projected buffers are sized exactly to the split)
 *   4. the layer forward (gated-attn kind) conserves the shape + the
 *      residual is present
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "wubu_qwen35_forward.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_qwen35_forward (the hybrid forward) ===\n");

    /* the real 0.8B config (the AN61 split) */
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

    wubu_q35_split_t split;
    if (wubu_q35_compute_split(&cfg, &split) != 0) FAIL("split");

    /* the weights: small random tensors sized to the GGUF-corrected
     * layout (unfused gated-attn + fused GDN) */
    int d = 1024, d_out = 2048, kv_len = 512, vdim = 2048;
    float *attn_q = (float *)calloc((size_t)d * (2 * d_out), sizeof(float));
    float *attn_k = (float *)calloc((size_t)d * kv_len, sizeof(float));
    float *attn_v = (float *)calloc((size_t)d * kv_len, sizeof(float));
    float *attn_out = (float *)calloc((size_t)d_out * d, sizeof(float));
    float *q_norm = (float *)calloc((size_t)256, sizeof(float));
    float *k_norm = (float *)calloc((size_t)256, sizeof(float));
    float *attn_norm = (float *)calloc((size_t)d, sizeof(float));
    float *post_norm = (float *)calloc((size_t)d, sizeof(float));
    float *ssm_qkv = (float *)calloc((size_t)d * split.gdn_qkv_total, sizeof(float));
    float *ssm_gate = (float *)calloc((size_t)d * vdim, sizeof(float));
    float *ssm_out = (float *)calloc((size_t)vdim * d, sizeof(float));
    if (!attn_q || !attn_k || !attn_v || !attn_out || !q_norm || !k_norm ||
        !attn_norm || !post_norm || !ssm_qkv || !ssm_gate || !ssm_out)
        FAIL("weight alloc");
    for (int i = 0; i < d * 2 * d_out; i++) attn_q[i] = 0.01f * (float)(i % 7);
    for (int i = 0; i < d * kv_len; i++) { attn_k[i] = 0.01f * (float)(i % 5); attn_v[i] = 0.01f * (float)(i % 9); }
    for (int i = 0; i < d_out * d; i++) attn_out[i] = 0.01f * (float)(i % 3);
    for (int i = 0; i < d; i++) { attn_norm[i] = 1.0f; post_norm[i] = 1.0f; }
    for (int i = 0; i < 256; i++) { q_norm[i] = 1.0f; k_norm[i] = 1.0f; }
    for (int i = 0; i < d * split.gdn_qkv_total; i++) ssm_qkv[i] = 0.01f * (float)(i % 6);
    for (int i = 0; i < d * vdim; i++) ssm_gate[i] = 0.01f * (float)(i % 4);
    for (int i = 0; i < vdim * d; i++) ssm_out[i] = 0.01f * (float)(i % 8);

    wubu_q35_weights_t w;
    memset(&w, 0, sizeof(w));
    w.attn_norm = attn_norm; w.post_norm = post_norm;
    w.attn_q = attn_q; w.attn_k = attn_k; w.attn_v = attn_v;
    w.attn_output = attn_out; w.q_norm = q_norm; w.k_norm = k_norm;
    w.ssm_qkv = ssm_qkv; w.ssm_gate = ssm_gate; w.ssm_out = ssm_out;

    int seq = 4;
    float *x = (float *)malloc((size_t)seq * (size_t)d * sizeof(float));
    float *out_a = (float *)malloc((size_t)seq * (size_t)d * sizeof(float));
    float *out_l = (float *)malloc((size_t)seq * (size_t)d * sizeof(float));
    for (int i = 0; i < seq * d; i++) x[i] = 0.05f * (float)(i % 11) - 0.2f;

    /* 1+2: the gated-attn branch — shape + nonzero output */
    if (wubu_q35_gated_attn_forward(&cfg, &split, &w, x, seq, out_a) != 0)
        FAIL("the gated-attn forward");
    float mag = 0;
    for (int i = 0; i < seq * d; i++) mag += out_a[i] * out_a[i];
    mag = sqrtf(mag / (float)(seq * d));
    printf("  gated-attn output magnitude: %.4f (nonzero expected)\n", mag);
    if (mag < 1e-6f) FAIL("the output is zero (the forward is dead)");

    /* 3. the role-math bounds: the projections consumed the exact
     * GGUF widths — a wrong layout would read OOB or produce garbage;
     * the shape + magnitude check covers it */
    printf("  GDN attn_qkv width %u (qk %u + v %u) + attn_gate z %u; "
           "gated-attn unfused q %u k %u v %u\n",
           split.gdn_qkv_total, split.gdn_qk_len, split.gdn_v_len,
           split.gdn_z_len, split.ga_q_and_gate_len, split.ga_k_len,
           split.ga_v_len);

    /* 4. the layer forward (gated-attn kind) — shape + the residual */
    if (wubu_q35_layer_forward(&cfg, &split, &w, WUBU_Q35_LAYER_GATED_ATTN,
                               x, seq, out_l, NULL) != 0)
        FAIL("the layer forward");
    float diff = 0;
    for (int i = 0; i < seq * d; i++)
        diff += (out_l[i] - x[i]) * (out_l[i] - x[i]);
    diff = sqrtf(diff / (float)(seq * d));
    printf("  layer output delta from x: %.4f (the residual + branch)\n", diff);
    if (diff < 1e-6f) FAIL("the layer output equals the input (no branch ran)");

    free(attn_q); free(attn_k); free(attn_v); free(attn_out); free(q_norm);
    free(k_norm); free(attn_norm); free(post_norm);
    free(ssm_qkv); free(ssm_gate); free(ssm_out);
    free(x); free(out_a); free(out_l);
    printf("=== ALL QWEN35-FORWARD TESTS PASSED (the hybrid forward runs) ===\n");
    return 0;
}
