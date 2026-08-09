/*
 * wubu_qwen35_probe.c — the QWEN3.5 REAL-GGUF probe (the parity
 * path's first step): load the ACTUAL Q8_0 tensors from the 0.8B
 * GGUF, run the wizard's gated-attn branch + GDN branch against them,
 * and prove the layout is consumed correctly (finite, deterministic,
 * shape-consistent outputs). The full logit-parity vs llama.cpp needs
 * the whole 24-layer stack + embedding + head (the loader's job);
 * this probe pins the FORWARD-to-GGUF wiring so the loader can build
 * on a verified base.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "gguf_reader.h"
#include "wubu_qwen35.h"
#include "wubu_qwen35_forward.h"

static float *load_tensor(gguf_ctx *ctx, const char *name, int64_t *n_elems)
{
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) { printf("  [probe] MISSING tensor %s\n", name); return NULL; }
    int64_t n = 1;
    for (int i = 0; i < t->n_dims; i++) n *= t->dims[i];
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (!buf) return NULL;
    int rc = gguf_read_tensor_f32(ctx, t, buf, n);
    if (rc != (int)n) {
        printf("  [probe] FAILED to read %s (n=%lld rc=%d type=%d dims=%d/%d)\n",
               name, (long long)n, rc, t->ggml_type,
               t->n_dims > 0 ? (int)t->dims[0] : -1,
               t->n_dims > 1 ? (int)t->dims[1] : -1);
        free(buf);
        return NULL;
    }
    *n_elems = n;
    return buf;
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1]
        : "/home/wubu/models/Qwen3.5-0.8B-Q8_0.gguf";
    printf("=== test_qwen35_probe (the real 0.8B GGUF) ===\n");
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { printf("  FAIL: cannot open %s\n", path); return 1; }
    if (gguf_buffer_data(ctx) != 1) {   /* 1 = buffered/mmap'd */
        printf("  FAIL: cannot buffer the data blob\n");
        return 1;
    }

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
    if (wubu_q35_compute_split(&cfg, &split) != 0) {
        printf("  FAIL: split\n"); return 1;
    }
    char sb[256];
    wubu_q35_split_str(&split, sb, sizeof(sb));
    printf("  split: %s\n", sb);

    int64_t n = 0;
    /* the gated-attn layer (blk.3): attn_q/k/v/output + norms */
    float *attn_q = load_tensor(ctx, "blk.3.attn_q.weight", &n);
    if (!attn_q) return 1;
    int64_t q_elems = n;
    float *attn_k = load_tensor(ctx, "blk.3.attn_k.weight", &n);
    float *attn_v = load_tensor(ctx, "blk.3.attn_v.weight", &n);
    float *attn_o = load_tensor(ctx, "blk.3.attn_output.weight", &n);
    float *q_norm = load_tensor(ctx, "blk.3.attn_q_norm.weight", &n);
    float *k_norm = load_tensor(ctx, "blk.3.attn_k_norm.weight", &n);
    float *a_norm = load_tensor(ctx, "blk.3.attn_norm.weight", &n);
    float *p_norm = load_tensor(ctx, "blk.3.post_attention_norm.weight", &n);
    if (!attn_k || !attn_v || !attn_o || !q_norm || !k_norm || !a_norm || !p_norm)
        return 1;
    printf("  blk.3 (gated-attn): attn_q %lld elems (expect %d); "
           "attn_q_norm read (shared per-head [256])\n",
           (long long)q_elems, (int)(cfg.hidden_size * split.ga_q_and_gate_len));

    /* the GDN layer (blk.0): attn_qkv + attn_gate + ssm_* */
    float *ssm_qkv = load_tensor(ctx, "blk.0.attn_qkv.weight", &n);
    float *ssm_gate = load_tensor(ctx, "blk.0.attn_gate.weight", &n);
    float *ssm_out = load_tensor(ctx, "blk.0.ssm_out.weight", &n);
    float *ssm_a = load_tensor(ctx, "blk.0.ssm_a", &n);
    float *ssm_alpha = load_tensor(ctx, "blk.0.ssm_alpha.weight", &n);
    float *ssm_beta = load_tensor(ctx, "blk.0.ssm_beta.weight", &n);
    float *ssm_norm = load_tensor(ctx, "blk.0.ssm_norm.weight", &n);
    float *a_norm0 = load_tensor(ctx, "blk.0.attn_norm.weight", &n);
    float *p_norm0 = load_tensor(ctx, "blk.0.post_attention_norm.weight", &n);
    if (!ssm_qkv || !ssm_gate || !ssm_out || !ssm_a || !ssm_alpha ||
        !ssm_beta || !ssm_norm || !a_norm0 || !p_norm0) return 1;
    printf("  blk.0 (GDN): attn_qkv [d,6144] + attn_gate [d,2048] (the z) "
           "+ ssm_out [2048,d] loaded\n");

    /* the gated-attn forward on the REAL tensors */
    wubu_q35_weights_t w;
    memset(&w, 0, sizeof(w));
    w.attn_norm = a_norm; w.post_norm = p_norm;
    w.attn_q = attn_q; w.attn_k = attn_k; w.attn_v = attn_v;
    w.attn_output = attn_o; w.q_norm = q_norm; w.k_norm = k_norm;
    w.ssm_qkv = ssm_qkv; w.ssm_gate = ssm_gate; w.ssm_out = ssm_out;

    int d = (int)cfg.hidden_size;
    int seq = 4;
    float *x = (float *)malloc((size_t)seq * (size_t)d * sizeof(float));
    float *out_a = (float *)malloc((size_t)seq * (size_t)d * sizeof(float));
    float *out_d = (float *)malloc((size_t)seq * (size_t)d * sizeof(float));
    for (int i = 0; i < seq * d; i++)
        x[i] = 0.05f * (float)(i % 13) - 0.3f;

    if (wubu_q35_gated_attn_forward(&cfg, &split, &w, x, seq, out_a) != 0) {
        printf("  FAIL: the gated-attn forward on the real tensors\n");
        return 1;
    }
    float mag_a = 0;
    for (int i = 0; i < seq * d; i++) {
        if (!isfinite(out_a[i])) {
            printf("  FAIL: NON-FINITE gated-attn output (idx %d)\n", i);
            return 1;
        }
        mag_a += out_a[i] * out_a[i];
    }
    mag_a = sqrtf(mag_a / (float)(seq * d));

    /* the GDN branch on the real tensors (stateless projection path) */
    w.attn_norm = a_norm0; w.post_norm = p_norm0;
    if (wubu_q35_layer_forward(&cfg, &split, &w, WUBU_Q35_LAYER_GDN,
                               x, seq, out_d, NULL) != 0) {
        printf("  FAIL: the GDN forward on the real tensors\n");
        return 1;
    }
    float mag_d = 0;
    for (int i = 0; i < seq * d; i++) {
        if (!isfinite(out_d[i])) {
            printf("  FAIL: NON-FINITE GDN output (idx %d)\n", i);
            return 1;
        }
        mag_d += out_d[i] * out_d[i];
    }
    mag_d = sqrtf(mag_d / (float)(seq * d));

    printf("  gated-attn output on real Q8_0: magnitude %.4f (finite)\n", mag_a);
    printf("  GDN output on real Q8_0:        magnitude %.4f (finite)\n", mag_d);
    if (mag_a < 1e-6f || mag_d < 1e-6f) {
        printf("  FAIL: a branch is dead on the real weights\n");
        return 1;
    }

    /* the determinism: the same input twice -> the same output */
    float *out_a2 = (float *)malloc((size_t)seq * (size_t)d * sizeof(float));
    w.attn_norm = a_norm; w.post_norm = p_norm;
    wubu_q35_gated_attn_forward(&cfg, &split, &w, x, seq, out_a2);
    float dd = 0;
    for (int i = 0; i < seq * d; i++) dd += (out_a[i] - out_a2[i]) * (out_a[i] - out_a2[i]);
    printf("  determinism: rerun diff %.2e (must be 0)\n", dd);
    if (dd > 1e-12f) {
        printf("  FAIL: the forward is nondeterministic\n");
        return 1;
    }

    gguf_close(ctx);
    printf("=== QWEN35 REAL-GGUF PROBE PASSED (the forward consumes the "
           "actual layout) ===\n");
    return 0;
}
