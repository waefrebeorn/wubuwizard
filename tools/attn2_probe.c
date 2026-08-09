/* attn2_probe.c -- compute attn2 (cross) output for cond vs empty ctx on
 * a REAL intermediate. If outputs are nearly identical, the attention is
 * degenerate (flat softmax or wrong scale). */
#define _GNU_SOURCE
#include "wubu_sd_clip.h"
#include "gguf_reader.h"
#include "wubu_sd_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint64_t rng_state;
static double rng_rand(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return (double)(rng_state & 0xFFFFFFFFu) / 4294967296.0; }
static double rng_gauss(void) { double u = rng_rand(), v = rng_rand(); if (u < 1e-12) u = 1e-12; return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v); }

int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g->data_blob) gguf_buffer_data(g);
    wubu_sd_clip_t *clip = wubu_sd_clip_load(g);
    float ctx[77*768], pool[768], ectx[77*768], epool[768];
    wubu_sd_clip_encode(clip, "a serene anime landscape, soft morning light, detailed", ctx, pool);
    wubu_sd_clip_encode(clip, "", ectx, epool);
    /* x = noise at t=999, scaled like the sampler */
    float x[4*64*64];
    rng_state = 42;
    for (int i = 0; i < 4*64*64; i++) x[i] = (float)rng_gauss();
    float beta[1000], alpha_bar[1000];
    const float bs = sqrtf(0.00085f), be = sqrtf(0.012f);
    for (int t = 0; t < 1000; t++) {
        float b = bs + (be - bs) * (float)t / 999.0f;
        beta[t] = b * b;
        alpha_bar[t] = t == 0 ? (1.0f - beta[0]) : alpha_bar[t - 1] * (1.0f - beta[t]);
    }
    float sigmas[1000];
    for (int t = 0; t < 1000; t++) sigmas[t] = sqrtf((1.0f - alpha_bar[t]) / alpha_bar[t]);
    for (int i = 0; i < 4*64*64; i++) x[i] *= sigmas[999];
    float c_in = 1.0f / sqrtf(sigmas[999]*sigmas[999] + 1.0f);
    for (int i = 0; i < 4*64*64; i++) x[i] *= c_in;
    /* attn2 in input_blocks.1.1 (C=320, 8 heads, hd=40) */
    const int C = 320, n = 64*64, hd = 40, H = 8, Tk = 77;
    gguf_tensor_info *ti;
    extern void wubu_sd_linear_q(const float *x, const void *W, int type, int M, int K, int N, float *y);
    /* q from x (approximate ln3 = x for this probe) */
    ti = gguf_find_tensor(g, "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight");
    float *q = (float *)malloc((size_t)n * C * sizeof(float));
    wubu_sd_linear_q(x, (const uint8_t*)g->data_blob + ti->data_offset, ti->ggml_type, n, C, C, q);
    /* k/v for cond and empty */
    ti = gguf_find_tensor(g, "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight");
    float *kc = (float *)malloc((size_t)Tk * C * sizeof(float));
    float *ke = (float *)malloc((size_t)Tk * C * sizeof(float));
    wubu_sd_linear_q(ctx, (const uint8_t*)g->data_blob + ti->data_offset, ti->ggml_type, Tk, 768, C, kc);
    wubu_sd_linear_q(ectx, (const uint8_t*)g->data_blob + ti->data_offset, ti->ggml_type, Tk, 768, C, ke);
    ti = gguf_find_tensor(g, "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight");
    float *vc = (float *)malloc((size_t)Tk * C * sizeof(float));
    float *ve = (float *)malloc((size_t)Tk * C * sizeof(float));
    wubu_sd_linear_q(ctx, (const uint8_t*)g->data_blob + ti->data_offset, ti->ggml_type, Tk, 768, C, vc);
    wubu_sd_linear_q(ectx, (const uint8_t*)g->data_blob + ti->data_offset, ti->ggml_type, Tk, 768, C, ve);
    /* attention for ONE query (position 0, head 0) */
    const float scale = 1.0f / sqrtf((float)hd);
    float q0[hd];
    memcpy(q0, q + 0*C + 0*hd, hd*4);
    /* cond attn weights */
    float wc[77], we2[77];
    double mx = -1e30;
    for (int j = 0; j < 77; j++) {
        double s = 0;
        for (int d = 0; d < hd; d++) s += q0[d] * kc[j*C + d];
        wc[j] = s * scale; if (wc[j] > mx) mx = wc[j];
    }
    double sumc = 0;
    for (int j = 0; j < 77; j++) { wc[j] = expf(wc[j] - mx); sumc += wc[j]; }
    mx = -1e30;
    for (int j = 0; j < 77; j++) {
        double s = 0;
        for (int d = 0; d < hd; d++) s += q0[d] * ke[j*C + d];
        we2[j] = s * scale; if (we2[j] > mx) mx = we2[j];
    }
    double sume = 0;
    for (int j = 0; j < 77; j++) { we2[j] = expf(we2[j] - mx); sume += we2[j]; }
    /* output vectors */
    float oc[40], oe[40];
    memset(oc, 0, sizeof oc); memset(oe, 0, sizeof oe);
    for (int j = 0; j < 77; j++)
        for (int d = 0; d < hd; d++) {
            oc[d] += (wc[j]/sumc) * vc[j*C + d];
            oe[d] += (we2[j]/sume) * ve[j*C + d];
        }
    double dod = 0, roc = 0;
    for (int d = 0; d < hd; d++) { dod += (oc[d]-oe[d])*(oc[d]-oe[d]); roc += oc[d]*oc[d]; }
    printf("attn2 out: cond rms=%+.4f cond-vs-empty diff rms=%+.4f %s\n",
           sqrt(roc/hd), sqrt(dod/hd),
           sqrt(dod/hd) > 0.2*sqrt(roc/hd) ? "DIFFERS" : "DEGENERATE (flat softmax?)");
    /* entropy-ish check: how peaked is the cond distribution? */
    double e = 0;
    for (int j = 0; j < 77; j++) { double p = wc[j]/sumc; if (p > 1e-12) e -= p*log(p); }
    printf("cond softmax entropy: %.3f of %.3f max (flat=%0.3f)\n", e, log(77.0), log(77.0));
    return 0;
}
