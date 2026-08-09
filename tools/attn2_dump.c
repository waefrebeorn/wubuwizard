/* attn2_dump.c -- capture attn2 output for cond vs uncond ctx at t=500
 * (mid-schedule where the prompt should matter), via the debug hook. */
#define _GNU_SOURCE
#include "wubu_sd_clip.h"
#include "wubu_sd_unet.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int g_attn2_capture;
extern char g_attn2_prefix[128];
extern float g_attn2_out[3 * 1280 * 4096];
extern int g_attn2_slot;
extern int g_attn2_slot2;

static uint64_t rng_state;
static double rng_rand(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return (double)(rng_state & 0xFFFFFFFFu) / 4294967296.0; }
static double rng_gauss(void) { double u = rng_rand(), v = rng_rand(); if (u < 1e-12) u = 1e-12; return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v); }

int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    wubu_sd_clip_t *clip = wubu_sd_clip_load(g);
    wubu_sd_unet_t *unet = wubu_sd_unet_load(g);
    float ctx[77*768], pooled[768], ectx[77*768], epool[768];
    wubu_sd_clip_encode(clip, "a serene anime landscape, soft morning light, detailed", ctx, pooled);
    wubu_sd_clip_encode(clip, "", ectx, epool);
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

    int t = 500;
    float sigma = sigmas[t];
    float c_in = 1.0f / sqrtf(sigma*sigma + 1.0f);
    float *xs = (float *)malloc(4*64*64*sizeof(float));
    for (int i = 0; i < 4*64*64; i++) xs[i] = x[i] * c_in;

    /* capture attn2 output of an output transformer block */
    snprintf(g_attn2_prefix, sizeof(g_attn2_prefix),
             "model.diffusion_model.output_blocks.11.1.transformer_blocks.0");
    float eps[4*64*64];

    g_attn2_capture = 1; g_attn2_slot = 0; g_attn2_slot2 = 2;
    wubu_sd_unet_forward(unet, xs, t, ctx, eps);
    g_attn2_capture = 1; g_attn2_slot = 1; g_attn2_slot2 = -1;
    wubu_sd_unet_forward(unet, xs, t, ectx, eps);

    /* slot 0/1: attn2 output (ln3) cond vs uncond */
    /* slot 2: x AFTER attn2 residual (cond) */
    int n = 320 * 4096;  /* output_blocks.9.1: C=320, 64x64 */
    double sum=0, sumsq=0, sumd=0, sumd2=0;
    for (int i = 0; i < n; i++) {
        double a = g_attn2_out[i], b = g_attn2_out[n + i], d = a - b;
        sum += a; sumsq += a*a; sumd += d; sumd2 += d*d;
    }
    double rmse = sqrt(sumd2/n), rmsc = sqrt(sumsq/n - (sum/n)*(sum/n));
    printf("attn2 out: cond rms=%.4f  cond-uncond diff rms=%.4f (%.1f%%)\n",
           rmsc, rmse, 100.0*rmse/rmsc);

    /* x post-attn2 (slot 2) vs x pre-attn2: the residual DELTA added by attn2.
     * x_pre = x_post - attn2_out (cond). Compare with total x rms. */
    double s2=0, ss2=0;
    for (int i = 0; i < n; i++) {
        double a = g_attn2_out[2*n + i];
        s2 += a; ss2 += a*a;
    }
    double rmsx = sqrt(ss2/n - (s2/n)*(s2/n));
    printf("x post-attn2: cond rms=%.4f  (attn2 adds %.1f%% of x's magnitude)\n",
           rmsx, 100.0*rmsc/rmsx);
    printf("%s\n", rmse > 0.2*rmsc ? "ATTN2 DIFF FLOWS (cross-attn alive)" : "ATTN2 DIFF DEAD");
    return 0;
}
