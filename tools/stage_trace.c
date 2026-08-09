/* stage_trace.c -- trace where the cond-vs-uncond signal dies:
 * stage 1 = middle-block input, 2 = after middle transformer,
 * 3 = final pre-conv (out.0 input), 4 = eps (out.2 output). */
#define _GNU_SOURCE
#include "wubu_sd_clip.h"
#include "wubu_sd_unet.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int g_stage_capture;
extern float g_stage_out[2][1280 * 64 * 64];

static uint64_t rng_state;
static double rng_rand(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return (double)(rng_state & 0xFFFFFFFFu) / 4294967296.0; }
static double rng_gauss(void) { double u = rng_rand(), v = rng_rand(); if (u < 1e-12) u = 1e-12; return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v); }

static void rms_diff(const float *a, const float *b, int n, const char *label, double *rms_out, double *diff_out) {
    double sum=0, sumsq=0, sumd=0, sumd2=0;
    for (int i = 0; i < n; i++) {
        double x = a[i], y = b[i], d = x - y;
        sum += x; sumsq += x*x; sumd += d; sumd2 += d*d;
    }
    double rmsc = sqrt(sumsq/n - (sum/n)*(sum/n)), rmse = sqrt(sumd2/n);
    printf("%-28s cond rms=%8.4f  diff rms=%8.4f  (%.1f%%)\n", label, rmsc, rmse, 100.0*rmse/rmsc);
    *rms_out = rmsc; *diff_out = rmse;
}

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
    float c_in = 1.0f / sqrtf(sigmas[t]*sigmas[t] + 1.0f);
    float *xs = (float *)malloc(4*64*64*sizeof(float));
    for (int i = 0; i < 4*64*64; i++) xs[i] = x[i] * c_in;
    float eps[4*64*64];

    printf("signal trace at t=%d (cond vs uncond):\n", t);
    for (int stage = 1; stage <= 4; stage++) {
        double r1=0, d1=0, r2=0, d2=0;
        g_stage_capture = stage;
        wubu_sd_unet_forward(unet, xs, t, ctx, eps);
        memcpy(g_stage_out[1], g_stage_out[0], sizeof(g_stage_out[0]));
        wubu_sd_unet_forward(unet, xs, t, ectx, eps);
        int n = (stage == 4) ? 4*64*64 : (stage == 3 ? 320*64*64 : 1280*64*64);
        const char *lab = stage==1 ? "middle in" : stage==2 ? "after mid attn" :
                          stage==3 ? "pre out conv" : "eps (out)";
        rms_diff(g_stage_out[0], g_stage_out[1], n, lab, &r1, &d1);
    }
    return 0;
}
