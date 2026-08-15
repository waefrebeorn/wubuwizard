/* outblock_trace.c -- trace the cond-vs-uncond signal through the 12
 * output blocks (after each resblock + after each transformer). */
#define WUBU_HOSTED
#include "wubu_sd_clip.h"
#include "wubu_sd_unet.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int g_stage_capture;
extern int g_stage_block;
extern int g_stage_count;
extern float g_stage_out[2][1280 * 64 * 64];

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
    float c_in = 1.0f / sqrtf(sigmas[t]*sigmas[t] + 1.0f);
    float *xs = (float *)malloc(4*64*64*sizeof(float));
    for (int i = 0; i < 4*64*64; i++) xs[i] = x[i] * c_in;
    float eps[4*64*64];

    printf("output-block signal trace at t=%d (cond vs uncond):\n", t);
    for (int b = 0; b < 12; b++) {
        g_stage_block = b;
        g_stage_capture = 5;  /* after resblock */
        wubu_sd_unet_forward(unet, xs, t, ctx, eps);
        int ncap = g_stage_count;
        memcpy(g_stage_out[1], g_stage_out[0], (size_t)ncap * sizeof(float));
        wubu_sd_unet_forward(unet, xs, t, ectx, eps);
        double s=0, ss=0, sd=0, sd2=0;
        int n = ncap;
        for (int i = 0; i < n; i++) {
            double a = g_stage_out[0][i], c = g_stage_out[1][i], d = a - c;
            s += a; ss += a*a; sd += d; sd2 += d*d;
        }
        double rmsc = sqrt(ss/n - (s/n)*(s/n)), rmse = sqrt(sd2/n);
        /* detect the real channel count: where the signal actually lives */
        printf("out.%2d res:  diff rms=%8.4f  (%.1f%%)  [%d elems]\n", b, rmse,
               rmsc > 1e-9 ? 100.0*rmse/rmsc : 0, ncap);
    }
    /* also: middle block input for reference */
    g_stage_capture = 1;
    wubu_sd_unet_forward(unet, xs, t, ctx, eps);
    memcpy(g_stage_out[1], g_stage_out[0], sizeof(g_stage_out[0]));
    wubu_sd_unet_forward(unet, xs, t, ectx, eps);
    double s=0, ss=0, sd=0, sd2=0;
    for (int i = 0; i < 1280*64*64; i++) {
        double a = g_stage_out[0][i], c = g_stage_out[1][i], d = a - c;
        s += a; ss += a*a; sd += d; sd2 += d*d;
    }
    printf("middle in:    diff rms=%8.4f  (%.1f%%)\n", sqrt(sd2/(1280*64*64)),
           100.0*sqrt(sd2/(1280*64*64))/sqrt(ss/(1280*64*64)));
    return 0;
}
