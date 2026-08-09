/* cfg_scan.c -- cond vs uncond eps diff across the whole schedule.
 * At t=999 (max noise) prompt influence is small; it must GROW at
 * mid/low t where the prompt guides structure. */
#define _GNU_SOURCE
#include "wubu_sd_clip.h"
#include "wubu_sd_unet.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

    int tests[] = {999, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 10, 0};
    int nts = sizeof(tests)/sizeof(tests[0]);
    printf("timestep   sigma     cond-rms  diff-rms  diff/cond\n");
    for (int ti = 0; ti < nts; ti++) {
        int t = tests[ti];
        float sigma = sigmas[t];
        float c_in = 1.0f / sqrtf(sigma*sigma + 1.0f);
        float *xs = (float *)malloc(4*64*64*sizeof(float));
        for (int i = 0; i < 4*64*64; i++) xs[i] = x[i] * c_in;
        float *eps = (float *)malloc(4*64*64*sizeof(float));
        float *nc  = (float *)malloc(4*64*64*sizeof(float));
        wubu_sd_unet_forward(unet, xs, t, ctx, eps);
        wubu_sd_unet_forward(unet, xs, t, ectx, nc);
        double sum=0, sumsq=0, sumd=0, sumd2=0;
        for (int i = 0; i < 4*64*64; i++) {
            double a = eps[i], b = nc[i], d = a - b;
            sum += a; sumsq += a*a; sumd += d; sumd2 += d*d;
        }
        int n = 4*64*64;
        double rmse = sqrt(sumd2/n), rmsc = sqrt(sumsq/n - (sum/n)*(sum/n));
        printf("%4d    %7.3f  %8.4f  %8.4f  %6.1f%%%s\n", t, sigma, rmsc, rmse,
               100.0*rmse/rmsc, rmse > 0.3*rmsc ? "  <-- strong" : "");
        free(xs); free(eps); free(nc);
    }
    return 0;
}
