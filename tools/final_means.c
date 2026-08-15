/* final_means.c -- reproduce the sampler, print FINAL latent channel means.
 * If means are ~0, the latent is fine and the bug is in VAE decode. */
#define WUBU_HOSTED
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
    int steps = 12;
    float sched[13];
    for (int s = 0; s < steps; s++)
        sched[s] = sigmas[(int)lrintf(999.0f - 999.0f * (float)s / (float)(steps - 1))];
    sched[steps] = 0.0f;
    for (int s = 0; s < steps; s++) {
        float sigma = sched[s], sigma_to = sched[s + 1];
        int t = (int)lrintf(999.0f - 999.0f * (float)s / (float)(steps - 1));
        float c_in = 1.0f / sqrtf(sigma * sigma + 1.0f);
        float xscaled[4*64*64];
        for (int i = 0; i < 4*64*64; i++) xscaled[i] = x[i] * c_in;
        float eps[4*64*64], nc[4*64*64];
        wubu_sd_unet_forward(unet, xscaled, t, ctx, eps);
        wubu_sd_unet_forward(unet, xscaled, t, ectx, nc);
        for (int i = 0; i < 4*64*64; i++) eps[i] = nc[i] + 7.5f * (eps[i] - nc[i]);
        float ratio = (sigma_to > 0.0f) ? sigma_to / sigma : 0.0f;
        for (int i = 0; i < 4*64*64; i++) {
            float denoised = x[i] - sigma * eps[i];
            x[i] = ratio * x[i] + (1.0f - ratio) * denoised;
        }
        double xm[4] = {0,0,0,0};
        for (int i = 0; i < 4*64*64; i++) xm[i/4096] += x[i];
        for (int c = 0; c < 4; c++) xm[c] /= 4096;
        printf("s%d t=%3d: x means [%+.3f %+.3f %+.3f %+.3f] rms=%.3f\n",
               s, t, xm[0],xm[1],xm[2],xm[3], sqrt(1.0/16384*(0)));
    }
    return 0;
}
