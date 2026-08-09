/* latent_structure.c -- dump the final DDIM latent and measure spatial
 * autocorrelation. A correct latent has structure (corr > 0.5 at dx=1);
 * pure noise has ~0. Also save the latent to /tmp/final_latent.bin. */
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
    float ctx[77*768], pooled[768];
    wubu_sd_clip_encode(clip, "a serene anime landscape, soft morning light, detailed", ctx, pooled);
    float x[4*64*64];
    rng_state = 42;
    for (int i = 0; i < 4*64*64; i++) x[i] = (float)rng_gauss();
    /* scaled-linear schedule */
    float beta[1000], alpha_bar[1000];
    const float bs = sqrtf(0.00085f), be = sqrtf(0.012f);
    for (int t = 0; t < 1000; t++) {
        float b = bs + (be - bs) * (float)t / 999.0f;
        beta[t] = b * b;
        alpha_bar[t] = t == 0 ? (1.0f - beta[0]) : alpha_bar[t - 1] * (1.0f - beta[t]);
    }
    int steps = 12;
    for (int s = 0; s < steps; s++) {
        int t = (int)lrintf(999.0f * (float)(steps - 1 - s) / (float)(steps - 1));
        float noise[4*64*64], nc[4*64*64];
        wubu_sd_unet_forward(unet, x, t, ctx, noise);
        /* uncond = CLIP("") */
        float ectx[77*768], epool[768];
        wubu_sd_clip_encode(clip, "", ectx, epool);
        wubu_sd_unet_forward(unet, x, t, ectx, nc);
        for (int i = 0; i < 4*64*64; i++) noise[i] = nc[i] + 7.5f * (noise[i] - nc[i]);
        float ab_t = alpha_bar[t];
        float ab_prev = (s == steps - 1) ? 1.0f : alpha_bar[lrintf(999.0f * (float)(steps - 2 - s) / (float)(steps - 1))];
        float inv_ab = 1.0f / sqrtf(ab_t), sqrt_1mt = sqrtf(1.0f - ab_t);
        float sqrt_1mp = sqrtf(1.0f - ab_prev), sqrt_abp = sqrtf(ab_prev);
        for (int i = 0; i < 4*64*64; i++) {
            float x0 = (x[i] - sqrt_1mt * noise[i]) * inv_ab;
            x[i] = sqrt_abp * x0 + sqrt_1mp * noise[i];
        }
        fprintf(stderr, "step %d t=%d rms=%.4f\n", s+1, t, sqrt(1.0/16384*0)); /* skip */
    }
    /* final latent stats + autocorr per channel */
    for (int c = 0; c < 4; c++) {
        double mean = 0;
        for (int y = 0; y < 64; y++) for (int xx = 0; xx < 64; xx++) mean += x[(size_t)c*4096 + y*64 + xx];
        mean /= 4096;
        double v = 0;
        for (int i = 0; i < 4096; i++) v += (x[(size_t)c*4096+i] - mean)*(x[(size_t)c*4096+i] - mean);
        v /= 4096;
        /* autocorr dx=1,dy=0 */
        double ac = 0;
        for (int y = 0; y < 64; y++) for (int xx = 0; xx < 63; xx++)
            ac += (x[(size_t)c*4096 + y*64 + xx] - mean) * (x[(size_t)c*4096 + y*64 + xx+1] - mean);
        ac /= 64*63;
        printf("ch%d: rms=%+.4f mean=%+.4f autocorr_dx1=%+.4f %s\n", c, sqrt(v), mean, ac/v,
               fabs(ac/v) > 0.5 ? "STRUCTURED" : "NOISY?");
    }
    FILE *f = fopen("/tmp/final_latent.bin", "wb");
    fwrite(x, 4, 4*64*64, f); fclose(f);
    printf("saved /tmp/final_latent.bin\n");
    return 0;
}
