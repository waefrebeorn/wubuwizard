/* unet_mag.c -- feed pure noise (RMS 1) at t=999, check eps RMS.
 * Correct UNet: eps ~ x (RMS ~1). Broken: RMS >> 1 or << 1. */
#define WUBU_HOSTED
#include "wubu_sd_unet.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static uint64_t rng = 42;
static double rnd(void) { /* xorshift */
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (double)(rng & 0xFFFFFFFFu) / 4294967296.0;
}
static double gauss(void) {
    double u = rnd(); if (u < 1e-12) u = 1e-12;
    double v = rnd();
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}
int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    wubu_sd_unet_t *u = wubu_sd_unet_load(g);
    float latent[4*64*64], ctx[77*768], out[4*64*64];
    for (int i = 0; i < 4*64*64; i++) latent[i] = (float)gauss();  /* RMS~1 */
    for (int i = 0; i < 77*768; i++) ctx[i] = (float)gauss() * 0.1f;
    for (int t = 999; t >= 0; t -= 333) {
        memset(out, 0, sizeof out);
        double t0 = (double)clock() / CLOCKS_PER_SEC;
        wubu_sd_unet_forward(u, latent, t, ctx, out);
        double dt = (double)clock() / CLOCKS_PER_SEC - t0;
        double mean = 0, var = 0;
        for (int i = 0; i < 4*64*64; i++) { mean += out[i]; var += out[i]*out[i]; }
        mean /= 4*64*64; var = var/(4*64*64) - mean*mean;
        printf("t=%4d: eps rms=%.4f mean=%+.5f  (%.1fs)\n", t, sqrt(var), mean, dt);
    }
    return 0;
}
