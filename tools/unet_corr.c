/* unet_corr.c -- feed noise x, check corr(x, eps). Working UNet at t=999
 * predicts ~the noise itself: corr > 0.5. Broken: corr ~ 0. */
#define WUBU_HOSTED
#include "wubu_sd_unet.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint64_t rng = 42;
static double rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (double)(rng & 0xFFFFFFFFu) / 4294967296.0; }
static double gauss(void) { double u = rnd(); if (u < 1e-12) u = 1e-12; double v = rnd(); return sqrt(-2.0*log(u))*cos(2.0*M_PI*v); }
int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    wubu_sd_unet_t *u = wubu_sd_unet_load(g);
    float latent[4*64*64], ctx[77*768], out[4*64*64];
    for (int i = 0; i < 4*64*64; i++) latent[i] = (float)gauss();
    for (int i = 0; i < 77*768; i++) ctx[i] = (float)gauss() * 0.1f;
    for (int t = 999; t >= 0; t -= 333) {
        memset(out, 0, sizeof out);
        wubu_sd_unet_forward(u, latent, t, ctx, out);
        double sx=0, sy=0, sxy=0, sxx=0, syy=0;
        for (int i = 0; i < 4*64*64; i++) {
            double xv = latent[i], yv = out[i];
            sx += xv; sy += yv; sxy += xv*yv; sxx += xv*xv; syy += yv*yv;
        }
        int n = 4*64*64;
        double corr = (n*sxy - sx*sy) / sqrt((n*sxx - sx*sx) * (n*syy - sy*sy));
        printf("t=%4d: corr(x,eps)=%+.4f  (working UNet: >0.5; broken: ~0)\n", t, corr);
    }
    return 0;
}
