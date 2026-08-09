/* eps_bias.c -- UNet eps per-channel mean for zero + noise inputs.
 * A correct UNet: eps per-channel mean ~0 for zero input (just bias).
 * Large channel means = bias leak (missing bias, wrong norm, etc). */
#define _GNU_SOURCE
#include "wubu_sd_unet.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    wubu_sd_unet_t *u = wubu_sd_unet_load(g);
    float latent[4*64*64], ctx[77*768], out[4*64*64];
    memset(latent, 0, sizeof latent);
    memset(ctx, 0, sizeof ctx);
    for (int t = 999; t >= 0; t -= 333) {
        memset(out, 0, sizeof out);
        wubu_sd_unet_forward(u, latent, t, ctx, out);
        for (int c = 0; c < 4; c++) {
            double m = 0;
            for (int i = 0; i < 4096; i++) m += out[(size_t)c*4096 + i];
            m /= 4096;
            printf("t=%4d zero-input ch%d mean=%+.5f\n", t, c, m);
        }
    }
    return 0;
}
