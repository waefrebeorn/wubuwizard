/* vae_pattern.c -- decode constant latents to find spatial bugs.
 * all=0.5 -> smooth mid color. all=1.0, all=-1.0, and a vertical gradient. */
#include "wubu_sd_vae.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void decode(gguf_ctx *g, wubu_sd_vae_t *v, float fill, int mode, const char *label) {
    float latent[4*64*64];
    float img[3*512*512];
    for (int i = 0; i < 4*64*64; i++) {
        int c = i / (64*64), p = i % (64*64);
        int y = p / 64, x = p % 64;
        if (mode == 0) latent[i] = fill;
        else if (mode == 1) latent[i] = fill * (float)x / 63.0f;      /* horizontal grad */
        else latent[i] = fill * (float)y / 63.0f;                     /* vertical grad */
    }
    wubu_sd_vae_decode(v, latent, 64, 64, img);
    double mx = -1e30, mn = 1e30, mean = 0;
    for (int i = 0; i < 3*512*512; i++) {
        double f = img[i]; mean += f;
        if (f > mx) mx = f; if (f < mn) mn = f;
    }
    mean /= 3*512*512;
    /* check for vertical stripes: compare column-mean variance across x */
    double colvar = 0;
    for (int x = 0; x < 512; x++) {
        double cm = 0;
        for (int y = 0; y < 512; y++) cm += img[(size_t)y*512 + x];
        cm /= 512; colvar += (cm - mean)*(cm - mean);
    }
    colvar /= 512;
    printf("%s: mean=%+.4f min=%+.4f max=%+.4f colvar=%g %s\n",
           label, mean, mn, mx, colvar, colvar > 1e-3 ? "STRIPES!" : "smooth");
}
int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g->data_blob) gguf_buffer_data(g);
    wubu_sd_vae_t *v = wubu_sd_vae_load(g);
    decode(g, v, 0.5f, 0, "const 0.5 ");
    decode(g, v, 1.0f, 0, "const 1.0 ");
    decode(g, v, -1.0f, 0, "const -1.0");
    decode(g, v, 1.0f, 1, "x-grad    ");
    decode(g, v, 1.0f, 2, "y-grad    ");
    wubu_sd_vae_free(v);
    return 0;
}
