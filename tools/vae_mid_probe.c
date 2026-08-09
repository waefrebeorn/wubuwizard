/* vae_mid_probe.c -- decode constant latent, dump mid output (64x64, 512ch)
 * to check for stripes BEFORE the up blocks. */
#define _GNU_SOURCE
#include "wubu_sd_vae.h"
#include "gguf_reader.h"
#include "wubu_sd_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g->data_blob) gguf_buffer_data(g);
    float latent[4*64*64];
    for (int i = 0; i < 4*64*64; i++) latent[i] = 0.5f;
    /* run decode internals manually: conv_in -> mid */
    float *cur = (float *)malloc(512 * 64 * 64 * sizeof(float));
    extern void wubu_sd_conv2d_q(const float *x, int N, int C_in, int H, int W,
                                 const void *w, int wtype, const float *b,
                                 int C_out, int KH, int KW, int stride,
                                 int pad_h, int pad_w, float *y, int *H_out, int *W_out);
    gguf_tensor_info *ti = gguf_find_tensor(g, "first_stage_model.decoder.conv_in.weight");
    wubu_sd_conv2d_q(latent, 1, 4, 64, 64,
                     (const uint8_t*)g->data_blob + ti->data_offset, ti->ggml_type, NULL,
                     512, 3, 3, 1, 1, 1, cur, NULL, NULL);
    /* mid.block_1: resnet (needs norm weights etc) — use the public VAE? It's
     * static... instead check conv_in output for stripes directly. */
    /* colvar of conv_in output per channel */
    for (int c = 0; c < 8; c++) {
        double colvar = 0, mean = 0;
        for (int y = 0; y < 64; y++)
            for (int x = 0; x < 64; x++) mean += cur[(size_t)c*64*64 + y*64 + x];
        mean /= 64*64;
        for (int x = 0; x < 64; x++) {
            double cm = 0;
            for (int y = 0; y < 64; y++) cm += cur[(size_t)c*64*64 + y*64 + x];
            cm /= 64;
            colvar += (cm - mean)*(cm - mean);
        }
        colvar /= 64;
        printf("conv_in ch%02d: mean=%+.4f colvar=%g %s\n", c, mean, colvar,
               colvar > 1e-6 ? "STRIPES!" : "smooth");
    }
    return 0;
}
