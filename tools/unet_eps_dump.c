/* unet_eps_dump.c -- run OUR C11 UNet on the exact same input as the
 * sd.cpp harness (noise file, sigma_max*c_in scaling, t=999, ZERO context)
 * and dump eps + stats for direct comparison. */
#define WUBU_HOSTED
#include "wubu_sd_unet.h"
#include "wubu_sd_ops.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf noise.bin [out.eps]\n", argv[0]); return 1; }
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g) return 1;
    wubu_sd_unet_t *unet = wubu_sd_unet_load(g);
    if (!unet) return 1;

    float x[4*64*64];
    FILE *f = fopen(argv[2], "rb");
    if (!f || fread(x, sizeof(float), 4*64*64, f) != (size_t)(4*64*64)) {
        fprintf(stderr, "noise read failed\n"); return 1;
    }
    fclose(f);

    /* same scaling as the harness: x *= sigma_max * c_in */
    float bs = sqrtf(0.00085f), be = sqrtf(0.012f);
    float alpha_bar = 1.0f;
    for (int i = 0; i < 1000; i++) {
        float b = bs + (be - bs) * (float)i / 999.0f;
        alpha_bar *= 1.0f - b * b;
    }
    float sigma_max = sqrtf((1.0f - alpha_bar) / alpha_bar);
    float c_in = 1.0f / sqrtf(sigma_max * sigma_max + 1.0f);
    for (int i = 0; i < 4*64*64; i++) x[i] *= sigma_max * c_in;

    /* zero context (77x768) */
    float ctx[77*768];
    memset(ctx, 0, sizeof(ctx));

    float eps[4*64*64];
    if (wubu_sd_unet_forward(unet, x, 999, ctx, eps) != 0) { fprintf(stderr, "forward failed\n"); return 1; }

    double sum[4] = {0}, sumsq[4] = {0};
    for (int c = 0; c < 4; c++)
        for (int i = 0; i < 64*64; i++) {
            double v = eps[(size_t)c*64*64 + i];
            sum[c] += v; sumsq[c] += v*v;
        }
    for (int c = 0; c < 4; c++) {
        double mean = sum[c] / (64.0*64.0);
        double rms = sqrt(sumsq[c] / (64.0*64.0));
        printf("ch%d: mean=%+.4f rms=%.4f\n", c, mean, rms);
    }
    if (argc > 3) {
        FILE *o = fopen(argv[3], "wb");
        fwrite(eps, sizeof(float), 4*64*64, o);
        fclose(o);
        printf("wrote %s\n", argv[3]);
    }
    return 0;
}
