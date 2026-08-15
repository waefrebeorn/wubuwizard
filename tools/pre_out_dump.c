/* pre_out_dump.c -- capture OUR pre-out activation (stage 3) on zero input
 * and dump per-channel means/rms, for direct comparison with sd.cpp's
 * wz_pre_out capture (rms 3.83, per-channel means +/- 1-2). */
#define WUBU_HOSTED
#include "wubu_sd_unet.h"
#include "wubu_sd_ops.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int g_stage_capture;
extern float g_stage_out[2][1280 * 64 * 64];
extern int g_stage_count;

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [out.bin]\n", argv[0]); return 1; }
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g) return 1;
    wubu_sd_unet_t *unet = wubu_sd_unet_load(g);
    if (!unet) return 1;

    float x[4*64*64];
    FILE *f = fopen("/tmp/sd_zero.bin", "rb");
    if (!f || fread(x, sizeof(float), 4*64*64, f) != (size_t)(4*64*64)) {
        fprintf(stderr, "zero read failed\n"); return 1;
    }
    fclose(f);

    float bs = sqrtf(0.00085f), be = sqrtf(0.012f);
    float alpha_bar = 1.0f;
    for (int i = 0; i < 1000; i++) {
        float b = bs + (be - bs) * (float)i / 999.0f;
        alpha_bar *= 1.0f - b * b;
    }
    float sigma_max = sqrtf((1.0f - alpha_bar) / alpha_bar);
    float c_in = 1.0f / sqrtf(sigma_max * sigma_max + 1.0f);
    for (int i = 0; i < 4*64*64; i++) x[i] *= sigma_max * c_in;

    float ctx[77*768];
    memset(ctx, 0, sizeof(ctx));

    g_stage_capture = 3;   /* pre-out activation (before out.0 gn) */
    float eps[4*64*64];
    if (wubu_sd_unet_forward(unet, x, 999, ctx, eps) != 0) { fprintf(stderr, "forward failed\n"); return 1; }

    int nch = 320, hw = 64*64;
    double all = 0, allq = 0;
    for (int c = 0; c < nch; c++) {
        double s = 0, sq = 0;
        for (int i = 0; i < hw; i++) {
            double v = g_stage_out[0][(size_t)c * hw + i];
            s += v; sq += v*v;
        }
        if (c < 6 || c >= nch - 2)
            printf("pre_out ch%d: mean=%+.4f rms=%.4f\n", c, s/hw, sqrt(sq/hw));
        all += s; allq += sq;
    }
    printf("pre_out ALLCH: mean=%+.4f rms=%.4f\n", all/(nch*hw), sqrt(allq/(nch*hw)));

    double es[4] = {0}, esq[4] = {0};
    for (int c = 0; c < 4; c++)
        for (int i = 0; i < hw; i++) { double v = eps[(size_t)c*hw+i]; es[c] += v; esq[c] += v*v; }
    for (int c = 0; c < 4; c++)
        printf("eps ch%d: mean=%+.4f rms=%.4f\n", c, es[c]/hw, sqrt(esq[c]/hw));

    if (argc > 2) {
        FILE *o = fopen(argv[2], "wb");
        fwrite(g_stage_out[0], sizeof(float), 320*hw, o);
        fclose(o);
        printf("wrote pre_out to %s\n", argv[2]);
    }
    return 0;
}
