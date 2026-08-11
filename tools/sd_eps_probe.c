/* sd_eps_probe.c — run OUR C11 UNet on the same noise + t, dump eps .bin.
 * Mirror of the sd.cpp oracle harness (sd_eps_compare).
 * Usage: sd_eps_probe model.gguf noise.bin t out_eps.bin
 */
#include "wubu_sd_unet.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model.gguf noise.bin t out_eps.bin\n", argv[0]); return 1; }
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g) return 1;
    wubu_sd_unet_t *u = wubu_sd_unet_load(g);
    if (!u) return 1;

    FILE *nf = fopen(argv[2], "rb");
    if (!nf) { perror("noise"); return 1; }
    float noise[4*64*64];
    if (fread(noise, 4, 4*64*64, nf) != (size_t)(4*64*64)) { fprintf(stderr, "short noise\n"); return 1; }
    fclose(nf);

    int t = atoi(argv[3]);
    const char *out_path = argv[4];

    float beta[1000], ab[1000];
    for (int i = 0; i < 1000; i++) {
        float tt = (float)i / 999.0f;
        float b = (sqrtf(0.00085f) + (sqrtf(0.012f)-sqrtf(0.00085f))*tt);
        beta[i] = b*b;
        ab[i] = (i==0) ? (1.0f-beta[0]) : ab[i-1]*(1.0f-beta[i]);
    }
    float smax = sqrtf((1.0f-ab[999])/ab[999]);
    float sigma = sqrtf((1.0f-ab[t])/ab[t]);
    float c_in = 1.0f/sqrtf(sigma*sigma+1.0f);

    float x[4*64*64];
    for (int i = 0; i < 4*64*64; i++) x[i] = noise[i] * smax * c_in;

    float ctx[77*768]; memset(ctx, 0, sizeof(ctx));
    extern int g_stage_capture;
    extern int g_stage_block;
    extern float g_stage_out[5][1280 * 64 * 64];
    const char *capgn = getenv("SD_CAP_GN");
    /* bit 1 -> slot0 (first_rb, 320ch@64x64), bit 2 -> slot1 (mid_in, 1280ch@8x8),
     * bit 4 -> slot2 (mid after attn), bit 8 -> slot3 (pre-gn), bit 16 -> slot4 (eps out) */
    g_stage_capture = capgn ? (1 | 2 | 4 | 8 | 16) : 0;
    g_stage_block = 0;
    float out[4*64*64];
    if (wubu_sd_unet_forward(u, x, t, ctx, out) != 0) { fprintf(stderr, "fwd fail\n"); return 1; }
    if (capgn) {
        char fn[512];
        /* Capture conv0 (via g_stage_out[4] post-gn slot repurposed) and ib2/etc */
        struct { int slot; int ch; int hw; const char *name; } caps[] = {
            { 0, 320, 64*64,   "our_first_rb" },
            { 1, 1280, 8*8,    "our_mid_in"   },
            { 2, 1280, 8*8,    "our_mid_attn" },
            { 3, 320, 64*64,   "our_pre_gn"   },
        };
        for (unsigned k = 0; k < sizeof(caps)/sizeof(caps[0]); k++) {
            snprintf(fn, sizeof(fn), "%s/%s.bin", capgn, caps[k].name);
            FILE *cf = fopen(fn, "wb");
            if (cf) { fwrite(g_stage_out[caps[k].slot], 4, (size_t)caps[k].ch * caps[k].hw, cf); fclose(cf); }
        }
        /* Also capture downsample and ib2/ib4/ib7/ib8 outputs via SD_DEBUG_CAP */
        const char *dbg = getenv("SD_DEBUG_CAP");
        if (dbg) {
            /* These are captured inside wubu_sd_unet_forward via SD_DEBUG_CAP */
        }
    }

    FILE *of = fopen(out_path, "wb");
    if (!of) { perror("out"); return 1; }
    fwrite(out, 4, 4*64*64, of);
    fclose(of);

    for (int c = 0; c < 4; c++) {
        double s = 0, ss = 0;
        for (int i = 0; i < 4096; i++) { double v = out[c*4096+i]; s += v; ss += v*v; }
        printf("ch%d: mean=%.5f rms=%.5f\n", c, s/4096.0, sqrt(ss/4096.0));
    }
    double s = 0, ss = 0;
    for (int i = 0; i < 4*64*64; i++) { double v = out[i]; s += v; ss += v*v; }
    printf("total: mean=%.5f rms=%.5f\n", s/16384.0, sqrt(ss/16384.0));

    wubu_sd_unet_free(u);
    return 0;
}
