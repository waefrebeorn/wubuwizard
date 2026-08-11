/* wubu_sd_taesd.c -- Tiny AutoEncoder for Stable Diffusion (TAESD) decoder.
 *
 * Architecture (madebyollin/taesd, decoder only, all 3x3 s1 p1 convs):
 *   Clamp(tanh(x/3)*3)                          latent 4xHxW
 *   conv(4->64) + ReLU
 *   [Block(64,64)] x3                          Block = conv->ReLU->conv->ReLU
 *                                               ->conv, + skip (identity),
 *                                               ReLU fuse
 *   Upsample(2) + conv(64->64, no bias)        -> 8H x 8W
 *   [Block(64,64)] x3
 *   Upsample(2) + conv(64->64, no bias)
 *   [Block(64,64)] x3
 *   Upsample(2) + conv(64->64, no bias)
 *   Block(64,64)
 *   conv(64->3)                                img 3x8Hx8W in [0,1]
 *
 * Module numbering (Sequential): 0=Clamp 1=conv 2=ReLU 3-5=Blocks
 * 6=Up 7=conv 8-10=Blocks 11=Up 12=conv 13-15=Blocks 16=Up 17=conv
 * 18=Block 19=conv(64->3). Each Block owns 3 convs "N.conv.{0,2,4}".
 *
 * Weights are F32 safetensors [C_out][C_in][KH][KW]; converted to the
 * conv2d_q k-order (k = ci*9 + kw*3 + kh, kh innermost).
 *
 * NOTE: TAESD's latent scale_factor is 1 (NOT 1/0.18215) and its output
 * is [0,1] (NOT [-1,1]) — caller must NOT apply the SD VAE scale, and
 * must map [0,1] -> pixels directly.
 */
#include "wubu_sd_taesd.h"
#include "safetensors_reader.h"
#include "wubu_sd_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* slot = mod*3 + sub (sub in {0,1,2} for block convs, 0 otherwise) */
struct wubu_sd_taesd {
    uint16_t *w_f16[64];
    float    *bias[64];
    int       has[64];
};

static int load_conv(wubu_sd_taesd_t *t, st_ctx *st, int mod, int sub,
                     int C_in, int C_out, int is_block) {
    char nm[256];
    if (is_block)
        snprintf(nm, sizeof(nm), "%d.conv.%d.weight", mod, sub * 2);
    else
        snprintf(nm, sizeof(nm), "%d.weight", mod);
    const st_tensor_info *info = st_find_tensor(st, nm);
    if (!info || info->dims[0] != C_out || info->dims[1] != C_in) {
        fprintf(stderr, "taesd: missing/mismatched %s (%d x %d)\n",
                nm, C_out, C_in);
        return -1;
    }
    float *raw = (float *)malloc((size_t)C_out * C_in * 9 * sizeof(float));
    if (!raw) return -1;
    if (st_read_tensor_f32(st, info, raw, (int64_t)C_out * C_in * 9) !=
        (int64_t)C_out * C_in * 9) {
        free(raw);
        return -1;
    }
    /* permute [co][ci][kh][kw] -> k-order k = ci*9 + kw*3 + kh, then
     * store as F16 (wtype=1) — the conv GEMM dequants once, and F16
     * halves weight memory/traffic on the CM4. */
    uint16_t *w = (uint16_t *)malloc((size_t)C_out * C_in * 9 * sizeof(uint16_t));
    if (!w) { free(raw); return -1; }
    for (int co = 0; co < C_out; co++)
        for (int ci = 0; ci < C_in; ci++)
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    w[(size_t)co * C_in * 9 + ci * 9 + kw * 3 + kh] =
                        wubu_sd_f32_to_f16(
                            raw[((size_t)co * C_in + ci) * 9 + kh * 3 + kw]);
    free(raw);
    int slot = mod * 3 + sub;
    t->w_f16[slot] = w;
    {
        char bn[256];
        if (is_block)
            snprintf(bn, sizeof(bn), "%d.conv.%d.bias", mod, sub * 2);
        else
            snprintf(bn, sizeof(bn), "%d.bias", mod);
        info = st_find_tensor(st, bn);
        if (info && info->n_elems == C_out) {
            float *b = (float *)malloc((size_t)C_out * sizeof(float));
            if (b && st_read_tensor_f32(st, info, b, C_out) == C_out)
                t->bias[slot] = b;
            else
                free(b);
        }
    }
    t->has[slot] = 1;
    return 0;
}

wubu_sd_taesd_t *wubu_sd_taesd_load(const char *path) {
    st_ctx *st = st_open(path);
    if (!st) { fprintf(stderr, "taesd: cannot open %s\n", path); return NULL; }
    wubu_sd_taesd_t *t = (wubu_sd_taesd_t *)calloc(1, sizeof(*t));
    if (!t) { st_close(st); return NULL; }
    /* single convs (mod, C_in, C_out) */
    const struct { int mod; int cin; int cout; } singles[] = {
        { 1, 4, 64 }, { 7, 64, 64 }, { 12, 64, 64 }, { 17, 64, 64 },
        { 19, 64, 3 },
    };
    for (size_t i = 0; i < sizeof(singles) / sizeof(singles[0]); i++)
        if (load_conv(t, st, singles[i].mod, 0,
                      singles[i].cin, singles[i].cout, 0) != 0) goto fail;
    /* blocks: 3,4,5 / 8,9,10 / 13,14,15 / 18 — each 3 convs 64->64 */
    const int blocks[] = { 3, 4, 5, 8, 9, 10, 13, 14, 15, 18 };
    for (size_t i = 0; i < sizeof(blocks) / sizeof(blocks[0]); i++)
        for (int sub = 0; sub < 3; sub++)
            if (load_conv(t, st, blocks[i], sub, 64, 64, 1) != 0) goto fail;
    st_close(st);
    return t;
fail:
    fprintf(stderr, "taesd: load failed\n");
    wubu_sd_taesd_free(t);
    st_close(st);
    return NULL;
}

/* one conv with a loaded slot; us = upsample factor (2 for up convs) */
static int run_conv(wubu_sd_taesd_t *t, int slot,
                    const float *x, int C_in, int C_out, int H, int W,
                    float *y, int us) {
    if (!t->has[slot]) return -1;
    int Hout = 0, Wout = 0;
    wubu_sd_conv2d_q(x, 1, C_in, H, W, t->w_f16[slot], 1, t->bias[slot],
                     C_out, 3, 3, 1, 1, 1, y, &Hout, &Wout, us);
    return (Hout > 0 && Wout > 0) ? 0 : -1;
}

static void relu_(float *x, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}

/* Block at module m: y = ReLU(conv2(ReLU(conv1(ReLU(conv0(x))))) + x) */
static void taesd_block(wubu_sd_taesd_t *t, int mod,
                        const float *x, int H, int W,
                        float *buf1, float *buf2, float *y) {
    const int n = 64 * H * W;
    run_conv(t, mod * 3 + 0, x, 64, 64, H, W, buf1, 0);
    relu_(buf1, n);
    run_conv(t, mod * 3 + 1, buf1, 64, 64, H, W, buf2, 0);
    relu_(buf2, n);
    run_conv(t, mod * 3 + 2, buf2, 64, 64, H, W, buf1, 0);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float v = buf1[i] + x[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

int wubu_sd_taesd_decode(wubu_sd_taesd_t *t, const float *latent,
                         int H, int W, float *img) {
    if (!t || H <= 0 || W <= 0) return -1;
    const int C = 64;
    int ch = H, cw = W;
    /* buffers grow with each 2x upsample: resize to (8H x 8W) at the end */
    for (int s = 0; s < 3; s++) ch *= 2, cw *= 2;
    float *x  = (float *)malloc((size_t)C * ch * cw * sizeof(float));
    float *b1 = (float *)malloc((size_t)C * ch * cw * sizeof(float));
    float *b2 = (float *)malloc((size_t)C * ch * cw * sizeof(float));
    float *b3 = (float *)malloc((size_t)C * ch * cw * sizeof(float));
    if (!x || !b1 || !b2 || !b3) { free(x); free(b1); free(b2); free(b3); return -1; }
    ch = H; cw = W;
    /* Clamp: tanh(x/3)*3 */
    #pragma omp parallel for
    for (int i = 0; i < 4 * H * W; i++)
        x[i] = tanhf(latent[i] / 3.0f) * 3.0f;
    if (run_conv(t, 3, x, 4, C, H, W, b1, 0) != 0) goto fail;
    relu_(b1, C * H * W);
    /* stage 1: blocks 3,4,5 @ HxW */
    taesd_block(t, 3,  b1, ch, cw, b2, b3, x);  memcpy(b1, x, (size_t)C*ch*cw*4);
    taesd_block(t, 4,  b1, ch, cw, b2, b3, x);  memcpy(b1, x, (size_t)C*ch*cw*4);
    taesd_block(t, 5,  b1, ch, cw, b2, b3, x);  memcpy(b1, x, (size_t)C*ch*cw*4);
    /* upsample -> 2H x 2W + conv(7) */
    ch *= 2; cw *= 2;
    if (run_conv(t, 21, b1, C, C, ch / 2, cw / 2, x, 2) != 0) goto fail;
    memcpy(b1, x, (size_t)C*ch*cw*4);
    /* stage 2: blocks 8,9,10 @ 2H */
    taesd_block(t, 8,  b1, ch, cw, b2, b3, x);  memcpy(b1, x, (size_t)C*ch*cw*4);
    taesd_block(t, 9,  b1, ch, cw, b2, b3, x);  memcpy(b1, x, (size_t)C*ch*cw*4);
    taesd_block(t, 10, b1, ch, cw, b2, b3, x);  memcpy(b1, x, (size_t)C*ch*cw*4);
    /* upsample -> 4H x 4W + conv(12) */
    ch *= 2; cw *= 2;
    if (run_conv(t, 36, b1, C, C, ch / 2, cw / 2, x, 2) != 0) goto fail;
    memcpy(b1, x, (size_t)C*ch*cw*4);
    /* stage 3: blocks 13,14,15 @ 4H */
    taesd_block(t, 13, b1, ch, cw, b2, b3, x);  memcpy(b1, x, (size_t)C*ch*cw*4);
    taesd_block(t, 14, b1, ch, cw, b2, b3, x);  memcpy(b1, x, (size_t)C*ch*cw*4);
    taesd_block(t, 15, b1, ch, cw, b2, b3, x);  memcpy(b1, x, (size_t)C*ch*cw*4);
    /* upsample -> 8H x 8W + conv(17) */
    ch *= 2; cw *= 2;
    if (run_conv(t, 51, b1, C, C, ch / 2, cw / 2, x, 2) != 0) goto fail;
    memcpy(b1, x, (size_t)C*ch*cw*4);
    /* block 18 @ 8H */
    taesd_block(t, 18, b1, ch, cw, b2, b3, x);  memcpy(b1, x, (size_t)C*ch*cw*4);
    /* final conv(64->3) -> img [3][8H][8W] in [0,1] */
    if (run_conv(t, 57, b1, C, 3, ch, cw, img, 0) != 0) goto fail;
    free(x); free(b1); free(b2); free(b3);
    return 0;
fail:
    free(x); free(b1); free(b2); free(b3);
    return -1;
}

void wubu_sd_taesd_free(wubu_sd_taesd_t *t) {
    if (!t) return;
    for (int i = 0; i < 64; i++) {
        free(t->w_f16[i]);
        free(t->bias[i]);
    }
    free(t);
}
