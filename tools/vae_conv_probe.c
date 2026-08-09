/* vae_conv_probe.c -- verify conv2d_q vs canonical for a REAL VAE conv
 * (up.0.block.0.conv1: 256->128, 3x3, at 512x512 would take forever —
 * use a 32x32 crop). Also test conv_in 4->512. */
#define _GNU_SOURCE
#include "wubu_sd_ops.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double gauss(uint64_t *s) { *s = *s*6364136223846793005ULL+1; uint64_t v=*s>>33; return ((double)(v&0xFFFFFF)/16777215.0)*2-1; }

int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g->data_blob) gguf_buffer_data(g);
    const char *names[] = {
        "first_stage_model.decoder.conv_in.weight",   /* 4->512 */
        "first_stage_model.decoder.up.0.block.0.conv1.weight", /* 256->128 */
        "first_stage_model.decoder.up.0.block.0.conv2.weight", /* 128->128 */
        "first_stage_model.decoder.up.1.block.0.conv1.weight", /* 512->256 */
    };
    for (int n = 0; n < 4; n++) {
        gguf_tensor_info *ti = gguf_find_tensor(g, names[n]);
        if (!ti) { printf("%s MISSING\n", names[n]); continue; }
        /* dims: (kh,kw,ci,co) column-major */
        int64_t ne = 1; for (int d=0; d<ti->n_dims; d++) ne *= ti->dims[d];
        int kh = (int)ti->dims[0], kw = (int)ti->dims[1],
            ci = (int)ti->dims[2], co = (int)ti->dims[3];
        printf("%s: kh=%d kw=%d ci=%d co=%d type=%d\n", names[n], kh, kw, ci, co, ti->ggml_type);
        /* reference: dequant canonical, build [co][ci][kh][kw] and run a
         * manual direct conv on a small input, compare vs conv2d_q */
        int H = 16, W = 16;
        float *x = (float *)malloc((size_t)ci * H * W * sizeof(float));
        uint64_t s = 7;
        for (int i = 0; i < ci*H*W; i++) x[i] = (float)gauss(&s);
        /* manual direct conv (true semantics): out[co][oh][ow] =
         * sum_{c,k1,k2} x[c][oh+k1-1][ow+k2-1] * W[k1,k2,c,co] */
        float *ref = (float *)calloc((size_t)co * H * W, sizeof(float));
        float *wf = (float *)malloc((size_t)ne * sizeof(float));
        gguf_read_tensor_f32(g, ti, wf, ne); /* raw layout: (k1,k2,c,co) */
        for (int oc = 0; oc < co; oc++)
            for (int oh = 0; oh < H; oh++)
                for (int ow = 0; ow < W; ow++) {
                    double acc = 0;
                    for (int c = 0; c < ci; c++)
                        for (int k1 = 0; k1 < kh; k1++) {
                            int ih = oh + k1 - 1;
                            if (ih < 0 || ih >= H) continue;
                            for (int k2 = 0; k2 < kw; k2++) {
                                int iw = ow + k2 - 1;
                                if (iw < 0 || iw >= W) continue;
                                float wv = wf[k1 + kh*(k2 + kw*(c + ci*oc))];
                                ref[((size_t)oc*H + oh)*W + ow] += x[((size_t)c*H + ih)*W + iw] * wv;
                            }
                        }
                }
        /* conv2d_q output */
        float *y = (float *)malloc((size_t)co * H * W * sizeof(float));
        float *b = NULL;
        wubu_sd_conv2d_q(x, 1, ci, H, W, (const uint8_t*)g->data_blob + ti->data_offset,
                         ti->ggml_type, b, co, kh, kw, 1, 1, 1, y, NULL, NULL);
        double maxd = 0; int nbad = 0;
        for (int i = 0; i < co*H*W; i++) {
            double d = fabs(y[i] - ref[i]);
            if (d > maxd) maxd = d;
            if (d > 1e-3) nbad++;
        }
        printf("  conv2d_q vs manual: maxdiff=%g nbad=%d %s\n", maxd, nbad,
               (nbad == 0) ? "OK" : "MISMATCH");
        free(x); free(ref); free(wf); free(y);
    }
    return 0;
}
