/* vae64.c -- conv2d_q (conv_in 4->512) at the REAL 64x64 size, vs manual.
 * Tiling T=8 runs 8 tiles — catches stale-xcol bugs the 16x16 test missed. */
#define WUBU_HOSTED
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
    gguf_tensor_info *ti = gguf_find_tensor(g, "first_stage_model.decoder.conv_in.weight");
    int kh=(int)ti->dims[0], kw=(int)ti->dims[1], ci=(int)ti->dims[2], co=(int)ti->dims[3];
    int H=64, W=64;
    float *x = (float *)malloc((size_t)ci*H*W*sizeof(float));
    uint64_t s = 7;
    for (int i = 0; i < ci*H*W; i++) x[i] = (float)gauss(&s);
    int64_t ne = 1; for (int d=0; d<ti->n_dims; d++) ne *= ti->dims[d];
    float *wf = (float *)malloc(ne*4);
    gguf_read_tensor_f32(g, ti, wf, ne);
    float *ref = (float *)calloc((size_t)co*H*W, sizeof(float));
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
                            acc += x[((size_t)c*H+ih)*W+iw] * wf[k1 + kh*(k2 + kw*(c + ci*oc))];
                        }
                    }
                ref[((size_t)oc*H+oh)*W+ow] = (float)acc;
            }
    float *y = (float *)malloc((size_t)co*H*W*sizeof(float));
    wubu_sd_conv2d_q(x, 1, ci, H, W, (const uint8_t*)g->data_blob + ti->data_offset,
                     ti->ggml_type, NULL, co, kh, kw, 1, 1, 1, y, NULL, NULL);
    /* per-tile-column error: check tiles 0 and 7 separately */
    for (int tile = 0; tile < 8; tile += 7) {
        double maxd = 0; int nbad = 0;
        for (int oc = 0; oc < co; oc++)
            for (int r = tile*8; r < tile*8+8; r++)
                for (int c2 = 0; c2 < W; c2++) {
                    double d = fabs(y[((size_t)oc*H+r)*W+c2] - ref[((size_t)oc*H+r)*W+c2]);
                    if (d > maxd) maxd = d;
                    if (d > 1e-3) nbad++;
                }
        printf("tile %d: maxdiff=%g nbad=%d %s\n", tile, maxd, nbad, nbad==0?"OK":"MISMATCH");
    }
    return 0;
}
