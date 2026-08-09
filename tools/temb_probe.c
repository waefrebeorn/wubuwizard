/* temb_probe.c -- dump time embedding for t=0 vs t=999 */
#define _GNU_SOURCE
#include "wubu_sd_unet.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* opaque struct access — peek via the load ctx instead */
extern void wubu_sd_linear_qb(const float *x, const void *W, int type,
                              const float *b, int M, int K, int N, float *y);
static void temb_probe(gguf_ctx *g, int t, float *out) {
    #define U_CH 320
    #define U_TIME 1280
    float emb[U_CH];
    const int half = U_CH / 2;
    for (int i = 0; i < half; i++) {
        float w = expf(-logf(10000.0f) * (float)i / (float)half);
        emb[i] = sinf((float)t * w);
        emb[i + half] = cosf((float)t * w);
    }
    gguf_tensor_info *ti;
    ti = gguf_find_tensor(g, "model.diffusion_model.time_embed.0.weight");
    if (!ti) { fprintf(stderr, "missing time_embed.0.weight\n"); return; }
    const uint8_t *w1 = (const uint8_t *)g->data_blob + ti->data_offset;
    ti = gguf_find_tensor(g, "model.diffusion_model.time_embed.0.bias");
    float *b1 = (float *)malloc(ti->dims[0] * 4);
    gguf_read_tensor_f32(g, ti, b1, ti->dims[0]);
    ti = gguf_find_tensor(g, "model.diffusion_model.time_embed.2.weight");
    const uint8_t *w2 = (const uint8_t *)g->data_blob + ti->data_offset;
    ti = gguf_find_tensor(g, "model.diffusion_model.time_embed.2.bias");
    float *b2 = (float *)malloc(ti->dims[0] * 4);
    gguf_read_tensor_f32(g, ti, b2, ti->dims[0]);
    float mid[U_TIME];
    wubu_sd_linear_qb(emb, w1, 1, b1, 1, U_CH, U_TIME, mid);
    for (int i = 0; i < U_TIME; i++) mid[i] = mid[i] / (1.0f + expf(-mid[i]));
    wubu_sd_linear_qb(mid, w2, 1, b2, 1, U_TIME, U_TIME, out);
    free(b1); free(b2);
}
int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g->data_blob) gguf_buffer_data(g);
    float a[1280], b[1280];
    temb_probe(g, 0, a);
    temb_probe(g, 999, b);
    float maxdiff = 0;
    for (int i = 0; i < 1280; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > maxdiff) maxdiff = d;
    }
    printf("temb t=0:   first8: %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f\n",
           a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
    printf("temb t=999: first8: %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f\n",
           b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]);
    printf("max|t0 - t999| = %f  (must be LARGE if time path works)\n", maxdiff);
    return 0;
}
