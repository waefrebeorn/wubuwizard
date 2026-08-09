/* temb_rms.c -- full time-embed diagnostic: output rms + diff for t values */
#define _GNU_SOURCE
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void wubu_sd_linear_qb(const float *x, const void *W, int type,
                              const float *b, int M, int K, int N, float *y);
static void temb_full(gguf_ctx *g, int t, float *out) {
    enum { U_CH = 320, U_TIME = 1280 };
    float emb[U_CH];
    const int half = U_CH / 2;
    for (int i = 0; i < half; i++) {
        float w = expf(-logf(10000.0f) * (float)i / (float)half);
        emb[i] = sinf((float)t * w);
        emb[i + half] = cosf((float)t * w);
    }
    gguf_tensor_info *ti;
    ti = gguf_find_tensor(g, "model.diffusion_model.time_embed.0.weight");
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
    float a[1280], b[1280], c[1280];
    temb_full(g, 0, a);
    temb_full(g, 500, b);
    temb_full(g, 999, c);
    double ma=0, mb=0, mc=0, dab=0, dac=0;
    for (int i = 0; i < 1280; i++) {
        ma += a[i]*a[i]; mb += b[i]*b[i]; mc += c[i]*c[i];
        dab += (a[i]-b[i])*(a[i]-b[i]); dac += (a[i]-c[i])*(a[i]-c[i]);
    }
    ma = sqrt(ma/1280); mb = sqrt(mb/1280); mc = sqrt(mc/1280);
    dab = sqrt(dab/1280); dac = sqrt(dac/1280);
    printf("temb rms: t=0 %g  t=500 %g  t=999 %g\n", ma, mb, mc);
    printf("diff rms: |0-500| %g  |0-999| %g\n", dab, dac);
    printf("=> time signal is %s of output scale\n", (dac > 0.5*ma) ? "LARGE (path OK)" : "TINY (path DEAD)");
    return 0;
}
