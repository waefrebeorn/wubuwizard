/* l1_probe.c -- compare sd_dequant (via linear_q) vs gguf_read_tensor_f32
 * for time_embed.0.weight: if they differ, the dequant path is broken. */
#define _GNU_SOURCE
#include "gguf_reader.h"
#include "wubu_sd_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g->data_blob) gguf_buffer_data(g);
    gguf_tensor_info *ti = gguf_find_tensor(g, "model.diffusion_model.time_embed.0.weight");
    if (!ti) { fprintf(stderr, "missing\n"); return 1; }
    int64_t ne = 1; for (int d = 0; d < ti->n_dims; d++) ne *= ti->dims[d];
    printf("type=%d dims=%d ne=%lld\n", ti->ggml_type, ti->n_dims, (long long)ne);
    /* canonical dequant */
    float *ref = (float *)malloc(ne * 4);
    gguf_read_tensor_f32(g, ti, ref, ne);
    /* raw blob */
    const uint8_t *raw = (const uint8_t *)g->data_blob + ti->data_offset;
    /* feed emb = ones, compare linear_q output vs manual ref matmul */
    int K = (int)ti->dims[0], N = (int)ti->dims[1];
    printf("K=%d N=%d\n", K, N);
    float *x = (float *)malloc(K * sizeof(float));
    for (int i = 0; i < K; i++) x[i] = 1.0f;
    float *yq = (float *)malloc(N * sizeof(float));
    float *yr = (float *)malloc(N * sizeof(float));
    wubu_sd_linear_q(x, raw, 1, 1, K, N, yq);
    for (int j = 0; j < N; j++) {
        double s = 0;
        for (int k = 0; k < K; k++) s += x[k] * ref[(size_t)j * K + k];
        yr[j] = (float)s;
    }
    float maxd = 0; int worst = -1;
    for (int j = 0; j < N; j++) {
        float d = fabsf(yq[j] - yr[j]);
        if (d > maxd) { maxd = d; worst = j; }
    }
    printf("linear_q vs canonical ref: maxdiff=%g (worst j=%d)\n", maxd, worst);
    printf("yq[0..3]=%g %g %g %g  yr[0..3]=%g %g %g %g\n",
           yq[0],yq[1],yq[2],yq[3], yr[0],yr[1],yr[2],yr[3]);
    /* also print ref weight stats: is the canonical data itself tiny? */
    double wmean = 0, wvar = 0;
    for (int i = 0; i < ne; i++) { wmean += ref[i]; wvar += ref[i]*ref[i]; }
    wmean /= ne; wvar = wvar/ne - wmean*wmean;
    printf("ref weight: mean=%g rms=%g\n", wmean, sqrt(wvar));
    return 0;
}
