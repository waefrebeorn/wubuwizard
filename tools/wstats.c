/* wstats.c -- dump weight statistics for time_embed + emb_layers tensors */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void stats(gguf_ctx *g, const char *name) {
    gguf_tensor_info *ti = gguf_find_tensor(g, name);
    if (!ti) { printf("%s: MISSING\n", name); return; }
    int64_t ne = 1;
    for (int d = 0; d < ti->n_dims; d++) ne *= ti->dims[d];
    float *f = (float *)malloc(ne * 4);
    gguf_read_tensor_f32(g, ti, f, ne);
    double sum = 0, sumsq = 0, mx = 0;
    int nz = 0;
    for (int64_t i = 0; i < ne; i++) {
        double v = f[i];
        sum += v; sumsq += v * v;
        if (fabs(v) > mx) mx = fabs(v);
        if (v != 0) nz++;
    }
    double mean = sum / ne, rms = sqrt(sumsq / ne);
    /* count how many rows are all-zero */
    printf("%s: type=%d dims=%d,%d ne=%lld mean=%g rms=%g maxabs=%g nonzero=%d/%lld\n",
           name, ti->ggml_type, (int)ti->dims[0], (int)ti->dims[1],
           (long long)ne, mean, rms, mx, nz, (long long)ne);
    free(f);
}
int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g->data_blob) gguf_buffer_data(g);
    stats(g, "model.diffusion_model.time_embed.0.weight");
    stats(g, "model.diffusion_model.time_embed.2.weight");
    stats(g, "model.diffusion_model.input_blocks.1.0.emb_layers.1.weight");
    stats(g, "model.diffusion_model.input_blocks.1.0.in_layers.2.weight");
    return 0;
}
