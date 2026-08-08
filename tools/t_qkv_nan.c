/* t_qkv_nan.c — isolate: does quantized_matmul produce NaN on
 * Qwen3.5-0.8B's real attn_qkv (Q8_0, [6144,1024])?
 * Usage: t_qkv_nan <model.gguf> [layer] */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [layer]\n", argv[0]); return 2; }
    int layer = argc > 2 ? atoi(argv[2]) : 0;
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "open failed\n"); return 1; }
    gguf_buffer_data(ctx); const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    if (!blob) { fprintf(stderr, "no blob\n"); return 1; }

    /* find blk.<layer>.attn_qkv.weight */
    char name[128];
    snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", layer);
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "tensor %s not found\n", name); return 1; }
    int64_t n_elems = t->dims[0] * (t->n_dims > 1 ? t->dims[1] : 1);
    fprintf(stderr, "%s: dims=[%lld,%lld] type=%d n_elems=%lld\n",
            name, (long long)t->dims[0], (long long)t->dims[1],
            t->ggml_type, (long long)n_elems);

    int in = (int)t->dims[0], out = (int)t->dims[1];
    float *x = malloc((size_t)in * sizeof(float));
    for (int i = 0; i < in; i++) x[i] = 0.01f; /* small, real-ish */
    float *y = malloc((size_t)out * sizeof(float));
    memset(y, 0, (size_t)out * sizeof(float));

    quantized_matmul(x, blob + t->data_offset, t->ggml_type,
                     in, out, 0, y);

    int nnan = 0, ninf = 0; float maxv = 0, minv = 0;
    for (int i = 0; i < out; i++) {
        if (isnan(y[i])) nnan++;
        else if (isinf(y[i])) ninf++;
        else { if (y[i] > maxv) maxv = y[i]; if (y[i] < minv) minv = y[i]; }
    }
    printf("layer %d %s: out=%d nan=%d inf=%d min=%.6g max=%.6g\n",
           layer, name, out, nnan, ninf, minv, maxv);
    printf("  y[0..4] = %.6g %.6g %.6g %.6g %.6g\n", y[0], y[1], y[2], y[3], y[4]);
    return (nnan || ninf) ? 1 : 0;
}
