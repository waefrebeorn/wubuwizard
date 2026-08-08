/* t_deq_compare.c — dequantize the first N floats of a GGUF tensor with
 * OUR gguf_dequantize and dump them for comparison against llama.cpp. */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <gguf> <tensor> <n>\n", argv[0]); return 1; }
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    gguf_tensor_info *t = gguf_find_tensor(ctx, argv[2]);
    if (!t) { fprintf(stderr, "tensor not found\n"); return 1; }
    int64_t ne = 1;
    for (int d = 0; d < t->n_dims; d++) ne *= t->dims[d];
    int n = atoi(argv[3]);
    if (n > ne) n = (int)ne;
    printf("tensor=%s type=%d ne=%lld n=%d\n", t->name, t->ggml_type,
           (long long)ne, n);
    float *f = (float *)malloc((size_t)n * sizeof(float));
    const uint8_t *data = (const uint8_t *)ctx->data_blob + t->data_offset;
    gguf_dequantize(data, t->ggml_type, n, f);
    for (int i = 0; i < n; i++) printf("%.9g\n", f[i]);
    return 0;
}
