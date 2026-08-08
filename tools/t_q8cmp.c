/* t_q8cmp.c — compare quantized_matmul (F32 x) vs quantized_matmul_from_q8
 * (Q8_K x) on the REAL attn_qkv weight. If they disagree, the q8 path that
 * the SSM forward uses is broken.
 * Usage: t_q8cmp <model.gguf> <layer> */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    int layer = argc > 2 ? atoi(argv[2]) : 0;
    gguf_ctx *ctx = gguf_open(argv[1]);
    gguf_buffer_data(ctx);
    char name[128];
    snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", layer);
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    int in = (int)t->dims[0], out = (int)t->dims[1];
    const uint8_t *W = (const uint8_t *)ctx->data_blob + t->data_offset;
    printf("%s: in=%d out=%d type=%d\n", name, in, out, t->ggml_type);

    float *x = malloc((size_t)in * 4);
    for (int i = 0; i < in; i++) x[i] = 0.01f * (i % 7) - 0.02f;

    /* path A: quantized_matmul (F32 x) */
    float *ya = malloc((size_t)out * 4);
    quantized_matmul(x, W, t->ggml_type, in, out, 0, ya);

    /* path B: quantize x to Q8_K then from_q8 */
    int n_q8_blocks = (in + 255) / 256;
    uint8_t *q8x = malloc((size_t)n_q8_blocks * 292);
    quantize_row_q8_K(x, (block_q8_K *)q8x, in);
    float *yb = malloc((size_t)out * 4);
    quantized_matmul_from_q8(q8x, W, t->ggml_type, in, out, 0, yb);

    double dot = 0, n1 = 0, n2 = 0; float max_e = 0;
    for (int j = 0; j < out; j++) {
        dot += (double)ya[j] * yb[j];
        n1  += (double)ya[j] * ya[j];
        n2  += (double)yb[j] * yb[j];
        float e = fabsf(ya[j] - yb[j]);
        if (e > max_e) max_e = e;
    }
    printf("cos-sim = %.8f  max_err = %.6f\n",
           dot / (sqrt(n1) * sqrt(n2)), max_e);
    printf("ya[0..3]=%.4f %.4f %.4f %.4f\n", ya[0], ya[1], ya[2], ya[3]);
    printf("yb[0..3]=%.4f %.4f %.4f %.4f\n", yb[0], yb[1], yb[2], yb[3]);
    return 0;
}
