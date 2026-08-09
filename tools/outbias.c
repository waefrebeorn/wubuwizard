/* outbias.c -- check out.2 conv bias + groupnorm output stats. If the
 * groupnorm output is near-zero, eps DC = out.2 bias directly. */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g->data_blob) gguf_buffer_data(g);
    gguf_tensor_info *ti = gguf_find_tensor(g, "model.diffusion_model.out.2.bias");
    float *b = (float *)malloc(4 * sizeof(float));
    gguf_read_tensor_f32(g, ti, b, 4);
    printf("out.2.bias: %+.5f %+.5f %+.5f %+.5f\n", b[0], b[1], b[2], b[3]);
    ti = gguf_find_tensor(g, "model.diffusion_model.out.0.bias");
    float *gb = (float *)malloc(320 * sizeof(float));
    gguf_read_tensor_f32(g, ti, gb, 320);
    double m = 0;
    for (int i = 0; i < 320; i++) m += gb[i];
    printf("out.0 (gn) bias mean=%+.5f\n", m/320);
    return 0;
}
