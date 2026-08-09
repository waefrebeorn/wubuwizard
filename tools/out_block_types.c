/* out_block_types.c -- dump types of the out.* tensors in the UNet GGUF */
#define _GNU_SOURCE
#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
    if (argc < 2) return 1;
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g) return 1;
    const char *names[] = {
        "model.diffusion_model.out.0.weight",
        "model.diffusion_model.out.0.bias",
        "model.diffusion_model.out.2.weight",
        "model.diffusion_model.out.2.bias",
        "model.diffusion_model.input_blocks.1.0.in_layers.0.weight",
        NULL
    };
    for (int i = 0; names[i]; i++) {
        gguf_tensor_info *ti = gguf_find_tensor(g, names[i]);
        if (!ti) { printf("%s: NOT FOUND\n", names[i]); continue; }
        int64_t ne = 1;
        for (int d = 0; d < ti->n_dims; d++) ne *= ti->dims[d];
        printf("%s: type=%d ne=%lld dims=%lld,%lld\n", names[i], ti->ggml_type,
               (long long)ne, (long long)ti->dims[0],
               ti->n_dims > 1 ? (long long)ti->dims[1] : 1);
    }
    return 0;
}
