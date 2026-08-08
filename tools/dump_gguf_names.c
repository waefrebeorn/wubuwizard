/* dump_gguf_names.c — list all tensor names in a GGUF (for
 * loader name-mapping work). */
#include "gguf_reader.h"
#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 2; }
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "open failed\n"); return 1; }
    printf("tensors: %lld\n", (long long)ctx->n_tensors);
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        const gguf_tensor_info *t = &ctx->tensors[i];
        int64_t ne = 1;
        for (int di = 0; di < t->n_dims; di++) ne *= t->dims[di];
        printf("%s  type=%d dims=%d ne=%lld\n", t->name, t->ggml_type,
               t->n_dims, (long long)ne);
    }
    gguf_close(ctx);
    return 0;
}
