/* t_offsets.c — print GGUF tensor names + our reader's data_offsets. */
#include "gguf_reader.h"
#include <stdio.h>

int main(int argc, char **argv) {
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    for (int64_t i = 0; i < ctx->n_tensors && i < 12; i++) {
        const gguf_tensor_info *t = &ctx->tensors[i];
        printf("%-40s type=%d off=%llu\n", t->name, t->ggml_type,
               (unsigned long long)t->data_offset);
    }
    /* also find ffn_gate directly */
    gguf_tensor_info *tg = gguf_find_tensor(ctx, "blk.0.ffn_gate.weight");
    if (tg) printf("ffn_gate off=%llu type=%d\n",
                   (unsigned long long)tg->data_offset, tg->ggml_type);
    return 0;
}
