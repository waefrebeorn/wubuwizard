/* gguf_probe -- open any GGUF with the engine's own reader and report what it finds.
 *
 * Written for the Inkling 2 Mini port: before a model family can be implemented, the reader has to be
 * shown to open that architecture, and the tensor names/dims it reports have to match the file. This
 * prints the arch, the tensor count, and resolves a handful of known names so a mismatch shows up as a
 * clear MISS rather than as a silent zero tensor later. */
#include <stdio.h>
#include <string.h>
#include "gguf_reader.h"

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: gguf_probe <model.gguf>\n"); return 2; }

    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "gguf_open failed\n"); return 1; }

    char arch[64] = {0};
    if (gguf_kv_get_str(ctx, "general.architecture", arch, sizeof arch) != 0) strcpy(arch, "(none)");
    char name[128] = {0};
    if (gguf_kv_get_str(ctx, "general.name", name, sizeof name) != 0) strcpy(name, "(none)");

    int n_blocks = -1, n_embd = -1, n_experts = -1, n_used = -1, n_loops = -1;
    gguf_kv_get_i32(ctx, "inkling2.block_count", &n_blocks);
    gguf_kv_get_i32(ctx, "inkling2.embedding_length", &n_embd);
    gguf_kv_get_i32(ctx, "inkling2.expert_count", &n_experts);
    gguf_kv_get_i32(ctx, "inkling2.expert_used_count", &n_used);
    gguf_kv_get_i32(ctx, "inkling2.num_loops", &n_loops);

    printf("arch        : %s\n", arch);
    printf("name        : %s\n", name);
    printf("block_count : %d   embd: %d   experts: %d   used: %d   loops: %d\n",
           n_blocks, n_embd, n_experts, n_used, n_loops);
    printf("tokens      : %lld\n", (long long)gguf_tokenizer_token_count(ctx));

    static const char *probe[] = {
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_rel_bias.weight",
        "blk.0.ffn_gate_up_exps.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.attn_k_conv.weight",
        "blk.6.loop_attn_down.weight",
        "blk.6.loop_mlp_gain.weight",
        "blk.17.loop_v_forget.weight",
        "blk.23.attn_k_conv_proj.weight",
        "output.weight",
        "output_res_query.weight",
    };
    int miss = 0;
    printf("\n%-34s %-8s %s\n", "tensor", "type", "dims");
    for (unsigned i = 0; i < sizeof probe / sizeof probe[0]; ++i) {
        gguf_tensor_info *t = gguf_find_tensor(ctx, probe[i]);
        if (!t) { printf("%-34s MISS\n", probe[i]); ++miss; continue; }
        printf("%-34s %-8d", probe[i], t->ggml_type);
        int64_t n_elems = 1;
        for (int d = 0; d < t->n_dims; ++d) { printf(" %lld", (long long)t->dims[d]); n_elems *= t->dims[d]; }
        printf("   (%lld bytes)\n", (long long)gguf_raw_size(t->ggml_type, n_elems));
    }
    printf("\n%d of %u probes missing\n", miss, (unsigned)(sizeof probe / sizeof probe[0]));

    gguf_close(ctx);
    return miss == 0 ? 0 : 3;
}
