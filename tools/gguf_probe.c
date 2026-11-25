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

    /* Dequantise a sample of every type present. The port's data path is only real once each type in
     * the file decodes to finite, sane values through this engine's own dequantisers. */
    printf("\n%-4s %-10s %-8s %-12s %s\n", "id", "type", "count", "sample range", "finite");
    int seen[64] = {0}, counts[64] = {0};
    for (int64_t i = 0; i < ctx->n_tensors; ++i)
        if (ctx->tensors[i].ggml_type >= 0 && ctx->tensors[i].ggml_type < 64)
            ++counts[ctx->tensors[i].ggml_type];
    for (int64_t i = 0; i < ctx->n_tensors; ++i) {
        gguf_tensor_info *t = &ctx->tensors[i];
        const int ty = t->ggml_type;
        if (ty < 0 || ty >= 64 || seen[ty]) continue;
        seen[ty] = 1;
        const int64_t n = (int64_t)t->dims[0] < 256 ? (int64_t)t->dims[0] : 256;
        /* Read straight from the file: the 8.5 GB blob is not mapped. */
        uint8_t raw[4096];
        const int64_t nbytes = gguf_raw_size(ty, n);
        if (nbytes <= 0 || nbytes > (int64_t)sizeof raw) continue;
        /* 64-bit seek: this file is 8.5 GB and the tensors past 2 GB would otherwise be read from
         * the wrong place, which looks exactly like a dequant bug (Q4_K came out at 7e6). */
        if (_fseeki64(ctx->file, (long long)(ctx->data_blob_offset + t->data_offset), SEEK_SET) != 0) continue;
        if (fread(raw, 1, (size_t)nbytes, ctx->file) != (size_t)nbytes) continue;
        float buf[256];
        gguf_dequantize(raw, ty, n, buf);
        float lo = 1e30f, hi = -1e30f; int bad = 0;
        for (int64_t k = 0; k < n; ++k) {
            if (!(buf[k] == buf[k]) || buf[k] > 3e38f || buf[k] < -3e38f) ++bad;
            if (buf[k] < lo) lo = buf[k];
            if (buf[k] > hi) hi = buf[k];
        }
        printf("%-4d %-10s %-8d [%+.5g, %+.5g]  %s\n", ty, "", counts[ty], lo, hi, bad ? "NAN/INF" : "ok");
    }

    gguf_close(ctx);
    return miss == 0 ? 0 : 3;
}
