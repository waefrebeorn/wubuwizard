/* wubu_inkling2_meta.c -- Inkling-2 GGUF probe and tensor map. Strict C11.
 *
 * Step 1 of the six in docs/research/inkling2-gguf-inventory.md: no forward pass
 * yet, just the metadata contract and the tensor slots the forward will need,
 * validated against the file instead of assumed.
 */
#include "wubu_inkling2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The gguf_kv_get_* helpers return 1 on success and 0 when the key is absent
 * (or has the wrong type) - a detail worth stating because it is inverted
 * relative to the file-level gguf_read_* conventions. */
static int kv_i32(gguf_ctx *ctx, const char *key, int *out)
{
    return gguf_kv_get_i32(ctx, key, out) != 0;
}

static int kv_f32(gguf_ctx *ctx, const char *key, float *out)
{
    return gguf_kv_get_f32(ctx, key, out) != 0;
}

int wubu_inkling2_probe(gguf_ctx *ctx, wubu_inkling2_cfg_t *cfg)
{
    if (!ctx || !cfg) return WUBU_INKLING2_ERR_ARCH;
    memset(cfg, 0, sizeof(*cfg));

    char arch[64] = {0};
    if (gguf_kv_get_str(ctx, "general.architecture", arch, sizeof(arch)) == 0) return WUBU_INKLING2_ERR_ARCH;
    if (strcmp(arch, "inkling2") != 0) return WUBU_INKLING2_ERR_ARCH;

    /* Every key below is required: a missing one is a load failure, not a default. */
    if (!kv_i32(ctx, "inkling2.block_count", &cfg->n_layer) ||
        !kv_i32(ctx, "inkling2.embedding_length", &cfg->d_model) ||
        !kv_i32(ctx, "inkling2.feed_forward_length", &cfg->d_ff) ||
        !kv_i32(ctx, "inkling2.expert_count", &cfg->n_expert) ||
        !kv_i32(ctx, "inkling2.expert_used_count", &cfg->n_expert_used) ||
        !kv_i32(ctx, "inkling2.expert_feed_forward_length", &cfg->d_ff_expert) ||
        !kv_i32(ctx, "inkling2.attention.key_length", &cfg->head_dim) ||
        !kv_i32(ctx, "inkling2.num_loops", &cfg->n_loops) ||
        !kv_i32(ctx, "inkling2.loop.core_start", &cfg->loop_core_start) ||
        !kv_i32(ctx, "inkling2.loop.core_end", &cfg->loop_core_end) ||
        !kv_i32(ctx, "inkling2.loop.gate_rank", &cfg->loop_gate_rank) ||
        !kv_i32(ctx, "inkling2.attention.sliding_window", &cfg->swa_window) ||
        !kv_i32(ctx, "inkling2.shortconv.l_cache", &cfg->conv_lcache) ||
        !kv_i32(ctx, "inkling2.attn_res.block_size", &cfg->attn_res_block) ||
        !kv_i32(ctx, "inkling2.vocab_size", &cfg->vocab_size) ||
        !kv_i32(ctx, "inkling2.vocab_size_unpadded", &cfg->vocab_size_unpadded) ||
        !kv_i32(ctx, "inkling2.rope.dimension_count", &cfg->rope_dim))
        return WUBU_INKLING2_ERR_MISSING;

    if (!kv_f32(ctx, "inkling2.attention.layer_norm_rms_epsilon", &cfg->rms_eps) ||
        !kv_f32(ctx, "inkling2.expert_weights_scale", &cfg->expert_weights_scale) ||
        !kv_f32(ctx, "inkling2.activation.situ_beta", &cfg->situ_beta) ||
        !kv_f32(ctx, "inkling2.activation.situ_linear_beta", &cfg->situ_linear_beta) ||
        !kv_f32(ctx, "inkling2.final_logit_softcapping", &cfg->logit_softcap))
        return WUBU_INKLING2_ERR_MISSING;

    /* Inkling has no RoPE; a non-zero value here means the file is a different
     * architecture generation and the relative-bias path would be wrong. */
    if (cfg->rope_dim != 0) return WUBU_INKLING2_ERR_SHAPE;

    if (cfg->n_layer <= 0 || cfg->d_model <= 0 || cfg->head_dim <= 0) return WUBU_INKLING2_ERR_SHAPE;
    /* Head counts come from the file when published (they are not derivable from
     * d_model: q is 3072 wide for a 2304-wide model = 24 heads of 128). */
    if (!kv_i32(ctx, "inkling2.attention.head_count", &cfg->n_head) || cfg->n_head <= 0) {
        int64_t qw = 0;
        gguf_tensor_info *tq = gguf_find_tensor(ctx, "blk.0.attn_q.weight");
        if (!tq || tq->n_dims != 2) return WUBU_INKLING2_ERR_SHAPE;
        qw = tq->dims[1];
        cfg->n_head = (int)(qw / cfg->head_dim);
    }
    if (!kv_i32(ctx, "inkling2.attention.head_count_kv", &cfg->n_kv_head) || cfg->n_kv_head <= 0) {
        gguf_tensor_info *tk = gguf_find_tensor(ctx, "blk.0.attn_k.weight");
        if (!tk || tk->n_dims != 2) return WUBU_INKLING2_ERR_SHAPE;
        cfg->n_kv_head = (int)(tk->dims[1] / cfg->head_dim);
    }
    if (cfg->n_head <= 0 || cfg->n_kv_head <= 0) return WUBU_INKLING2_ERR_SHAPE;

    if (cfg->loop_core_end <= cfg->loop_core_start || cfg->n_loops < 1)
        return WUBU_INKLING2_ERR_SHAPE;

    /* The core layers are stored once and run n_loops times, so the number of
     * physical blocks is smaller than block_count: 48 logical = 24 stored. */
    const int core_len = cfg->loop_core_end - cfg->loop_core_start;
    cfg->n_layer_phys = cfg->n_layer - (cfg->n_loops - 1) * core_len;
    if (cfg->n_layer_phys <= 0 || cfg->n_layer_phys > cfg->n_layer) return WUBU_INKLING2_ERR_SHAPE;

    return WUBU_INKLING2_OK;
}

int wubu_inkling2_phys_block(const wubu_inkling2_cfg_t *cfg, int il)
{
    if (!cfg || il < 0 || il >= cfg->n_layer) return -1;
    const int core_len = cfg->loop_core_end - cfg->loop_core_start;
    if (il < cfg->loop_core_start) return il;
    const int after = il - cfg->loop_core_start;
    if (after < core_len * cfg->n_loops) return cfg->loop_core_start + (after % core_len);
    return cfg->loop_core_end + (after - core_len * cfg->n_loops);
}

int wubu_inkling2_layer_tensor(gguf_ctx *ctx, int il, const char *family, char *buf, size_t cap)
{
    if (!ctx || !family || !buf || cap == 0) return 0;
    snprintf(buf, cap, "blk.%d.%s", il, family);
    return gguf_find_tensor(ctx, buf) ? 1 : 0;
}

int wubu_inkling2_global_tensor(gguf_ctx *ctx, const char *name, gguf_tensor_info **out)
{
    if (!ctx || !name || !out) return 0;
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) return 0;
    *out = t;
    return 1;
}

/* dims[0] is the innermost (row) dimension in GGUF, so the projections are
 * stored as (out_features, in_features): attn_q is (d_model, 2*d_model). */
/* Quant types present in Inkling GGUFs that this engine cannot decode yet. The
 * ids come from ggml's enum; 30 is BF16 in upstream ggml, 40/42 are the newer
 * 4-bit float family (project-maya documents NVFP4/MXFP4 for the same model).
 * Everything else in the file already has a dequantiser. */
static int inkling2_type_supported(int ggml_type)
{
    switch (ggml_type) {
    case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16:
    case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0: case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
    case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_IQ4_NL:
    /* 4-bit-float / 2-bit microscaling: NVFP4 (40) and Q2_0 (42) are what this
     * file's routed experts use; MXFP4 (39) is the same family. Decoders in
     * src/wubu_dequant_fp4.c, gated by test_eg119_fp4_upstream. */
    case 39: case 40: case 42:
        return 1;
    default:
        return 0;
    }
}

int wubu_inkling2_tensor_f32(gguf_ctx *ctx, const char *name, float *out,
                             int64_t max_elems, int64_t *n_out)
{
    if (!ctx || !name || !out) return WUBU_INKLING2_ERR_IO;
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) return WUBU_INKLING2_ERR_MISSING_TENSOR;
    if (!inkling2_type_supported(t->ggml_type)) return WUBU_INKLING2_ERR_UNKNOWN_TYPE;

    int64_t n = 1;
    for (int i = 0; i < t->n_dims; ++i) n *= t->dims[i];
    if (n <= 0 || n > max_elems) return WUBU_INKLING2_ERR_SHAPE;
    if (n_out) *n_out = n;

    const uint8_t *raw = NULL;
    uint8_t *owned = NULL;
    if (ctx->data_blob) {
        raw = (const uint8_t *)ctx->data_blob + t->data_offset;
    } else {
        const int64_t bytes = gguf_raw_size(t->ggml_type, n);
        if (bytes <= 0) return WUBU_INKLING2_ERR_UNKNOWN_TYPE;
        owned = (uint8_t *)malloc((size_t)bytes);
        if (!owned) return WUBU_INKLING2_ERR_IO;
        if (!gguf_read_raw_tensor(ctx, t, owned)) { free(owned); return WUBU_INKLING2_ERR_IO; }
        raw = owned;
    }
    gguf_dequantize(raw, t->ggml_type, n, out);
    free(owned);
    return 0;
}

static int check2(gguf_ctx *ctx, int il, const char *family, int64_t d0, int64_t d1,
                  int *problems, int *shown)
{
    char name[256];
    if (!wubu_inkling2_layer_tensor(ctx, il, family, name, sizeof(name))) {
        if (*problems < 8 && shown) { fprintf(stderr, "[INKLING2] missing %s\n", name); ++*shown; }
        ++*problems;
        return 0;
    }
    const gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (t->n_dims != 2 || t->dims[0] != d0 || t->dims[1] != d1) {
        if (*problems < 8 && shown) {
            fprintf(stderr, "[INKLING2] %s dims %lldx%lld, want %lldx%lld\n", name,
                    (long long)t->dims[0], (long long)t->dims[1], (long long)d0, (long long)d1);
            ++*shown;
        }
        ++*problems;
        return 0;
    }
    return 1;
}

static int check3(gguf_ctx *ctx, int il, const char *family,
                  int64_t d0, int64_t d1, int64_t d2, int *problems, int *shown)
{
    char name[256];
    if (!wubu_inkling2_layer_tensor(ctx, il, family, name, sizeof(name))) {
        if (*problems < 8 && shown) { fprintf(stderr, "[INKLING2] missing %s\n", name); ++*shown; }
        ++*problems;
        return 0;
    }
    const gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (t->n_dims != 3 || t->dims[0] != d0 || t->dims[1] != d1 || t->dims[2] != d2) {
        if (*problems < 8 && shown) {
            fprintf(stderr, "[INKLING2] %s dims %lldx%lldx%lld, want %lldx%lldx%lld\n", name,
                    (long long)t->dims[0], (long long)t->dims[1], (long long)t->dims[2],
                    (long long)d0, (long long)d1, (long long)d2);
            ++*shown;
        }
        ++*problems;
        return 0;
    }
    return 1;
}

int wubu_inkling2_validate(gguf_ctx *ctx, const wubu_inkling2_cfg_t *cfg)
{
    if (!ctx || !cfg) return 1;
    int problems = 0, shown = 0;
    const int64_t D = cfg->d_model;
    const int64_t QD = (int64_t)cfg->n_head * cfg->head_dim;   /* 3072 for the 2304 model */
    const int64_t KVD = (int64_t)cfg->n_kv_head * cfg->head_dim;

    /* Iterate the STORED blocks; the loop expansion happens in the forward. */
    for (int il = 0; il < cfg->n_layer_phys; ++il) {
        check2(ctx, il, "attn_q.weight", D, QD, &problems, &shown);
        check2(ctx, il, "attn_k.weight", D, KVD, &problems, &shown);
        check2(ctx, il, "attn_v.weight", D, KVD, &problems, &shown);
        check2(ctx, il, "attn_output.weight", QD, D, &problems, &shown);
        check2(ctx, il, "attn_gate.weight", D, QD, &problems, &shown);
        check2(ctx, il, "attn_rel.weight", D, (int64_t)cfg->head_dim * 3, &problems, &shown);
        check2(ctx, il, "attn_rel_bias.weight", 16, cfg->swa_window, &problems, &shown);
        check2(ctx, il, "attn_sink_proj.weight", D, cfg->n_head, &problems, &shown);
        check2(ctx, il, "attn_k_conv.weight", cfg->conv_lcache, KVD, &problems, &shown);
        check2(ctx, il, "attn_v_conv.weight", cfg->conv_lcache, KVD, &problems, &shown);
        check2(ctx, il, "ffn_gate_inp.weight", D, cfg->n_expert, &problems, &shown);
        /* Routed experts are 3-D: [rows, cols, n_expert]. */
        check3(ctx, il, "ffn_down_exps.weight", cfg->d_ff_expert, D, cfg->n_expert,
               &problems, &shown);
        check3(ctx, il, "ffn_gate_up_exps.weight", D, 2 * (int64_t)cfg->d_ff_expert, cfg->n_expert,
               &problems, &shown);

        /* Loop gates exist only on the looped core. */
        if (il >= cfg->loop_core_start && il < cfg->loop_core_end) {
            check2(ctx, il, "loop_k_gain.weight", cfg->loop_gate_rank, D, &problems, &shown);
            check2(ctx, il, "loop_k_forget.weight", cfg->loop_gate_rank, D, &problems, &shown);
            check2(ctx, il, "loop_k_down.weight", 2 * D, cfg->loop_gate_rank, &problems, &shown);
        }
    }

    gguf_tensor_info *t = NULL;
    if (!wubu_inkling2_global_tensor(ctx, "token_embd.weight", &t) ||
        t->dims[0] != D || t->dims[1] != cfg->vocab_size) {
        fprintf(stderr, "[INKLING2] token_embd.weight missing or wrong dims\n");
        ++problems;
    }
    if (!wubu_inkling2_global_tensor(ctx, "output.weight", &t) ||
        t->dims[0] != D || t->dims[1] != cfg->vocab_size) {
        fprintf(stderr, "[INKLING2] output.weight missing or wrong dims\n");
        ++problems;
    }

    if (problems) fprintf(stderr, "[INKLING2] %d tensor problems (%d shown)\n", problems, shown);
    return problems;
}
