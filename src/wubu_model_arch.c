/* wubu_model_arch.c — architecture registry and GGUF geometry probe.
 *
 * Model dispatch follows WuBuOS's manifest/capability pattern: a registry
 * maps metadata names to a kernel family, and a probe produces a concrete
 * capability/geometry record. Unknown models are described generically
 * instead of being mis-routed to a hard-coded model implementation.
 */
#include "wubu_model_arch.h"
#include <string.h>

static const wubu_arch_entry_t g_arch_registry[] = {
    { "llama",       WUBU_ARCH_LLAMA,     WUBU_FAMILY_TRANSFORMER, WUBU_LAYOUT_BLK,          true,  true,  false },
    { "mistral",     WUBU_ARCH_MISTRAL,   WUBU_FAMILY_TRANSFORMER, WUBU_LAYOUT_BLK,          true,  true,  true  },
    { "gemma",       WUBU_ARCH_GEMMA,     WUBU_FAMILY_TRANSFORMER, WUBU_LAYOUT_BLK,          true,  true,  false },
    { "gemma2",      WUBU_ARCH_GEMMA,     WUBU_FAMILY_TRANSFORMER, WUBU_LAYOUT_BLK,          true,  true,  true  },
    { "gemma3",      WUBU_ARCH_GEMMA,     WUBU_FAMILY_TRANSFORMER, WUBU_LAYOUT_BLK,          true,  true,  true  },
    { "gemma3_text", WUBU_ARCH_GEMMA,     WUBU_FAMILY_TRANSFORMER, WUBU_LAYOUT_BLK,          true,  true,  true  },
    { "gemma4",      WUBU_ARCH_GEMMA4,    WUBU_FAMILY_TRANSFORMER, WUBU_LAYOUT_BLK,          true,  true,  true  },
    { "gemma4_text", WUBU_ARCH_GEMMA4,    WUBU_FAMILY_TRANSFORMER, WUBU_LAYOUT_BLK,          true,  true,  true  },
    { "qwen2",       WUBU_ARCH_QWEN,      WUBU_FAMILY_TRANSFORMER, WUBU_LAYOUT_BLK,          true,  true,  false },
    { "qwen3",       WUBU_ARCH_QWEN,      WUBU_FAMILY_TRANSFORMER, WUBU_LAYOUT_BLK,          true,  true,  false },
    { "qwen2moe",    WUBU_ARCH_QWEN_MOE,  WUBU_FAMILY_MOE,        WUBU_LAYOUT_BLK,          true,  true,  false },
    { "qwen3moe",    WUBU_ARCH_QWEN_MOE,  WUBU_FAMILY_MOE,        WUBU_LAYOUT_BLK,          true,  true,  false },
    { "deepseek2",   WUBU_ARCH_DEEPSEEK,  WUBU_FAMILY_MOE,        WUBU_LAYOUT_BLK,          true,  true,  false },
    { "phi2",        WUBU_ARCH_PHI,       WUBU_FAMILY_TRANSFORMER, WUBU_LAYOUT_BLK,          true,  true,  false },
    { "phi3",        WUBU_ARCH_PHI,       WUBU_FAMILY_TRANSFORMER, WUBU_LAYOUT_BLK,          true,  true,  false },
    { "lfm2",        WUBU_ARCH_LFM2,      WUBU_FAMILY_HYBRID_SSM, WUBU_LAYOUT_BLK,          true,  true,  false }
};

const wubu_arch_entry_t *wubu_arch_registry(void) { return g_arch_registry; }
int wubu_arch_registry_count(void) {
    return (int)(sizeof(g_arch_registry) / sizeof(g_arch_registry[0]));
}

const wubu_arch_entry_t *wubu_arch_lookup(const char *name)
{
    if (!name) return NULL;
    for (int i = 0; i < wubu_arch_registry_count(); i++)
        if (strcmp(g_arch_registry[i].gguf_arch, name) == 0) return &g_arch_registry[i];
    return NULL;
}

const char *wubu_arch_id_name(wubu_arch_id_t id)
{
    switch (id) {
    case WUBU_ARCH_LLAMA: return "llama";
    case WUBU_ARCH_GEMMA: return "gemma";
    case WUBU_ARCH_GEMMA4: return "gemma4";
    case WUBU_ARCH_QWEN: return "qwen";
    case WUBU_ARCH_QWEN_MOE: return "qwen-moe";
    case WUBU_ARCH_DEEPSEEK: return "deepseek";
    case WUBU_ARCH_MISTRAL: return "mistral";
    case WUBU_ARCH_PHI: return "phi";
    case WUBU_ARCH_LFM2: return "lfm2";
    default: return "unknown";
    }
}

const char *wubu_family_name(wubu_family_t f)
{
    switch (f) {
    case WUBU_FAMILY_HYBRID_SSM: return "hybrid-ssm";
    case WUBU_FAMILY_MOE: return "moe";
    default: return "transformer";
    }
}

const char *wubu_layout_name(wubu_tensor_layout_t l)
{
    return l == WUBU_LAYOUT_MODEL_LAYERS ? "model.layers.N" : "blk.N";
}

const char *wubu_runner_name(wubu_runner_t r)
{
    switch (r) {
    case WUBU_RUNNER_GEMMA4: return "gemma4";
    case WUBU_RUNNER_MOE: return "moe";
    case WUBU_RUNNER_SSM: return "hybrid-ssm";
    case WUBU_RUNNER_GENERIC_TRANSFORMER: return "generic-transformer";
    default: return "unknown";
    }
}

static int kv_i32(gguf_ctx *ctx, const char *key, int *out)
{
    int v = 0;
    if (!gguf_kv_get_i32(ctx, key, &v)) return 0;
    *out = v;
    return 1;
}

/* GGUF metadata keys are architecture-prefixed (gemma4.block_count,
 * llama.embedding_length, ...). Try the detected prefix, then every known
 * prefix, then the unprefixed key from older exporters. */
static const char *g_arch_prefixes[] = {
    "llama", "gemma", "gemma2", "gemma3", "gemma4", "mistral",
    "qwen2", "qwen3", "qwen2moe", "qwen3moe", "deepseek2", "phi2", "phi3", "lfm2", NULL
};

static int kv_arch_i32(gguf_ctx *ctx, const char *arch, const char *suffix, int *out)
{
    char key[160];
    if (arch && *arch) {
        snprintf(key, sizeof(key), "%s.%s", arch, suffix);
        if (kv_i32(ctx, key, out)) return 1;
    }
    for (int i = 0; g_arch_prefixes[i]; i++) {
        snprintf(key, sizeof(key), "%s.%s", g_arch_prefixes[i], suffix);
        if (kv_i32(ctx, key, out)) return 1;
    }
    return kv_i32(ctx, suffix, out);
}

static int kv_arch_f32(gguf_ctx *ctx, const char *arch, const char *suffix, float *out)
{
    char key[160];
    if (arch && *arch) {
        snprintf(key, sizeof(key), "%s.%s", arch, suffix);
        if (gguf_kv_get_f32(ctx, key, out)) return 1;
    }
    for (int i = 0; g_arch_prefixes[i]; i++) {
        snprintf(key, sizeof(key), "%s.%s", g_arch_prefixes[i], suffix);
        if (gguf_kv_get_f32(ctx, key, out)) return 1;
    }
    return gguf_kv_get_f32(ctx, suffix, out);
}

static int count_layers_from_tensors(gguf_ctx *ctx, wubu_tensor_layout_t layout)
{
    int max_layer = -1;
    for (int64_t i = 0; i < (int64_t)ctx->n_tensors; i++) {
        const char *name = ctx->tensors[i].name;
        int idx = -1;
        if (layout == WUBU_LAYOUT_MODEL_LAYERS) {
            if (strncmp(name, "model.layers.", 13) == 0) sscanf(name + 13, "%d.", &idx);
        } else {
            if (strncmp(name, "blk.", 4) == 0) sscanf(name + 4, "%d.", &idx);
        }
        if (idx > max_layer) max_layer = idx;
    }
    return max_layer + 1;
}

int wubu_arch_probe(gguf_ctx *ctx, wubu_arch_spec_t *out)
{
    if (!ctx || !out) return 0;
    memset(out, 0, sizeof(*out));
    out->family = WUBU_FAMILY_TRANSFORMER;
    out->layout = WUBU_LAYOUT_BLK;
    out->rope_base = 10000.0f;

    char arch[64] = {0};
    int have_arch = gguf_kv_get_str(ctx, "general.architecture", arch, sizeof(arch));
    const wubu_arch_entry_t *entry = have_arch ? wubu_arch_lookup(arch) : NULL;

    /* Tensor-name fallback: never assume Qwen/llama naming. */
    int has_model_layers = gguf_find_tensor(ctx, "model.layers.0.attn_norm.weight") ||
                           gguf_find_tensor(ctx, "model.layers.0.input_layernorm.weight");
    int has_blk = gguf_find_tensor(ctx, "blk.0.attn_norm.weight") != NULL;
    int has_ssm = gguf_find_tensor(ctx, "blk.0.ssm_beta.weight") ||
                  gguf_find_tensor(ctx, "model.layers.0.ssm_beta.weight");
    int has_moe = gguf_find_tensor(ctx, "blk.0.experts.0.weight") ||
                  gguf_find_tensor(ctx, "blk.0.ffn_experts.0.weight") ||
                  gguf_find_tensor(ctx, "model.layers.0.experts.0.weight");

    if (entry) {
        out->id = entry->id;
        out->family = entry->family;
        out->layout = entry->layout;
        out->supported = 1;
    } else {
        out->id = WUBU_ARCH_UNKNOWN;
        out->layout = has_model_layers ? WUBU_LAYOUT_MODEL_LAYERS : WUBU_LAYOUT_BLK;
        out->family = has_ssm ? WUBU_FAMILY_HYBRID_SSM :
                      (has_moe ? WUBU_FAMILY_MOE : WUBU_FAMILY_TRANSFORMER);
    }
    if (has_model_layers) out->layout = WUBU_LAYOUT_MODEL_LAYERS;
    else if (has_blk) out->layout = WUBU_LAYOUT_BLK;

    out->runner = out->id == WUBU_ARCH_GEMMA4 ? WUBU_RUNNER_GEMMA4 :
                  (out->family == WUBU_FAMILY_HYBRID_SSM ? WUBU_RUNNER_SSM :
                   (out->family == WUBU_FAMILY_MOE ? WUBU_RUNNER_MOE :
                    (entry ? WUBU_RUNNER_GENERIC_TRANSFORMER : WUBU_RUNNER_UNKNOWN)));
    if (have_arch) snprintf(out->arch_name, sizeof(out->arch_name), "%s", arch);
    else snprintf(out->arch_name, sizeof(out->arch_name), "unknown(%s)", wubu_family_name(out->family));

    int have = 1;
    have &= kv_arch_i32(ctx, arch, "block_count", &out->n_layers);
    have &= kv_arch_i32(ctx, arch, "embedding_length", &out->d_model);
    have &= kv_arch_i32(ctx, arch, "feed_forward_length", &out->d_ff);
    have &= kv_arch_i32(ctx, arch, "attention.head_count", &out->n_heads);
    have &= kv_arch_i32(ctx, arch, "attention.head_count_kv", &out->n_kv_heads);
    have &= kv_arch_i32(ctx, arch, "rope.dimension_count", &out->rotary_dim);
    kv_arch_i32(ctx, arch, "attention.sliding_window", &out->sliding_window);
    kv_arch_i32(ctx, arch, "attention.key_length", &out->head_dim);
    kv_arch_i32(ctx, arch, "attention.value_length", &out->head_dim);
    if (!kv_i32(ctx, "vocab_size", &out->vocab_size)) {
        int64_t v = gguf_tokenizer_token_count(ctx);
        if (v > 0) out->vocab_size = (int)v;
        else have = 0;
    }
    float base = 0.0f;
    if (kv_arch_f32(ctx, arch, "rope.theta", &base) || kv_arch_f32(ctx, arch, "rope.freq_base", &base)) {
        if (base > 0.0f) out->rope_base = base;
    } else have = 0;
    out->metadata_complete = have != 0;

    /* Shape-based fallbacks for models with sparse metadata. */
    if (out->n_layers <= 0) out->n_layers = count_layers_from_tensors(ctx, out->layout);
    const char *norm = out->layout == WUBU_LAYOUT_MODEL_LAYERS
        ? "model.layers.0.attn_norm.weight" : "blk.0.attn_norm.weight";
    gguf_tensor_info *nt = gguf_find_tensor(ctx, norm);
    if (out->d_model <= 0 && nt && nt->n_dims >= 1) out->d_model = (int)nt->dims[0];
    if (out->d_ff <= 0) {
        const char *gate = out->layout == WUBU_LAYOUT_MODEL_LAYERS
            ? "model.layers.0.ffn_gate.weight" : "blk.0.ffn_gate.weight";
        gguf_tensor_info *gt = gguf_find_tensor(ctx, gate);
        if (gt && gt->n_dims >= 2) out->d_ff = (int)gt->dims[1];
    }
    const char *qname = out->layout == WUBU_LAYOUT_MODEL_LAYERS
        ? "model.layers.0.attn_q.weight" : "blk.0.attn_q.weight";
    gguf_tensor_info *qt = gguf_find_tensor(ctx, qname);
    if (out->n_heads <= 0 && qt && qt->n_dims >= 2) {
        const char *qnorm = out->layout == WUBU_LAYOUT_MODEL_LAYERS
            ? "model.layers.0.attn_q_norm.weight" : "blk.0.attn_q_norm.weight";
        gguf_tensor_info *qn = gguf_find_tensor(ctx, qnorm);
        int hd = (qn && qn->n_dims >= 1) ? (int)qn->dims[0] : 0;
        if (hd > 0) out->n_heads = (int)(qt->dims[1] / hd);
        if (out->head_dim <= 0) out->head_dim = hd;
    }
    if (out->n_heads > 0 && out->d_model > 0 && out->head_dim <= 0)
        out->head_dim = out->d_model / out->n_heads;
    if (out->n_kv_heads <= 0) out->n_kv_heads = out->n_heads;
    if (out->rotary_dim <= 0) out->rotary_dim = out->head_dim;
    return 1;
}

void wubu_arch_print(const wubu_arch_spec_t *s, FILE *fp)
{
    if (!s || !fp) return;
    fprintf(fp, "architecture=%s id=%s family=%s layout=%s runner=%s supported=%s\n",
            s->arch_name, wubu_arch_id_name(s->id), wubu_family_name(s->family),
            wubu_layout_name(s->layout), wubu_runner_name(s->runner),
            s->supported ? "yes" : "generic");
    fprintf(fp, "layers=%d d_model=%d d_ff=%d heads=%d kv_heads=%d head_dim=%d rotary=%d vocab=%d rope_base=%.1f swa=%d metadata=%s\n",
            s->n_layers, s->d_model, s->d_ff, s->n_heads, s->n_kv_heads,
            s->head_dim, s->rotary_dim, s->vocab_size, s->rope_base,
            s->sliding_window, s->metadata_complete ? "complete" : "partial");
}
