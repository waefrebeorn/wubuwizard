/* lfm2_load.c -- LFM2.5 loader (C11, self-contained).
 *
 * MODIFIED (2026-08-07): loads BOTH safetensors shards AND GGUF
 * files via the engine's role-based name resolver (wubu_gguf_names
 * — handles Qwen/Gemma/HF naming dialects + SSM/dense/MoE
 * detection). "Make it load whatever": pass a .gguf file (or a
 * dir containing one) and tensors come through the in-tree
 * gguf_reader, resolved by ROLE not by hardcoded name.
 */
#define _POSIX_C_SOURCE 200809L
#include "lfm2_load.h"
#include "safetensors_reader.h"
#include "gguf_reader.h"
#include "wubu_gguf_names.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

/* ---- multi-shard safetensors context (self-scanned) ---- */
static st_ctx *g_shards[8];
static int g_nsh = 0;
static gguf_ctx *g_gguf = NULL;
static const uint8_t *g_gguf_blob = NULL;
static wubu_gguf_names_t g_names;

static int is_gguf_path(const char *p) {
    size_t n = strlen(p);
    return n > 5 && strcmp(p + n - 5, ".gguf") == 0;
}

int lfm2_open_shards(const char *model_dir) {
    DIR *d = opendir(model_dir);
    if (!d) return 0;
    struct dirent *e;
    while ((e = readdir(d)) && g_nsh < 8) {
        if (strstr(e->d_name, "model-") && strstr(e->d_name, "-of-") &&
            strstr(e->d_name, ".safetensors")) {
            char p[2048]; snprintf(p, sizeof(p), "%s/%s", model_dir, e->d_name);
            st_ctx *s = st_open(p);
            if (s) g_shards[g_nsh++] = s;
        }
    }
    closedir(d);
    return g_nsh;
}

/* Open a GGUF (either the path itself or a .gguf found in the dir). */
static int lfm2_open_gguf(const char *model_dir) {
    char path[2048];
    if (is_gguf_path(model_dir)) {
        snprintf(path, sizeof(path), "%s", model_dir);
    } else {
        path[0] = 0;
        DIR *d = opendir(model_dir);
        if (!d) return 0;
        struct dirent *e;
        while ((e = readdir(d))) {
            if (strstr(e->d_name, ".gguf")) {
                snprintf(path, sizeof(path), "%s/%s", model_dir, e->d_name);
                break;
            }
        }
        closedir(d);
        if (!path[0]) return 0;
    }
    g_gguf = gguf_open(path);
    if (!g_gguf) return 0;
    wubu_gguf_names_detect(g_gguf, &g_names);
    gguf_buffer_data(g_gguf);   /* mmap the quantized blob, zero copy */
    g_gguf_blob = (const uint8_t *)g_gguf->data_blob;
    if (!g_gguf_blob) { fprintf(stderr, "lfm2: gguf buffer_data failed\n"); return 0; }
    fprintf(stderr, "[lfm2] gguf detect: conv=%d layers=%d ssm=%d moe=%d dense=%d gqa=%d\n",
            g_names.convention, g_names.n_layers, g_names.has_ssm,
            g_names.has_moe, g_names.has_dense_ffn, g_names.has_gqa);
    return 1;
}

/* ---- safetensors -> GGUF tensor name translation (LFM2.5) ----
 * The engine's role-based resolver (wubu_gguf_find) handles all
 * naming dialects; this maps a safetensors path to a role. */
static wubu_gguf_role_t name_to_role(const char *st_name) {
    if (strstr(st_name, "conv.in_proj.weight"))
        return WUBU_T_CONV_IN;
    if (strstr(st_name, "conv.conv.weight"))
        return WUBU_T_CONV_W;
    if (strstr(st_name, "conv.out_proj.weight"))
        return WUBU_T_CONV_OUT;
    if (strstr(st_name, "self_attn.q_proj.weight"))
        return WUBU_T_ATTN_Q;
    if (strstr(st_name, "self_attn.k_proj.weight"))
        return WUBU_T_ATTN_K;
    if (strstr(st_name, "self_attn.v_proj.weight"))
        return WUBU_T_ATTN_V;
    if (strstr(st_name, "self_attn.out_proj.weight"))
        return WUBU_T_ATTN_O;
    if (strstr(st_name, "self_attn.q_layernorm.weight"))
        return WUBU_T_ATTN_Q_NORM;
    if (strstr(st_name, "self_attn.k_layernorm.weight"))
        return WUBU_T_ATTN_K_NORM;
    if (strstr(st_name, "feed_forward.w1.weight"))
        return WUBU_T_FFN_GATE;
    if (strstr(st_name, "feed_forward.w2.weight"))
        return WUBU_T_FFN_DOWN;
    if (strstr(st_name, "feed_forward.w3.weight"))
        return WUBU_T_FFN_UP;
    if (strstr(st_name, "ffn_norm.weight"))
        return WUBU_T_FFN_NORM;
    if (strstr(st_name, "operator_norm.weight"))
        return WUBU_T_POST_ATTN_NORM;
    if (strcmp(st_name, "model.embed_tokens.weight") == 0)
        return WUBU_T_TOKEN_EMBD;
    if (strcmp(st_name, "model.embedding_norm.weight") == 0)
        return WUBU_T_OUTPUT_NORM;
    return WUBU_T_COUNT;
}

/* Find a tensor by safetensors name across opened shards OR GGUF.
 * Returns the GGUF tensor info (blob pointer + type) or NULL. */
static gguf_tensor_info *find_tensor_info(const char *name) {
    if (!g_gguf) return NULL;
    gguf_tensor_info *t = NULL;
    wubu_gguf_role_t role = name_to_role(name);
    int layer = -1;
    sscanf(name, "model.layers.%d.", &layer);
    if (role != WUBU_T_COUNT) {
        t = wubu_gguf_find(g_gguf, layer, role);
        if (!t && layer < 0) t = wubu_gguf_find_global(g_gguf, role);
    }
    if (!t) t = gguf_find_tensor(g_gguf, name);
    return t;
}

/* Load a BF16/F32 tensor by name across opened shards OR GGUF. */
static float *load_bf16_f32(const char *name) {
    if (g_gguf) {
        gguf_tensor_info *t = find_tensor_info(name);
        if (!t) return NULL;
        int64_t ne = 1;
        for (int di = 0; di < t->n_dims; di++) ne *= t->dims[di];
        float *f = (float *)malloc((size_t)ne * sizeof(float));
        if (!f) return NULL;
        if (gguf_read_tensor_f32(g_gguf, t, f, ne) != (int)ne) { free(f); return NULL; }
        return f;
    }
    for (int s = 0; s < g_nsh; s++) {
        const st_tensor_info *t = st_find_tensor(g_shards[s], name);
        if (!t) continue;
        int64_t ne = t->n_elems;
        float *f = (float *)malloc((size_t)ne * sizeof(float));
        if (st_read_tensor_f32(g_shards[s], t, f, ne) != ne) { free(f); continue; }
        return f;
    }
    return NULL;
}

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "lfm2 oom\n"); exit(1); } return p; }

bool lfm2_load(const char *model_dir, lfm2_model_t *m) {
    memset(m, 0, sizeof(*m));
    int have = 0;
    int head_dim_inf = 0;   /* from attn_q_norm [hd] in the GGUF, if present */
    if (lfm2_open_gguf(model_dir)) {
        have = 1;
        fprintf(stderr, "[lfm2] opened GGUF: %s\n", model_dir);
    } else if (lfm2_open_shards(model_dir) > 0) {
        have = 1;
        fprintf(stderr, "[lfm2] opened %d shard(s)\n", g_nsh);
    }
    if (!have) {
        fprintf(stderr, "lfm2: no safetensors shards or GGUF in %s\n", model_dir);
        return false;
    }
    /* ---- config.json: dims ---- */
    char cfg[2048]; snprintf(cfg, sizeof(cfg), "%s/config.json", model_dir);
    FILE *f = fopen(cfg, "rb");
    int have_config = (f != NULL);
    if (f) {
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        char *buf = (char *)xmalloc((size_t)sz + 1); size_t _r1 = fread(buf, 1, sz, f); (void)_r1; buf[sz] = 0; fclose(f);

        m->d_model = 0; m->n_layers = 0; m->ff_dim = 0; m->n_q_heads = 0;
        m->n_kv_heads = 0; m->vocab_size = 0; m->conv_dim = 0;
#define GETI(key, field) do { const char *p = strstr(buf, key); if (p) { p = strchr(p, ':'); if (p) m->field = (int)atoll(p + 1); } } while (0)
        GETI("\"hidden_size\"", d_model);
        GETI("\"num_hidden_layers\"", n_layers);
        GETI("\"intermediate_size\"", ff_dim);
        GETI("\"num_attention_heads\"", n_q_heads);
        GETI("\"num_key_value_heads\"", n_kv_heads);
        GETI("\"vocab_size\"", vocab_size);
        GETI("\"conv_dim\"", conv_dim);
#undef GETI
        free(buf);
    } else if (g_gguf) {
        /* ---- GGUF without config.json: use the lfm2.* KV metadata
         * (authoritative: block_count, head_count, head_count_kv,
         * rope.freq_base, shortconv.l_cache) with tensor-shape
         * fallbacks. ---- */
        m->d_model = 0; m->n_layers = 0; m->ff_dim = 0; m->n_q_heads = 0;
        m->n_kv_heads = 0; m->vocab_size = 0; m->conv_dim = 0;
        int head_dim_inf = 0;
        gguf_kv_get_i32(g_gguf, "lfm2.block_count", &m->n_layers);
        gguf_kv_get_i32(g_gguf, "lfm2.embedding_length", &m->d_model);
        gguf_kv_get_i32(g_gguf, "lfm2.feed_forward_length", &m->ff_dim);
        gguf_kv_get_i32(g_gguf, "lfm2.attention.head_count", &m->n_q_heads);
        gguf_kv_get_i32(g_gguf, "lfm2.vocab_size", &m->vocab_size);
        gguf_kv_get_f32(g_gguf, "lfm2.rope.freq_base", &m->rope_theta);
        int lcache = 0;
        gguf_kv_get_i32(g_gguf, "lfm2.shortconv.l_cache", &lcache);
        if (lcache > 1) m->conv_k = lcache;
        /* head_count_kv: per-layer array (0 = conv layer) — authoritative
         * for BOTH n_kv_heads and the conv/attn split. */
        int kv_arr[64];
        int kv_n = gguf_kv_get_i32_arr(g_gguf, "lfm2.attention.head_count_kv", kv_arr, 64);
        if (kv_n >= m->n_layers && m->n_layers > 0) {
            int max_kv = 0;
            for (int l = 0; l < m->n_layers; l++)
                if (kv_arr[l] > max_kv) max_kv = kv_arr[l];
            m->n_kv_heads = max_kv;
            m->is_conv_from_kv = 1;
            for (int l = 0; l < m->n_layers; l++)
                m->is_conv_kv[l] = (kv_arr[l] <= 0);
        }
        for (int64_t i = 0; i < g_gguf->n_tensors; i++) {
            const gguf_tensor_info *t = &g_gguf->tensors[i];
            const char *n = t->name;
            if (!strcmp(n, "token_embd.weight")) {
                if (!m->d_model) m->d_model = (int)t->dims[0];
                if (!m->vocab_size) m->vocab_size = (int)(t->n_dims > 1 ? t->dims[1] : t->dims[0]);
            } else if (strncmp(n, "blk.", 4) == 0) {
                int layer = atoi(n + 4);
                if (layer + 1 > m->n_layers) m->n_layers = layer + 1;
                if (strstr(n, "attn_q_norm.weight")) {
                    /* [head_dim] — the per-head q norm reveals head_dim */
                    head_dim_inf = (int)t->dims[0];
                } else if (strstr(n, "attn_q.weight")) {
                    if (!m->d_model) m->d_model = (int)t->dims[0];
                } else if (strstr(n, "attn_k.weight")) {
                    if (!m->n_kv_heads) m->n_kv_heads = (int)t->dims[1];
                } else if (strstr(n, "shortconv.in_proj.weight")) {
                    /* GGUF dims: [d_model, 3*conv_dim] — conv_dim = dims[1]/3 */
                    int d1 = (int)(t->n_dims > 1 ? t->dims[1] : 0);
                    if (d1 % 3 == 0) m->conv_dim = d1 / 3;
                } else if (strstr(n, "shortconv.out_proj.weight")) {
                    /* [d_model, conv_dim] — cross-check */
                    int d1 = (int)(t->n_dims > 1 ? t->dims[1] : 0);
                    if (!m->conv_dim) m->conv_dim = d1;
                } else if (strstr(n, "ffn_gate.weight") || strstr(n, "ffn_up.weight")) {
                    if (!m->ff_dim) m->ff_dim = (int)t->dims[1];
                }
            }
        }
        fprintf(stderr, "[lfm2] config.json absent; inferred d=%d layers=%d ff=%d q=%d kv=%d vocab=%d hd=%d rope=%.3g\n",
                m->d_model, m->n_layers, m->ff_dim, m->n_q_heads, m->n_kv_heads, m->vocab_size, head_dim_inf, m->rope_theta);
    }
    if (!have_config && !g_gguf) {
        fprintf(stderr, "lfm2: no config.json\n");
        return false;
    }

    m->head_dim = (head_dim_inf > 0) ? head_dim_inf :
                  (m->n_q_heads > 0 ? m->d_model / m->n_q_heads : 128);
    /* n_kv_heads from attn_k dims[1] is actually kv_dim (heads*hd); if it
     * exceeds d_model it's kv_dim — normalize to heads. */
    if (m->n_kv_heads > 0 && m->head_dim > 0 && m->n_kv_heads > m->d_model / m->head_dim)
        m->n_kv_heads = m->n_kv_heads / m->head_dim;
    if (m->n_kv_heads <= 0) m->n_kv_heads = 8;   /* LFM2.5 default */
    /* rope_theta: KV freq_base (1e7 for LFM2.5) wins; config.json
     * rope_parameters.rope_theta overrides; 10000 is the LAST resort. */
    if (m->rope_theta <= 0.0f) m->rope_theta = 10000.0f;
    {
        FILE *fp = fopen(cfg, "rb");
        if (fp) {
            fseek(fp, 0, SEEK_END); long z = ftell(fp); fseek(fp, 0, SEEK_SET);
            char *b = (char *)xmalloc((size_t)z + 1); size_t _r2 = fread(b, 1, z, fp); (void)_r2; b[z] = 0; fclose(fp);
            const char *rp = strstr(b, "\"rope_parameters\"");
            const char *th = rp ? strstr(rp, "\"rope_theta\"") : strstr(b, "\"rope_theta\"");
            if (th) { th = strchr(th, ':'); if (th) m->rope_theta = (float)atof(th + 1); }
            free(b);
        }
    }
    if (!m->d_model || !m->n_layers) { fprintf(stderr, "lfm2: bad config\n"); return false; }

    m->is_conv = (bool *)xmalloc(m->n_layers * sizeof(bool));
    m->layers = (lfm2_layer_t *)xmalloc(m->n_layers * sizeof(lfm2_layer_t));
    memset(m->layers, 0, m->n_layers * sizeof(lfm2_layer_t));

    /* layer_types from config.json (default: all full_attention
     * when absent — the GGUF path infers conv layers by checking
     * for conv tensors below) */
    for (int l = 0; l < m->n_layers; l++) m->is_conv[l] = false;
    {
        FILE *fc = fopen(cfg, "rb");
        if (fc) {
            fseek(fc, 0, SEEK_END); long csz = ftell(fc); fseek(fc, 0, SEEK_SET);
            char *cb = (char *)xmalloc((size_t)csz + 1); size_t _r3 = fread(cb, 1, csz, fc); (void)_r3; cb[csz] = 0; fclose(fc);
            const char *lt = strstr(cb, "\"layer_types\"");
            int li = 0;
            if (lt) {
                const char *p = strchr(lt, '[');
                if (p) while (*++p && *p != ']' && li < m->n_layers) {
                    if (*p == '"') {
                        if (!strncmp(p + 1, "conv", 4)) m->is_conv[li++] = true;
                        else if (!strncmp(p + 1, "full_attention", 14)) m->is_conv[li++] = false;
                        while (*p && *p != '"') p++;
                    }
                }
            }
            free(cb);
            if (li != m->n_layers) fprintf(stderr, "lfm2: warn layer_types count %d != %d\n", li, m->n_layers);
        } else if (g_gguf && m->is_conv_from_kv) {
            /* authoritative: head_count_kv per-layer (0 = conv) */
            for (int l = 0; l < m->n_layers && l < 128; l++)
                m->is_conv[l] = m->is_conv_kv[l];
        } else if (g_gguf) {
            /* infer conv layers: a layer with conv tensors has
             * blk.N.conv.weight or blk.N.shortconv.conv.weight —
             * check per layer (LFM2.5 GGUF names them shortconv.*) */
            for (int l = 0; l < m->n_layers; l++) {
                char tn[128]; snprintf(tn, sizeof(tn), "blk.%d.conv.weight", l);
                char tn2[128]; snprintf(tn2, sizeof(tn2), "blk.%d.shortconv.conv.weight", l);
                if (gguf_find_tensor(g_gguf, tn) || gguf_find_tensor(g_gguf, tn2))
                    m->is_conv[l] = true;
            }
        }
    }

    /* ---- per-layer weights ---- */
    for (int l = 0; l < m->n_layers; l++) {
        lfm2_layer_t *L = &m->layers[l];
        L->conv_k = (m->conv_k > 1) ? m->conv_k : 3;
        char nm[160];
        /* GGUF: big matrices stay QUANTIZED in the blob (lazy materialize
         * per layer in forward — never dequantize all 30 layers to F32,
         * that OOMs the 5.8GB box: ~10GB). Norms stay F32. */
#define LOADQ(var, qvar, qtype, fmt) do { snprintf(nm, sizeof(nm), fmt, l); \
            gguf_tensor_info *ti = g_gguf ? find_tensor_info(nm) : NULL; \
            if (ti) { L->qvar = (const uint8_t *)g_gguf_blob + ti->data_offset; L->qtype = ti->ggml_type; } \
            else { L->var = load_bf16_f32(nm); \
                   if (!L->var) fprintf(stderr, "lfm2: missing %s\n", nm); } } while (0)
#define LOAD(var, fmt) do { snprintf(nm, sizeof(nm), fmt, l); \
            L->var = load_bf16_f32(nm); \
            if (!L->var) fprintf(stderr, "lfm2: missing %s\n", nm); } while (0)
        if (m->is_conv[l]) {
            LOADQ(in_proj, q_in_proj, q_in_proj_t, "model.layers.%d.conv.in_proj.weight");
            LOADQ(conv_w,   q_conv_w,  q_conv_w_t,  "model.layers.%d.conv.conv.weight");
            LOADQ(out_proj, q_out_proj,q_out_proj_t,"model.layers.%d.conv.out_proj.weight");
        } else {
            LOADQ(q_proj, q_q_proj, q_q_proj_t, "model.layers.%d.self_attn.q_proj.weight");
            LOADQ(k_proj, q_k_proj, q_k_proj_t, "model.layers.%d.self_attn.k_proj.weight");
            LOADQ(v_proj, q_v_proj, q_v_proj_t, "model.layers.%d.self_attn.v_proj.weight");
            LOADQ(o_proj, q_o_proj, q_o_proj_t, "model.layers.%d.self_attn.out_proj.weight");
            LOAD(q_ln,   "model.layers.%d.self_attn.q_layernorm.weight");
            LOAD(k_ln,   "model.layers.%d.self_attn.k_layernorm.weight");
        }
        LOADQ(w1,      q_w1, q_w1_t, "model.layers.%d.feed_forward.w1.weight");
        LOADQ(w2,      q_w2, q_w2_t, "model.layers.%d.feed_forward.w2.weight");
        LOADQ(w3,      q_w3, q_w3_t, "model.layers.%d.feed_forward.w3.weight");
        LOAD(ffn_norm, "model.layers.%d.ffn_norm.weight");
        LOAD(op_norm,  "model.layers.%d.operator_norm.weight");
#undef LOADQ
#undef LOAD
    }

    /* embeddings + norms (tied lm_head) */
    if (g_gguf) {
        gguf_tensor_info *te = find_tensor_info("model.embed_tokens.weight");
        if (te && g_gguf_blob) {
            m->q_embed = g_gguf_blob + te->data_offset;
            m->q_embed_type = te->ggml_type;
            int64_t n_elems = 1;
            for (int di = 0; di < te->n_dims; di++) n_elems *= te->dims[di];
            int64_t raw = gguf_raw_size(te->ggml_type, n_elems);
            m->embed_bytes_per_row = (int)(raw / te->dims[1]); /* exact bytes per vocab row */
            if (m->embed_bytes_per_row <= 0) m->embed_bytes_per_row = m->d_model * 4;
        }
    }
    if (!m->q_embed) m->embed = load_bf16_f32("model.embed_tokens.weight");
    m->embed_norm = load_bf16_f32("model.embedding_norm.weight");
    m->kv_max_t = 8192;
    size_t kv_bytes = (size_t)m->n_layers * 2 * m->n_kv_heads * m->head_dim * m->kv_max_t;
    m->kv_cache = (float *)xmalloc(kv_bytes * sizeof(float));
    memset(m->kv_cache, 0, kv_bytes * sizeof(float));

    if ((!m->embed && !m->q_embed) || !m->embed_norm) { fprintf(stderr, "lfm2: missing embed/embed_norm\n"); return false; }
    fprintf(stderr, "[lfm2] loaded d=%d layers=%d q=%d kv=%d hd=%d ff=%d vocab=%d conv_dim=%d rope_theta=%.0f\n",
            m->d_model, m->n_layers, m->n_q_heads, m->n_kv_heads, m->head_dim, m->ff_dim, m->vocab_size, m->conv_dim, m->rope_theta);
    return true;
}

void lfm2_free(lfm2_model_t *m) {
    if (m->layers) {
        for (int l = 0; l < m->n_layers; l++) {
            lfm2_layer_t *L = &m->layers[l];
            free(L->in_proj); free(L->conv_w); free(L->out_proj);
            free(L->q_proj); free(L->k_proj); free(L->v_proj); free(L->o_proj);
            free(L->q_ln); free(L->k_ln);
            free(L->w1); free(L->w2); free(L->w3);
            free(L->ffn_norm); free(L->op_norm);
        }
        free(m->layers);
    }
    free(m->is_conv); free(m->embed); free(m->embed_norm); free(m->kv_cache);
    for (int s = 0; s < g_nsh; s++) if (g_shards[s]) st_close(g_shards[s]);
    g_nsh = 0;
    if (g_gguf) { gguf_close(g_gguf); g_gguf = NULL; }
    memset(m, 0, sizeof(*m));
}
