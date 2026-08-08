/* lfm2_load.c -- LFM2.5 loader (C11, self-contained).
 *
 * MODIFIED (2026-08-07): loads BOTH safetensors shards AND GGUF
 * files. "make it load whatever" — if model_dir is a .gguf file
 * (or a dir containing one), tensors come through the in-tree
 * gguf_reader (dequant to F32) with safetensors->GGUF name
 * translation. Otherwise the existing safetensors shard path.
 */
#define _POSIX_C_SOURCE 200809L
#include "lfm2_load.h"
#include "safetensors_reader.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

/* ---- multi-shard safetensors context (self-scanned) ---- */
static st_ctx *g_shards[8];
static int g_nsh = 0;
static gguf_ctx *g_gguf = NULL;

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
    return g_gguf ? 1 : 0;
}

/* ---- safetensors -> GGUF tensor name translation (LFM2.5) ----
 * GGUF uses blk.N.attn_q.weight / attn_k / attn_v / attn_output,
 * ffn_gate / ffn_down / ffn_up, attn_q_norm / attn_k_norm,
 * token_embd.weight, output_norm.weight. The safetensors names are
 * model.layers.N.self_attn.q_proj.weight etc. Map role + layer. */
static void translate_name(const char *st_name, char *out, size_t outsz) {
    /* try direct first (same name in some GGUFs) */
    snprintf(out, outsz, "%s", st_name);
    int layer = -1;
    char role[64] = {0};
    if (sscanf(st_name, "model.layers.%d.", &layer) == 1) {
        const char *rest = strstr(st_name, ".");
        rest = strstr(rest + 1, ".");
        rest = strstr(rest + 1, ".");
        if (rest) {
            rest++; /* past the third dot */
            if      (strstr(rest, "self_attn.q_proj.weight"))     snprintf(role, sizeof(role), "attn_q.weight");
            else if (strstr(rest, "self_attn.k_proj.weight"))     snprintf(role, sizeof(role), "attn_k.weight");
            else if (strstr(rest, "self_attn.v_proj.weight"))     snprintf(role, sizeof(role), "attn_v.weight");
            else if (strstr(rest, "self_attn.out_proj.weight"))   snprintf(role, sizeof(role), "attn_output.weight");
            else if (strstr(rest, "self_attn.q_layernorm.weight"))snprintf(role, sizeof(role), "attn_q_norm.weight");
            else if (strstr(rest, "self_attn.k_layernorm.weight"))snprintf(role, sizeof(role), "attn_k_norm.weight");
            else if (strstr(rest, "feed_forward.w1.weight"))      snprintf(role, sizeof(role), "ffn_gate.weight");
            else if (strstr(rest, "feed_forward.w2.weight"))      snprintf(role, sizeof(role), "ffn_down.weight");
            else if (strstr(rest, "feed_forward.w3.weight"))      snprintf(role, sizeof(role), "ffn_up.weight");
            else if (strstr(rest, "ffn_norm.weight"))             snprintf(role, sizeof(role), "ffn_norm.weight");
            else if (strstr(rest, "operator_norm.weight"))        snprintf(role, sizeof(role), "attn_output_norm.weight");
            else if (strstr(rest, "conv.in_proj.weight"))         snprintf(role, sizeof(role), "attn_q.weight");
            else if (strstr(rest, "conv.conv.weight"))            snprintf(role, sizeof(role), "attn_k.weight");
            else if (strstr(rest, "conv.out_proj.weight"))        snprintf(role, sizeof(role), "attn_v.weight");
            if (role[0]) {
                snprintf(out, outsz, "blk.%d.%s", layer, role);
                return;
            }
        }
    }
    if      (strcmp(st_name, "model.embed_tokens.weight") == 0)   snprintf(out, outsz, "token_embd.weight");
    else if (strcmp(st_name, "model.embedding_norm.weight") == 0) snprintf(out, outsz, "output_norm.weight");
}

/* Load a BF16/F32 tensor by name across opened shards OR GGUF. */
static float *load_bf16_f32(const char *name) {
    if (g_gguf) {
        char gname[256];
        translate_name(name, gname, sizeof(gname));
        gguf_tensor_info *t = gguf_find_tensor(g_gguf, gname);
        if (!t && strcmp(gname, name) != 0) t = gguf_find_tensor(g_gguf, name);
        if (!t) {
            /* "load whatever": prefix-scan the GGUF's actual tensors.
             * Match by the trailing role (e.g. ".self_attn.q_proj.weight"
             * or ".conv.conv.weight") + layer number. This handles any
             * naming dialect. */
            int layer = -1;
            if (sscanf(name, "model.layers.%d.", &layer) == 1) {
                const char *role = strstr(name, ".self_attn.");
                if (!role) role = strstr(name, ".feed_forward.");
                if (!role) role = strstr(name, ".ffn_norm.");
                if (!role) role = strstr(name, ".operator_norm.");
                if (!role) role = strstr(name, ".conv.");
                if (!role) role = strstr(name, ".embedding_norm.");
                if (role) {
                    for (int64_t i = 0; i < g_gguf->n_tensors && !t; i++) {
                        const gguf_tensor_info *cand = &g_gguf->tensors[i];
                        const char *cn = cand->name;
                        /* same layer AND same role suffix */
                        char layerbuf[16];
                        snprintf(layerbuf, sizeof(layerbuf), "%d.", layer);
                        if (strstr(cn, layerbuf) && strstr(cn, role + 1)) {
                            t = (gguf_tensor_info *)cand;
                        }
                    }
                }
            } else if (strcmp(name, "model.embed_tokens.weight") == 0) {
                t = gguf_find_tensor(g_gguf, "token_embd.weight");
            }
        }
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
        /* ---- GGUF without config.json: infer from tensor shapes ----
         * token_embd.weight: dims = [d_model, vocab] (GGUF dims[0] is
         * innermost/rows). blk.N.* count = n_layers. attn_q rows =
         * n_q_heads * head_dim. ffn_gate rows = ff_dim. */
        m->d_model = 0; m->n_layers = 0; m->ff_dim = 0; m->n_q_heads = 0;
        m->n_kv_heads = 0; m->vocab_size = 0; m->conv_dim = 0;
        for (int64_t i = 0; i < g_gguf->n_tensors; i++) {
            const gguf_tensor_info *t = &g_gguf->tensors[i];
            const char *n = t->name;
            if (!strcmp(n, "token_embd.weight")) {
                m->d_model = (int)t->dims[0];
                m->vocab_size = (int)(t->n_dims > 1 ? t->dims[1] : t->dims[0]);
            } else if (strncmp(n, "blk.", 4) == 0) {
                int layer = atoi(n + 4);
                if (layer + 1 > m->n_layers) m->n_layers = layer + 1;
                if (strstr(n, "attn_q.weight")) {
                    m->n_q_heads = (int)t->dims[1]; /* n_q_heads * head_dim */
                    /* keep dims[0] as d_model if not set */
                    if (!m->d_model) m->d_model = (int)t->dims[0];
                } else if (strstr(n, "attn_k.weight")) {
                    m->n_kv_heads = (int)t->dims[1];
                } else if (strstr(n, "ffn_gate.weight") || strstr(n, "ffn_up.weight")) {
                    m->ff_dim = (int)t->dims[1];
                }
            }
        }
        fprintf(stderr, "[lfm2] config.json absent; inferred d=%d layers=%d ff=%d q=%d kv=%d vocab=%d\n",
                m->d_model, m->n_layers, m->ff_dim, m->n_q_heads, m->n_kv_heads, m->vocab_size);
    }
    if (!have_config && !g_gguf) {
        fprintf(stderr, "lfm2: no config.json\n");
        return false;
    }

    m->head_dim = m->d_model / m->n_q_heads;
    /* rope_theta: prefer nested rope_parameters.rope_theta (LFM2.5 = 1e7) */
    m->rope_theta = 10000.0f;
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
        } else if (g_gguf) {
            /* infer conv layers: a layer with conv tensors has
             * blk.N.conv.weight — check per layer */
            for (int l = 0; l < m->n_layers; l++) {
                char tn[128]; snprintf(tn, sizeof(tn), "blk.%d.conv.weight", l);
                if (gguf_find_tensor(g_gguf, tn)) m->is_conv[l] = true;
            }
        }
    }

    /* ---- per-layer weights ---- */
    for (int l = 0; l < m->n_layers; l++) {
        lfm2_layer_t *L = &m->layers[l];
        L->conv_k = 3;
        char nm[160];
#define LOAD(var, fmt) do { snprintf(nm, sizeof(nm), fmt, l); \
            L->var = load_bf16_f32(nm); \
            if (!L->var) fprintf(stderr, "lfm2: missing %s\n", nm); } while (0)
        if (m->is_conv[l]) {
            LOAD(in_proj, "model.layers.%d.conv.in_proj.weight");
            LOAD(conv_w,   "model.layers.%d.conv.conv.weight");
            LOAD(out_proj, "model.layers.%d.conv.out_proj.weight");
        } else {
            LOAD(q_proj, "model.layers.%d.self_attn.q_proj.weight");
            LOAD(k_proj, "model.layers.%d.self_attn.k_proj.weight");
            LOAD(v_proj, "model.layers.%d.self_attn.v_proj.weight");
            LOAD(o_proj, "model.layers.%d.self_attn.out_proj.weight");
            LOAD(q_ln,   "model.layers.%d.self_attn.q_layernorm.weight");
            LOAD(k_ln,   "model.layers.%d.self_attn.k_layernorm.weight");
        }
        LOAD(w1,      "model.layers.%d.feed_forward.w1.weight");
        LOAD(w2,      "model.layers.%d.feed_forward.w2.weight");
        LOAD(w3,      "model.layers.%d.feed_forward.w3.weight");
        LOAD(ffn_norm, "model.layers.%d.ffn_norm.weight");
        LOAD(op_norm,  "model.layers.%d.operator_norm.weight");
#undef LOAD
    }

    /* embeddings + norms (tied lm_head) */
    m->embed = load_bf16_f32("model.embed_tokens.weight");
    m->embed_norm = load_bf16_f32("model.embedding_norm.weight");
    m->kv_max_t = 8192;
    size_t kv_bytes = (size_t)m->n_layers * 2 * m->n_kv_heads * m->head_dim * m->kv_max_t;
    m->kv_cache = (float *)xmalloc(kv_bytes * sizeof(float));
    memset(m->kv_cache, 0, kv_bytes * sizeof(float));

    if (!m->embed || !m->embed_norm) { fprintf(stderr, "lfm2: missing embed/embed_norm\n"); return false; }
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
