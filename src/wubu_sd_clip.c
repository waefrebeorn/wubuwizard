/* wubu_sd_clip.c -- CLIP text encoder + byte-level BPE tokenizer for the
 * WuBu stable-diffusion engine (SD 1.5 / CLIP-ViT-L-14 text).
 *
 * C11, self-contained, opaque struct. Weights stay quantized in the GGUF
 * blob; each tensor is materialized to F32 once on first use (lazy cache)
 * -- never dequantize all 123M params up front on the 5.8GB box.
 *
 * Architecture (SD1.5 CLIP text):
 *   token_embedding  [49408, 768]  Q4_0
 *   position_embedding [77, 768]   Q4_0
 *   12 x EncoderLayer:
 *     layer_norm1 (768) -> self_attn (q/k/v/out each 768, 12 heads x 64,
 *     causal mask, 1/sqrt(64) scale) -> residual
 *     layer_norm2 (768) -> mlp fc1 768->3072 (quick_gelu) -> fc2 3072->768
 *     -> residual
 *   final_layer_norm (768)
 * Output: hidden[0..76] (the full 77x768 sequence -- SD1.5 UNet cross-attn
 * uses the whole sequence as K/V, no pooling/projection needed).
 */
#include "wubu_sd_clip.h"
#include "wubu_sd_ops.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define CLIP_DMODEL   768
#define CLIP_HEADS    12
#define CLIP_HEAD_D   64
#define CLIP_MLP      3072
#define CLIP_LAYERS   12
#define CLIP_CTX      77
#define CLIP_VOCAB    49408
#define CLIP_BOS      49406   /* <|startoftext|> */
#define CLIP_EOS      49407   /* <|endoftext|> */

/* ---------------- byte-level BPE ---------------- */

typedef struct {
    char **merges;      /* "a b" strings, sorted by rank (rank = index) */
    int n_merges;
    char **vocab;       /* token string -> id (id = index, 49408 entries) */
    char byte_to_char[256][5]; /* utf-8 encoded */
} clip_bpe_t;

static void bpe_bytes_to_unicode(char (*out)[5]) {
    /* standard GPT-2 bytes_to_unicode table */
    int bs = 0;
    int b;  /* MUST be int: unsigned char wraps 255->0 and loops forever */
    for (b = '!'; b <= '~'; b++) { /* 33..126 */
        out[b][0] = (char)b; out[b][1] = 0; bs++;
    }
    for (b = 161; b <= 172; b++) { out[b][0] = (char)b; out[b][1] = 0; bs++; }
    for (b = 174; b <= 255; b++) { out[b][0] = (char)b; out[b][1] = 0; bs++; }
    int n = 0;
    for (b = 0; b < 256; b++) {
        if (out[b][0]) continue;
        /* unicode 256+n */
        unsigned cp = 256 + n++;
        out[b][0] = (char)(0xC0 | (cp >> 6));
        out[b][1] = (char)(0x80 | (cp & 0x3F));
        out[b][2] = 0;
    }
}

static void bpe_free_dummy(clip_bpe_t *t);  /* fwd decl (defined below) */
static clip_bpe_t *bpe_load(const char *vocab_path, const char *merges_path);

/* Resolve the CLIP BPE files without /tmp: try (1) CLIP_DIR env,
 * (2) ./models/clip relative to CWD, (3) repo models/clip next to
 * the binary, (4) legacy /tmp/clip. Returns nonzero if found. */
static int bpe_find_paths(char *vbuf, size_t vsz, char *mbuf, size_t msz) {
    const char *cands[][2] = {
        { "models/clip/vocab.json",      "models/clip/merges.txt" },
        { "../models/clip/vocab.json",   "../models/clip/merges.txt" },
        { "/tmp/clip/vocab.json",        "/tmp/clip/merges.txt" },
    };
    const char *env = getenv("CLIP_DIR");
    if (env && env[0]) {
        snprintf(vbuf, vsz, "%s/vocab.json", env);
        snprintf(mbuf, msz, "%s/merges.txt", env);
        FILE *f = fopen(vbuf, "rb");
        if (f) { fclose(f); return 1; }
    }
    for (size_t i = 0; i < sizeof(cands) / sizeof(cands[0]); i++) {
        FILE *f = fopen(cands[i][0], "rb");
        if (!f) continue;
        fclose(f);
        f = fopen(cands[i][1], "rb");
        if (!f) continue;
        fclose(f);
        snprintf(vbuf, vsz, "%s", cands[i][0]);
        snprintf(mbuf, msz, "%s", cands[i][1]);
        return 1;
    }
    return 0;
}

static clip_bpe_t *bpe_load_paths(void) {
    char vpath[512], mpath[512];
    if (!bpe_find_paths(vpath, sizeof(vpath), mpath, sizeof(mpath))) {
        fprintf(stderr, "CLIP: tokenizer files not found (set CLIP_DIR or run from the repo with models/clip/)\n");
        return NULL;
    }
    return bpe_load(vpath, mpath);
}

static clip_bpe_t *bpe_load(const char *vocab_path, const char *merges_path) {
    clip_bpe_t *t = (clip_bpe_t *)calloc(1, sizeof(clip_bpe_t));
    if (!t) return NULL;
    bpe_bytes_to_unicode(t->byte_to_char);
    fprintf(stderr, "[bpe] unicode table done\n");

    /* vocab.json: token -> id. We need id -> string, so invert. */
    FILE *f = fopen(vocab_path, "rb");
    if (!f) { free(t); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); free(t); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = 0; fclose(f);

    t->vocab = (char **)calloc(CLIP_VOCAB, sizeof(char *));
    if (!t->vocab) { free(buf); free(t); return NULL; }
    /* scan JSON {"tok":id,...} -- proper string parsing with \" escapes */
    char *p = buf;
    while ((p = strchr(p, '"')) != NULL) {
        p++;  /* skip opening quote */
        /* read JSON string with escape handling */
        char *tok = p;
        size_t len = 0;
        while (*p && *p != '"') {
            if (*p == '\\' && p[1]) p++;  /* skip escaped char (\" \\ \/ etc.) */
            p++; len++;
        }
        if (!*p) break;                    /* unterminated */
        char *str = (char *)malloc(len + 1);
        if (!str) break;
        /* unescape into str */
        size_t o = 0;
        for (char *q = tok; q < p; q++) {
            if (*q == '\\' && q + 1 < p) {
                q++;
                if (*q == 'n') str[o++] = '\n';
                else if (*q == 't') str[o++] = '\t';
                else str[o++] = *q;        /* \" \\ \/ \u handled as-is (ASCII vocab) */
            } else str[o++] = *q;
        }
        str[o] = 0;
        p++;  /* skip closing quote */
        char *colon = strchr(p, ':');
        if (!colon) { free(str); break; }
        int id = atoi(colon + 1);
        if (id >= 0 && id < CLIP_VOCAB) {
            free(t->vocab[id]);
            t->vocab[id] = str;
        } else free(str);
        p = colon + 1;
    }
    free(buf);

    /* merges.txt: one "a b" per line, rank = line index */
    f = fopen(merges_path, "rb");
    if (!f) { bpe_free_dummy(t); return NULL; }
    fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
    buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); bpe_free_dummy(t); return NULL; }
    rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = 0; fclose(f);

    t->n_merges = 0;
    char *line = buf;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        if (*line && line[0] != '#' && strchr(line, ' '))
            t->n_merges++;
        line = nl ? nl + 1 : NULL;
    }

    /* Re-read the file: the count loop wrote '\0' over every '\n', so the
     * same buffer can no longer be line-split. Both loops must see the
     * same bytes. */
    f = fopen(merges_path, "rb");
    if (!f) { bpe_free_dummy(t); return NULL; }
    fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
    free(buf);
    buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); bpe_free_dummy(t); return NULL; }
    rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = 0; fclose(f);
    t->merges = (char **)calloc(t->n_merges ? t->n_merges : 1, sizeof(char *));
    int mi = 0;
    line = buf;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        if (*line && line[0] != '#' && strchr(line, ' '))
            t->merges[mi++] = strdup(line);
        line = nl ? nl + 1 : NULL;
    }
    free(buf);
    return t;
}

/* find merge rank: merges.txt is RANK-ORDERED (line i = rank i), not
 * lexicographically sorted, so a linear scan is required. 48k entries is
 * fine (a few ms per encode). */
static int bpe_rank(const clip_bpe_t *t, const char *pair) {
    for (int i = 0; i < t->n_merges; i++)
        if (strcmp(t->merges[i], pair) == 0) return i;
    return -1;
}

static void bpe_free_dummy(clip_bpe_t *t) {
    if (!t) return;
    if (t->vocab) {
        for (int i = 0; i < CLIP_VOCAB; i++) free(t->vocab[i]);
        free(t->vocab);
    }
    if (t->merges) {
        for (int i = 0; i < t->n_merges; i++) free(t->merges[i]);
        free(t->merges);
    }
    free(t);
}

/* CLIP token split: mimics the reference regex (after lowercase+ws-clean)
 *   's|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^ a-z0-9]+
 * Fills starts[]/lens[] (pointers into the input), returns count.
 * NOTE: [0-9] matches a SINGLE digit per the reference regex. */
static int clip_token_split(const char *s, const char **starts, int *lens, int max_out) {
    static const char *const conts[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    int n = 0;
    const char *p = s;
    while (*p && n < max_out) {
        if (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') { p++; continue; }
        const char *start = p;
        int matched = 0;
        /* contractions first (order matters: try 's before 't etc.) */
        if (*p == '\'') {
            for (int c = 0; c < 7; c++) {
                size_t cl = strlen(conts[c]);
                if (strncmp(p, conts[c], cl) == 0) { p += (int)cl; matched = 1; break; }
            }
        }
        if (!matched) {
            if ((*p >= 'a' && *p <= 'z')) {
                while (*p >= 'a' && *p <= 'z') p++;
            } else if (*p >= '0' && *p <= '9') {
                p++;  /* single digit */
            } else {
                while (*p && *p != ' ' &&
                       !(*p >= 'a' && *p <= 'z') &&
                       !(*p >= '0' && *p <= '9')) p++;
            }
        }
        starts[n] = start;
        lens[n] = (int)(p - start);
        n++;
    }
    return n;
}

/* BPE encode one token (already split by clip_token_split). The last
 * character gets the end-of-word suffix "</w>" before merging — this is
 * REQUIRED for correct CLIP ids (a word without it merges differently). */
static void bpe_encode_word(const clip_bpe_t *t, const char *word, int wlen,
                            int *ids, int *n_out) {
    /* split into byte-unicode chars */
    char **syms = (char **)calloc((size_t)wlen + 1, sizeof(char *));
    int n = 0;
    for (int i = 0; i < wlen; i++)
        syms[n++] = strdup(t->byte_to_char[(unsigned char)word[i]]);
    if (n == 0) { free(syms); *n_out = 0; return; }

    /* append </w> to the LAST symbol */
    {
        size_t l = strlen(syms[n - 1]);
        char *ns = (char *)malloc(l + 5);
        memcpy(ns, syms[n - 1], l);
        memcpy(ns + l, "</w>", 5);
        free(syms[n - 1]);
        syms[n - 1] = ns;
    }

    while (n > 1) {
        /* find lowest-rank adjacent pair */
        int best_rank = -1, best_i = -1;
        char pair[128];
        for (int i = 0; i < n - 1; i++) {
            snprintf(pair, sizeof(pair), "%s %s", syms[i], syms[i + 1]);
            int r = bpe_rank(t, pair);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) {
                best_rank = r; best_i = i;
            }
        }
        if (best_rank < 0) break;
        /* merge syms[best_i] + syms[best_i+1] */
        size_t a = strlen(syms[best_i]), b = strlen(syms[best_i + 1]);
        char *merged = (char *)malloc(a + b + 1);
        memcpy(merged, syms[best_i], a);
        memcpy(merged + a, syms[best_i + 1], b);
        merged[a + b] = 0;
        free(syms[best_i]); free(syms[best_i + 1]);
        syms[best_i] = merged;
        for (int j = best_i + 1; j < n - 1; j++) syms[j] = syms[j + 1];
        n--;
    }

    /* lookup each symbol in vocab */
    *n_out = 0;
    for (int i = 0; i < n; i++) {
        int id = -1;
        for (int v = 0; v < CLIP_VOCAB; v++) {
            if (t->vocab[v] && strcmp(t->vocab[v], syms[i]) == 0) { id = v; break; }
        }
        if (id >= 0) ids[(*n_out)++] = id;
        free(syms[i]);
    }
    free(syms);
}

/* full prompt -> 77 ids: [BOS] tokens... [EOS] pad-with-EOS.
 * Matches sd.cpp's CLIPTokenizer: normalize (lowercase + whitespace-clean),
 * token_split regex, byte-level BPE with </w> suffix, BOS/EOS padding. */
static int bpe_encode(const clip_bpe_t *t, const char *prompt, int *out_ids) {
    out_ids[0] = CLIP_BOS;
    int n = 1;
    /* normalize: lowercase ASCII + collapse whitespace to single spaces */
    char norm[2048];
    size_t ni = 0;
    int prev_space = 1;  /* strip leading whitespace */
    for (const unsigned char *c = (const unsigned char *)prompt; *c; c++) {
        if (*c == ' ' || *c == '\t' || *c == '\n' || *c == '\r' ||
            *c == '\v' || *c == '\f') {
            if (!prev_space && ni < sizeof(norm) - 1) norm[ni++] = ' ';
            prev_space = 1;
        } else {
            unsigned char ch = (*c >= 'A' && *c <= 'Z') ? (unsigned char)(*c + 32) : *c;
            if (ni < sizeof(norm) - 1) norm[ni++] = (char)ch;
            prev_space = 0;
        }
    }
    while (ni > 0 && norm[ni - 1] == ' ') ni--;  /* strip trailing space */
    norm[ni] = 0;

    const char *toks[512];
    int tlens[512];
    int ntok = clip_token_split(norm, toks, tlens, 512);
    for (int i = 0; i < ntok && n < CLIP_CTX - 1; i++) {
        int wid[128], wn = 0;
        bpe_encode_word(t, toks[i], tlens[i], wid, &wn);
        for (int j = 0; j < wn && n < CLIP_CTX - 1; j++)
            out_ids[n++] = wid[j];
    }
    out_ids[n++] = CLIP_EOS;
    while (n < CLIP_CTX) out_ids[n++] = CLIP_EOS;
    return n;
}

/* ---------------- the encoder ---------------- */

struct wubu_sd_clip {
    gguf_ctx *gguf;
    /* lazy F32 caches: name -> buffer */
    char **cache_names;
    float **cache_data;
    int64_t *cache_elems;
    int cache_n, cache_cap;
    clip_bpe_t *bpe;
};

/* raw (quantized, never-dequantized) weight descriptor */
typedef struct {
    const void *ptr;  /* blob + data_offset */
    int type;         /* GGML_TYPE_* */
    int64_t K;        /* dims[0] (contraction, innermost) */
    int64_t N;        /* dims[1] (columns) */
    int64_t n_elems;
    int found;
} clip_raw_t;

static clip_raw_t clip_get_raw(wubu_sd_clip_t *c, const char *name) {
    clip_raw_t r = {0};
    gguf_tensor_info *ti = gguf_find_tensor(c->gguf, name);
    if (!ti) { fprintf(stderr, "CLIP: missing tensor %s\n", name); return r; }
    r.ptr = (const uint8_t *)c->gguf->data_blob + ti->data_offset;
    r.type = ti->ggml_type;
    r.n_elems = 1;
    for (int d = 0; d < ti->n_dims; d++) r.n_elems *= ti->dims[d];
    r.K = ti->n_dims > 0 ? ti->dims[0] : 1;
    r.N = ti->n_dims > 1 ? ti->dims[1] : 1;
    r.found = 1;
    return r;
}
static float *clip_get_f32(wubu_sd_clip_t *c, const char *name, int64_t *n_elems) {
    for (int i = 0; i < c->cache_n; i++)
        if (strcmp(c->cache_names[i], name) == 0) {
            if (n_elems) *n_elems = c->cache_elems[i];
            return c->cache_data[i];
        }
    gguf_tensor_info *ti = gguf_find_tensor(c->gguf, name);
    if (!ti) { fprintf(stderr, "CLIP: missing tensor %s\n", name); return NULL; }
    int64_t ne = 1;
    for (int d = 0; d < ti->n_dims; d++) ne *= ti->dims[d];
    float *f = (float *)malloc((size_t)ne * sizeof(float));
    if (!f) return NULL;
    if (gguf_read_tensor_f32(c->gguf, ti, f, ne) != (int)ne) {
        fprintf(stderr, "CLIP: failed to read %s\n", name);
        free(f); return NULL;
    }
    if (c->cache_n == c->cache_cap) {
        c->cache_cap = c->cache_cap ? c->cache_cap * 2 : 16;
        c->cache_names = (char **)realloc(c->cache_names, c->cache_cap * sizeof(char *));
        c->cache_data = (float **)realloc(c->cache_data, c->cache_cap * sizeof(float *));
        c->cache_elems = (int64_t *)realloc(c->cache_elems, c->cache_cap * sizeof(int64_t));
    }
    c->cache_names[c->cache_n] = strdup(name);
    c->cache_data[c->cache_n] = f;
    c->cache_elems[c->cache_n] = ne;
    c->cache_n++;
    if (n_elems) *n_elems = ne;
    return f;
}

static float quick_gelu(float x) {
    /* x * sigmoid(1.702 x) */
    return x / (1.0f + expf(-1.702f * x));
}

/* causal self-attention, 77 tokens, 12 heads x 64. x in [77, 768].
 * q/k/v/o weights are RAW blob pointers (F16 or Q4_0) — quantized matmul. */
static void clip_attn(const float *x, const clip_raw_t *q_w, const float *q_b,
                      const clip_raw_t *k_w, const float *k_b,
                      const clip_raw_t *v_w, const float *v_b,
                      const clip_raw_t *o_w, const float *o_b,
                      float *out) {
    const int T = CLIP_CTX;
    const float scale = 1.0f / sqrtf((float)CLIP_HEAD_D);
    float *q = (float *)malloc((size_t)T * CLIP_DMODEL * sizeof(float));
    float *k = (float *)malloc((size_t)T * CLIP_DMODEL * sizeof(float));
    float *v = (float *)malloc((size_t)T * CLIP_DMODEL * sizeof(float));
    /* projections via the quantized linear (raw blob weights) */
    wubu_sd_linear_qb(x, q_w->ptr, q_w->type, q_b, CLIP_CTX, (int)q_w->K, (int)q_w->N, q);
    wubu_sd_linear_qb(x, k_w->ptr, k_w->type, k_b, CLIP_CTX, (int)k_w->K, (int)k_w->N, k);
    wubu_sd_linear_qb(x, v_w->ptr, v_w->type, v_b, CLIP_CTX, (int)v_w->K, (int)v_w->N, v);
    /* NOTE: biases are applied INSIDE wubu_sd_linear_qb — the old
     * post-loop bias add must NOT remain (double-bias bug). */
    /* attention per head with causal mask */
    float *att = (float *)malloc((size_t)T * T * sizeof(float));
    for (int h = 0; h < CLIP_HEADS; h++) {
        #pragma omp parallel for
        for (int t = 0; t < T; t++) {
            for (int s = 0; s < T; s++) {
                float a = 0;
                if (s <= t) {
                    for (int d = 0; d < CLIP_HEAD_D; d++)
                        a += q[(size_t)t * CLIP_DMODEL + (size_t)h * CLIP_HEAD_D + d]
                           * k[(size_t)s * CLIP_DMODEL + (size_t)h * CLIP_HEAD_D + d];
                    a *= scale;
                } else {
                    a = -INFINITY;
                }
                att[(size_t)t * T + s] = a;
            }
            /* softmax row t */
            float mx = -INFINITY;
            for (int s = 0; s <= t; s++) if (att[(size_t)t * T + s] > mx) mx = att[(size_t)t * T + s];
            float sum = 0;
            for (int s = 0; s <= t; s++) { att[(size_t)t * T + s] = expf(att[(size_t)t * T + s] - mx); sum += att[(size_t)t * T + s]; }
            for (int s = 0; s <= t; s++) att[(size_t)t * T + s] /= sum;
        }
    }
    /* out = att @ v per head */
    float *ctx = (float *)calloc((size_t)T * CLIP_DMODEL, sizeof(float));
    #pragma omp parallel for
    for (int t = 0; t < T; t++)
        for (int h = 0; h < CLIP_HEADS; h++)
            for (int s = 0; s <= t; s++) {
                float w_ = att[(size_t)t * T + s];
                const float *vr = v + (size_t)s * CLIP_DMODEL + (size_t)h * CLIP_HEAD_D;
                float *or_ = ctx + (size_t)t * CLIP_DMODEL + (size_t)h * CLIP_HEAD_D;
                for (int d = 0; d < CLIP_HEAD_D; d++) or_[d] += w_ * vr[d];
            }
    /* o_proj (quantized) */
    wubu_sd_linear_qb(ctx, o_w->ptr, o_w->type, o_b, CLIP_CTX, (int)o_w->K, (int)o_w->N, out);
    free(q); free(k); free(v); free(att); free(ctx);
}

wubu_sd_clip_t *wubu_sd_clip_load(void *ctx) {
    wubu_sd_clip_t *c = (wubu_sd_clip_t *)calloc(1, sizeof(wubu_sd_clip_t));
    if (!c) return NULL;
    c->gguf = (gguf_ctx *)ctx;
    /* mmap the data blob — the quantized path reads weights straight from
     * it (never dequantizes). Without it, raw pointers are NULL. */
    {
        gguf_ctx *g = (gguf_ctx *)ctx;
        if (!g->data_blob) gguf_buffer_data(g);
    }
    c->bpe = bpe_load_paths();
    if (!c->bpe) {
        fprintf(stderr, "CLIP: tokenizer load failed (need /tmp/clip/vocab.json + merges.txt)\n");
        free(c); return NULL;
    }
    return c;
}

/* release the weight cache (keep the tokenizer) — for phase-separated
 * pipelines on RAM-tight boxes (CLIP encode, then free before UNet). */
void wubu_sd_clip_clear_cache(wubu_sd_clip_t *c) {
    if (!c) return;
    for (int i = 0; i < c->cache_n; i++) {
        free(c->cache_names[i]);
        free(c->cache_data[i]);
    }
    free(c->cache_names); free(c->cache_data); free(c->cache_elems);
    c->cache_names = NULL; c->cache_data = NULL; c->cache_elems = NULL;
    c->cache_n = c->cache_cap = 0;
}

void wubu_sd_clip_free(wubu_sd_clip_t *c) {
    if (!c) return;
    if (c->bpe) bpe_free_dummy(c->bpe);
    for (int i = 0; i < c->cache_n; i++) {
        free(c->cache_names[i]);
        free(c->cache_data[i]);
    }
    free(c->cache_names); free(c->cache_data); free(c->cache_elems);
    free(c);
}

int wubu_sd_clip_encode(wubu_sd_clip_t *c, const char *prompt,
                        float *seq_out /* [77][768] */,
                        float *pooled_out /* [768] */) {
    double t0 = omp_get_wtime();
    int ids[CLIP_CTX];
    bpe_encode(c->bpe, prompt, ids);
    fprintf(stderr, "[clip] tokenize %.2fs\n", omp_get_wtime() - t0);

    /* embeddings — token is Q4_0, pos is Q4_0 (dims [768, N] column-major).
     * Gather + dequant per element, no F32 copy of the 75MB table. */
    clip_raw_t tok = clip_get_raw(c, "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight");
    clip_raw_t pos = clip_get_raw(c, "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight");
    if (!tok.found || !pos.found) return -1;
    fprintf(stderr, "[clip] embed weights %.2fs\n", omp_get_wtime() - t0);

    /* hidden [77, 768] = token_emb[ids[t]] + pos_emb[t] */
    float *hidden = (float *)malloc((size_t)CLIP_CTX * CLIP_DMODEL * sizeof(float));
    float *work = (float *)malloc((size_t)CLIP_CTX * CLIP_DMODEL * sizeof(float));
    for (int t = 0; t < CLIP_CTX; t++) {
        float *hr = hidden + (size_t)t * CLIP_DMODEL;
        /* token embedding row for ids[t] (K=768 per column) */
        if (tok.type == 2) { /* Q4_0: 24 blocks x 18B */
            const uint8_t *col = (const uint8_t *)tok.ptr + (size_t)ids[t] * 24 * 18;
            for (int b = 0; b < 24; b++) {
                const uint8_t *blk = col + (size_t)b * 18;
                float d = wubu_sd_f16_to_f32(*(const uint16_t *)blk);
                const uint8_t *qs = blk + 2;
                for (int l = 0; l < 32; l++) {
                    int8_t q = (int8_t)((qs[l >> 1] >> (4 * (l & 1))) & 0xF) - 8;
                    hr[b * 32 + l] = (float)q * d;
                }
            }
        } else if (tok.type == 1) { /* F16 */
            const uint16_t *te = (const uint16_t *)tok.ptr + (size_t)ids[t] * CLIP_DMODEL;
            for (int d = 0; d < CLIP_DMODEL; d++) hr[d] = wubu_sd_f16_to_f32(te[d]);
        } else { fprintf(stderr, "CLIP: token emb type %d unsupported\n", tok.type); return -1; }
        /* position embedding row t */
        if (pos.type == 2) {
            const uint8_t *col = (const uint8_t *)pos.ptr + (size_t)t * 24 * 18;
            for (int b = 0; b < 24; b++) {
                const uint8_t *blk = col + (size_t)b * 18;
                float d = wubu_sd_f16_to_f32(*(const uint16_t *)blk);
                const uint8_t *qs = blk + 2;
                for (int l = 0; l < 32; l++) {
                    int8_t q = (int8_t)((qs[l >> 1] >> (4 * (l & 1))) & 0xF) - 8;
                    hr[b * 32 + l] += (float)q * d;
                }
            }
        } else if (pos.type == 1) {
            const uint16_t *pe = (const uint16_t *)pos.ptr + (size_t)t * CLIP_DMODEL;
            for (int d = 0; d < CLIP_DMODEL; d++) hr[d] += wubu_sd_f16_to_f32(pe[d]);
        } else { fprintf(stderr, "CLIP: pos emb type %d unsupported\n", pos.type); return -1; }
    }

    char nm[192];
    for (int l = 0; l < CLIP_LAYERS; l++) {
        double tL = omp_get_wtime();
        /* DEBUG: dump layer-0 input for parity check */
        if (l == 0 && getenv("CLIP_DEBUG"))
            fprintf(stderr, "[clip-dbg] layer0 hidden[0][0..3] = %.5f %.5f %.5f %.5f\n",
                    hidden[0], hidden[1], hidden[2], hidden[3]);
        /* layer_norm1 -> attn -> residual */
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.layer_norm1.weight", l);
        float *ln1_w = clip_get_f32(c, nm, NULL);
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.layer_norm1.bias", l);
        float *ln1_b = clip_get_f32(c, nm, NULL);
        /* layer norm over the 768 feature dim, per token */
        for (int t = 0; t < CLIP_CTX; t++) {
            float mean = 0, var = 0;
            const float *hr = hidden + (size_t)t * CLIP_DMODEL;
            for (int d = 0; d < CLIP_DMODEL; d++) { mean += hr[d]; var += hr[d] * hr[d]; }
            mean /= CLIP_DMODEL;
            var = var / CLIP_DMODEL - mean * mean;
            float inv = 1.0f / sqrtf(var + 1e-5f);
            for (int d = 0; d < CLIP_DMODEL; d++)
                work[(size_t)t * CLIP_DMODEL + d] = (hr[d] - mean) * inv * ln1_w[d] + (ln1_b ? ln1_b[d] : 0);
        }
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.self_attn.q_proj.weight", l);
        clip_raw_t qw = clip_get_raw(c, nm);
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.self_attn.q_proj.bias", l);
        float *qb = clip_get_f32(c, nm, NULL);
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.self_attn.k_proj.weight", l);
        clip_raw_t kw = clip_get_raw(c, nm);
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.self_attn.k_proj.bias", l);
        float *kb = clip_get_f32(c, nm, NULL);
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.self_attn.v_proj.weight", l);
        clip_raw_t vw = clip_get_raw(c, nm);
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.self_attn.v_proj.bias", l);
        float *vb = clip_get_f32(c, nm, NULL);
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.self_attn.out_proj.weight", l);
        clip_raw_t ow = clip_get_raw(c, nm);
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.self_attn.out_proj.bias", l);
        float *ob = clip_get_f32(c, nm, NULL);
        if (!qw.found || !kw.found || !vw.found || !ow.found) { free(hidden); free(work); return -1; }
        clip_attn(work, &qw, qb, &kw, kb, &vw, vb, &ow, ob, work);
        for (int i = 0; i < CLIP_CTX * CLIP_DMODEL; i++) hidden[i] += work[i];

        /* layer_norm2 -> mlp -> residual */
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.layer_norm2.weight", l);
        float *ln2_w = clip_get_f32(c, nm, NULL);
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.layer_norm2.bias", l);
        float *ln2_b = clip_get_f32(c, nm, NULL);
        for (int t = 0; t < CLIP_CTX; t++) {
            float mean = 0, var = 0;
            const float *hr = hidden + (size_t)t * CLIP_DMODEL;
            for (int d = 0; d < CLIP_DMODEL; d++) { mean += hr[d]; var += hr[d] * hr[d]; }
            mean /= CLIP_DMODEL;
            var = var / CLIP_DMODEL - mean * mean;
            float inv = 1.0f / sqrtf(var + 1e-5f);
            for (int d = 0; d < CLIP_DMODEL; d++)
                work[(size_t)t * CLIP_DMODEL + d] = (hr[d] - mean) * inv * ln2_w[d] + (ln2_b ? ln2_b[d] : 0);
        }
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.mlp.fc1.weight", l);
        clip_raw_t fc1w = clip_get_raw(c, nm);
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.mlp.fc1.bias", l);
        float *fc1b = clip_get_f32(c, nm, NULL);
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.mlp.fc2.weight", l);
        clip_raw_t fc2w = clip_get_raw(c, nm);
        snprintf(nm, sizeof(nm), "cond_stage_model.transformer.text_model.encoder.layers.%d.mlp.fc2.bias", l);
        float *fc2b = clip_get_f32(c, nm, NULL);
        if (!fc1w.found || !fc2w.found) { free(hidden); free(work); return -1; }
        /* fc1: [77,768] x [768,3072] -> quick_gelu -> fc2: [77,3072] x [3072,768]
         * — quantized (raw blob) matmuls */
        float *mlp = (float *)malloc((size_t)CLIP_CTX * CLIP_MLP * sizeof(float));
        wubu_sd_linear_qb(work, fc1w.ptr, fc1w.type, fc1b, CLIP_CTX, (int)fc1w.K, (int)fc1w.N, mlp);
        #pragma omp parallel for
        for (int i = 0; i < CLIP_CTX * CLIP_MLP; i++) mlp[i] = quick_gelu(mlp[i]);
        wubu_sd_linear_qb(mlp, fc2w.ptr, fc2w.type, fc2b, CLIP_CTX, (int)fc2w.K, (int)fc2w.N, work);
        free(mlp);
        for (int i = 0; i < CLIP_CTX * CLIP_DMODEL; i++) hidden[i] += work[i];
        fprintf(stderr, "[clip] layer %d: %.2fs\n", l, omp_get_wtime() - tL);
    }

    /* final_layer_norm */
    float *fln_w = clip_get_f32(c, "cond_stage_model.transformer.text_model.final_layer_norm.weight", NULL);
    float *fln_b = clip_get_f32(c, "cond_stage_model.transformer.text_model.final_layer_norm.bias", NULL);
    if (!fln_w) { free(hidden); free(work); return -1; }
    for (int t = 0; t < CLIP_CTX; t++) {
        float mean = 0, var = 0;
        const float *hr = hidden + (size_t)t * CLIP_DMODEL;
        for (int d = 0; d < CLIP_DMODEL; d++) { mean += hr[d]; var += hr[d] * hr[d]; }
        mean /= CLIP_DMODEL;
        var = var / CLIP_DMODEL - mean * mean;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int d = 0; d < CLIP_DMODEL; d++)
            work[(size_t)t * CLIP_DMODEL + d] = (hr[d] - mean) * inv * fln_w[d] + (fln_b ? fln_b[d] : 0);
    }

    /* pooled = last token (EOS) hidden */
    if (pooled_out)
        memcpy(pooled_out, work + (size_t)(CLIP_CTX - 1) * CLIP_DMODEL,
               CLIP_DMODEL * sizeof(float));
    /* full sequence for UNet cross-attn K/V */
    if (seq_out)
        memcpy(seq_out, work, (size_t)CLIP_CTX * CLIP_DMODEL * sizeof(float));
    free(hidden);
    free(work);
    return 0;
}
