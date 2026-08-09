/*
 * wubu_tokenizer_hf.c -- HuggingFace BPE tokenizer loader for wubuwizard.
 * Self-contained: embeds a tiny, correct recursive-descent JSON scanner
 * (no external deps). The scanner advances a single forward cursor over the
 * whole file exactly once; it never re-scans. Hard safety cap prevents any
 * possibility of an infinite loop.
 *
 * Public API (see wubu_tokenizer_hf.h):
 *   wubu_tok_hf_load(path) / _encode(t,text,ids,max) / _decode(t,ids,n) /
 *   _free(t) / _bos_id / _eos_id / _vocab_size / _id_to_str
 */
#include "wubu_tokenizer_hf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* strdup is POSIX, not C11 -- under -std=c11 without _POSIX_C_SOURCE it
 * gets implicitly declared, returns int garbage, and every caller's
 * pointer is invalid (the WuBu port crashed here under ASan). The
 * no-third-party doctrine: provide it locally, freestanding-safe. */
static char *local_strdup(const char *s)
{
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *out = (char *)malloc(n);
    if (!out) return NULL;
    memcpy(out, s, n);
    return out;
}

#define MAX_OPS 200000000L   /* hard safety cap; real files finish in far less */

typedef struct {
    const char *p;   /* cursor (always advances forward) */
    const char *end;
    long ops;        /* global op counter for the safety cap */
} J;

/* ---- forward JSON scanner (single pass, depth-tracked) ---- */

static void j_ws(J *j) {
    while (j->p < j->end) {
        char c = *j->p;
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') j->p++;
        else break;
    }
}

/* read a JSON string at *p == '"'; returns malloc'd unescaped str, advances p. */
static char *j_str(J *j) {
    if (j->p >= j->end || *j->p != '"') return NULL;
    j->p++;
    size_t cap = 64, n = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;
    while (j->p < j->end && *j->p != '"') {
        char c = *j->p++;
        if (c == '\\') {
            if (j->p >= j->end) break;
            char e = *j->p++;
            switch (e) {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case '/': c = '/'; break;
                case '\\': c = '\\'; break;
                case '"': c = '"'; break;
                case 'u': {
                    int v = 0;
                    for (int i = 0; i < 4 && j->p < j->end; i++) {
                        char h = *j->p++;
                        v <<= 4;
                        if (h >= '0' && h <= '9') v |= h - '0';
                        else if (h >= 'a' && h <= 'f') v |= h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F') v |= h - 'A' + 10;
                    }
                    if (n + 4 >= cap) { cap *= 2; buf = (char *)realloc(buf, cap); }
                    if (v < 0x80) buf[n++] = (char)v;
                    else if (v < 0x800) {
                        buf[n++] = (char)(0xC0 | (v >> 6));
                        buf[n++] = (char)(0x80 | (v & 0x3F));
                    } else {
                        buf[n++] = (char)(0xE0 | (v >> 12));
                        buf[n++] = (char)(0x80 | ((v >> 6) & 0x3F));
                        buf[n++] = (char)(0x80 | (v & 0x3F));
                    }
                    continue;
                }
                default: c = e; break;
            }
        }
        if (n + 1 >= cap) { cap *= 2; buf = (char *)realloc(buf, cap); }
        buf[n++] = c;
    }
    if (j->p < j->end && *j->p == '"') j->p++;  /* consume closing quote */
    buf[n] = '\0';
    return buf;
}

/* true if a value (object/array/string/number) starts at j->p; skip it. */
static void j_skip(J *j) {
    j_ws(j);
    if (j->p >= j->end) return;
    char c = *j->p;
    if (c == '"') {
        /* skip the quoted string WITHOUT allocating (j_str mallocs —
         * discarding its result leaked 13×64B per load; fixed 2026-08-09) */
        j->p++;
        for (; j->p < j->end; j->p++) {
            if (*j->p == '\\') { if (j->p + 1 < j->end) j->p++; continue; }
            if (*j->p == '"') { j->p++; break; }
        }
        return;
    }
    if (c == '{' || c == '[') {
        char open = c, close = (c == '{') ? '}' : ']';
        int depth = 0;
        for (;;) {
            if (j->p >= j->end) break;
            char d = *j->p;
            if (d == '"') {
                /* same non-allocating string skip (leak #2) */
                j->p++;
                for (; j->p < j->end; j->p++) {
                    if (*j->p == '\\') { if (j->p + 1 < j->end) j->p++; continue; }
                    if (*j->p == '"') { j->p++; break; }
                }
            }
            else if (d == open) { depth++; j->p++; }
            else if (d == close) { depth--; j->p++; if (depth == 0) break; }
            else j->p++;
        }
        return;
    }
    /* number / true / false / null */
    while (j->p < j->end && *j->p != ',' && *j->p != '}' && *j->p != ']') j->p++;
}

/* Find the next child key equal to `name` in the current object scope,
 * leaving j->p at the value position. Returns malloc'd key (caller frees)
 * or NULL. j->p must be just inside an object (after '{'). */
static char *j_obj_find(J *j, const char *name) {
    for (;;) {
        j_ws(j);
        if (j->p >= j->end || *j->p == '}') return NULL;
        if (*j->p != '"') { j_skip(j); if (j->p < j->end && *j->p == ':') j_skip(j); if (j->p < j->end && *j->p == ',') j->p++; continue; }
        char *k = j_str(j);
        if (!k) return NULL;
        j_ws(j);
        if (j->p < j->end && *j->p == ':') j->p++;
        j_ws(j);
        if (strcmp(k, name) == 0) return k;  /* j->p now at value */
        j_skip(j);                            /* skip value */
        free(k);
        if (j->p < j->end && *j->p == ',') j->p++;
    }
}

/* ---- tokenizer state ---- */

struct wubu_tok_hf {
    char **vocab_str;     /* [id] */
    int vocab_size;
    char **merge_a, **merge_b;
    int n_merges;
    int bos_id, eos_id;
    /* the pair->rank hash table: the BPE merge loop is O(n^2) when it
     * scans all merges per adjacent pair; the hash makes each lookup
     * O(1). This is the difference between a 640MB corpus taking
     * hours and taking minutes. */
    uint32_t *pair_hash;   /* [PAIR_CAP] the key hash */
    int      *pair_rank;   /* [PAIR_CAP] the merge rank (-1 = empty) */
};

#define PAIR_CAP 65536
#define PAIR_EMPTY 0xFFFFFFFFu

static uint32_t pair_fnv(const char *a, const char *b)
{
    uint32_t h = 2166136261u;
    for (const char *p = a; *p; p++) { h ^= (uint8_t)*p; h *= 16777619u; }
    h ^= 0x1F3Du;
    for (const char *p = b; *p; p++) { h ^= (uint8_t)*p; h *= 16777619u; }
    return h;
}

/* insert a pair (rank) into the hash; collision -> linear probe. */
static void pair_insert(wubu_tok_hf_t *t, const char *a, const char *b, int rank)
{
    uint32_t h = pair_fnv(a, b);
    uint32_t slot = h & (PAIR_CAP - 1);
    while (t->pair_hash[slot] != PAIR_EMPTY &&
           t->pair_hash[slot] != h) {
        slot = (slot + 1) & (PAIR_CAP - 1);
    }
    t->pair_hash[slot] = h;
    t->pair_rank[slot] = rank;
}

/* look up a pair; returns the rank or -1. */
static int pair_lookup(const wubu_tok_hf_t *t, const char *a, const char *b)
{
    uint32_t h = pair_fnv(a, b);
    uint32_t slot = h & (PAIR_CAP - 1);
    while (t->pair_hash[slot] != PAIR_EMPTY) {
        if (t->pair_hash[slot] == h) return t->pair_rank[slot];
        slot = (slot + 1) & (PAIR_CAP - 1);
    }
    return -1;
}

/* byte-level BPE helpers (HuggingFace ByteLevel, exact bytes_to_unicode) */
static char *BYTE_TO_BL[256];    /* byte -> UTF-8 string (owned) */
static int   BL_TO_BYTE[258];    /* byte-level codepoint (0..257) -> original byte */
static int bl_ready = 0;

/* codepoint (>=256, the remapped control/space bytes) -> original byte */
static int bl_cp_to_byte(int cp) {
    if (cp >= 256 && cp < 256 + 256) return BL_TO_BYTE[cp - 256];
    return -1;
}

static void ensure_bl(void) {
    if (bl_ready) return;
    /* Step 1: build bytes_to_unicode exactly like HF tokenizers. */
    int bs[256]; int nbs = 0;
    for (int c = '!'; c <= '~'; c++) bs[nbs++] = c;
    for (int c = 0xA1; c <= 0xAC; c++) bs[nbs++] = c;
    for (int c = 0xAE; c <= 0xFF; c++) bs[nbs++] = c;
    int in_bs[256]; for (int i = 0; i < 256; i++) in_bs[i] = 0;
    for (int i = 0; i < nbs; i++) in_bs[bs[i]] = 1;
    /* cs[i] = unicode codepoint for byte i */
    int cs[256];
    for (int i = 0; i < nbs; i++) cs[bs[i]] = bs[i];   /* identity for printable */
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!in_bs[b]) { cs[b] = 256 + n; BL_TO_BYTE[n] = b; n++; }
    }
    /* Step 2: build BYTE_TO_BL[i] = UTF-8 encoding of cs[i] */
    for (int i = 0; i < 256; i++) {
        int cp = cs[i];
        char buf[5]; int k = 0;
        if (cp < 0x80) { buf[k++] = (char)cp; }
        else if (cp < 0x800) { buf[k++] = (char)(0xC0 | (cp >> 6)); buf[k++] = (char)(0x80 | (cp & 0x3F)); }
        else { buf[k++] = (char)(0xE0 | (cp >> 12)); buf[k++] = (char)(0x80 | ((cp >> 6) & 0x3F)); buf[k++] = (char)(0x80 | (cp & 0x3F)); }
        buf[k] = '\0';
        BYTE_TO_BL[i] = local_strdup(buf);
    }
    bl_ready = 1;
}

/* find vocab id for an exact string; returns id or -1 */
static int vocab_find(wubu_tok_hf_t *t, const char *s) {
    for (int i = 0; i < t->vocab_size; i++)
        if (t->vocab_str[i] && strcmp(t->vocab_str[i], s) == 0) return i;
    return -1;
}

/* apply BPE merges to a symbol list; returns new count.
 * The pair->rank hash makes each merge selection O(n) not O(n*merges). */
static int apply_bpe(wubu_tok_hf_t *t, char **syms, int n) {
    int changed = 1;
    while (changed) {
        changed = 0;
        int best_rank = -1, best_i = -1;
        for (int i = 0; i + 1 < n; i++) {
            int rank = pair_lookup(t, syms[i], syms[i + 1]);
            if (rank >= 0 && (best_rank < 0 || rank < best_rank)) { best_rank = rank; best_i = i; }
        }
        if (best_i < 0) break;
        size_t la = strlen(syms[best_i]), lb = strlen(syms[best_i + 1]);
        char *merged = (char *)malloc(la + lb + 1);
        memcpy(merged, syms[best_i], la);
        memcpy(merged + la, syms[best_i + 1], lb);
        merged[la + lb] = '\0';
        free(syms[best_i]); free(syms[best_i + 1]);
        syms[best_i] = merged;
        for (int i = best_i + 1; i + 1 < n; i++) syms[i] = syms[i + 1];
        n--;
        changed = 1;
    }
    return n;
}

/* ---- public API ---- */

wubu_tok_hf_t *wubu_tok_hf_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "tok_hf: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    fclose(f);
    buf[sz] = '\0';

    wubu_tok_hf_t *t = (wubu_tok_hf_t *)calloc(1, sizeof(wubu_tok_hf_t));
    if (!t) { free(buf); return NULL; }
    t->bos_id = -1; t->eos_id = -1;

    J j = { buf, buf + sz, 0 };
    j_ws(&j);
    if (j.p >= j.end || *j.p != '{') { free(t); free(buf); return NULL; }
    j.p++;  /* enter root object (scan forward only) */

    /* We need: model.vocab, model.merges, added_tokens[].content/id.
     * Scan root keys once; for "model" descend and pull vocab+merges. */
    int model_done = 0;
    for (;;) {
        if (j.ops++ > MAX_OPS) { free(t); free(buf); return NULL; }
        j_ws(&j);
        if (j.p >= j.end || *j.p == '}') break;
        if (*j.p != '"') { j_skip(&j); if (j.p < j.end && *j.p == ':') j_skip(&j); if (j.p < j.end && *j.p == ',') j.p++; continue; }
        char *key = j_str(&j);
        if (!key) break;
        j_ws(&j);
        if (j.p < j.end && *j.p == ':') j.p++;
        j_ws(&j);

        if (!model_done && strcmp(key, "model") == 0 && j.p < j.end && *j.p == '{') {
            j.p++;  /* enter model object */
            int vocab_done = 0, merges_done = 0;
            for (;;) {
                if (j.ops++ > MAX_OPS) { free(key); free(t); free(buf); return NULL; }
                j_ws(&j);
                if (j.p >= j.end || *j.p == '}') { if (j.p < j.end) j.p++; break; }
                if (*j.p != '"') { j_skip(&j); if (j.p < j.end && *j.p == ':') j_skip(&j); if (j.p < j.end && *j.p == ',') j.p++; continue; }
                char *mk = j_str(&j);
                if (!mk) break;
                j_ws(&j);
                if (j.p < j.end && *j.p == ':') j.p++;
                j_ws(&j);

                if (!vocab_done && strcmp(mk, "vocab") == 0 && j.p < j.end && *j.p == '{') {
                    j.p++;
                    size_t cap = 4096;
                    t->vocab_str = (char **)calloc(cap, sizeof(char *));
                    t->vocab_size = 0;
                    for (;;) {
                        if (j.ops++ > MAX_OPS) { free(mk); free(key); free(t); free(buf); return NULL; }
                        j_ws(&j);
                        if (j.p >= j.end || *j.p == '}') { if (j.p < j.end) j.p++; break; }
                        if (*j.p != '"') { j_skip(&j); if (j.p < j.end && *j.p == ':') j_skip(&j); if (j.p < j.end && *j.p == ',') j.p++; continue; }
                        char *tk = j_str(&j);
                        j_ws(&j);
                        if (j.p < j.end && *j.p == ':') j.p++;
                        j_ws(&j);
                        int id = 0;
                        while (j.p < j.end && *j.p >= '0' && *j.p <= '9') id = id * 10 + (*j.p++ - '0');
                        if (t->vocab_size >= (int)cap) { cap *= 2; t->vocab_str = (char **)realloc(t->vocab_str, cap * sizeof(char *)); }
                        t->vocab_str[t->vocab_size++] = tk;
                        if (j.p < j.end && *j.p == ',') j.p++;
                    }
                    vocab_done = 1;
                } else if (!merges_done && strcmp(mk, "merges") == 0 && j.p < j.end && *j.p == '[') {
                    j.p++;
                    t->merge_a = (char **)calloc(1, sizeof(char *));
                    t->merge_b = (char **)calloc(1, sizeof(char *));
                    t->n_merges = 0;
                    for (;;) {
                        if (j.ops++ > MAX_OPS) { free(mk); free(key); free(t); free(buf); return NULL; }
                        j_ws(&j);
                        if (j.p >= j.end || *j.p == ']') { if (j.p < j.end) j.p++; break; }
                        if (*j.p == '[') {
                            j.p++;
                            j_ws(&j);
                            char *a = (*j.p == '"') ? j_str(&j) : NULL;
                            j_ws(&j);
                            if (j.p < j.end && *j.p == ',') j.p++;
                            j_ws(&j);
                            char *b = (*j.p == '"') ? j_str(&j) : NULL;
                            j_ws(&j);
                            if (j.p < j.end && *j.p == ']') j.p++;
                            if (a && b) {
                                t->merge_a = (char **)realloc(t->merge_a, (t->n_merges + 1) * sizeof(char *));
                                t->merge_b = (char **)realloc(t->merge_b, (t->n_merges + 1) * sizeof(char *));
                                t->merge_a[t->n_merges] = a;
                                t->merge_b[t->n_merges] = b;
                                t->n_merges++;
                            } else { free(a); free(b); }
                        } else {
                            j_skip(&j);
                        }
                        if (j.p < j.end && *j.p == ',') j.p++;
                    }
                    merges_done = 1;
                } else {
                    j_skip(&j);
                }
                free(mk);
                if (j.p < j.end && *j.p == ',') j.p++;
            }
            model_done = 1;
        } else if (strcmp(key, "added_tokens") == 0 && j.p < j.end && *j.p == '[') {
            j.p++;
            for (;;) {
                if (j.ops++ > MAX_OPS) { free(key); free(t); free(buf); return NULL; }
                j_ws(&j);
                if (j.p >= j.end || *j.p == ']') { if (j.p < j.end) j.p++; break; }
                if (*j.p != '{') { j_skip(&j); if (j.p < j.end && *j.p == ',') j.p++; continue; }
                j.p++;  /* enter added token object */
                char *content = NULL; int id = -1;
                for (;;) {
                    j_ws(&j);
                    if (j.p >= j.end || *j.p == '}') { if (j.p < j.end) j.p++; break; }
                    if (*j.p != '"') { j_skip(&j); if (j.p < j.end && *j.p == ':') j_skip(&j); if (j.p < j.end && *j.p == ',') j.p++; continue; }
                    char *kk = j_str(&j);
                    j_ws(&j);
                    if (j.p < j.end && *j.p == ':') j.p++;
                    j_ws(&j);
                    if (strcmp(kk, "content") == 0 && j.p < j.end && *j.p == '"') content = j_str(&j);
                    else if (strcmp(kk, "id") == 0) { while (j.p < j.end && *j.p >= '0' && *j.p <= '9') id = id * 10 + (*j.p++ - '0'); }
                    else j_skip(&j);
                    free(kk);
                    if (j.p < j.end && *j.p == ',') j.p++;
                }
                if (content && id >= 0) {
                    /* register added token as a vocab entry (may override) */
                    int existing = vocab_find(t, content);
                    if (existing >= 0) {
                        free(t->vocab_str[existing]);
                        t->vocab_str[existing] = content;
                    } else {
                        size_t cap = (size_t)(t->vocab_size ? t->vocab_size * 2 : 4096);
                        if ((size_t)t->vocab_size + 1 > cap) { /* grow */ }
                        t->vocab_str = (char **)realloc(t->vocab_str, (t->vocab_size + 1) * sizeof(char *));
                        t->vocab_str[t->vocab_size++] = content;
                    }
                    if (content && (strcmp(content, "<|im_start|>") == 0 || strcmp(content, "<s>") == 0 || strcmp(content, "<|begin_of_text|>") == 0)) t->bos_id = id;
                    if (content && (strcmp(content, "<|im_end|>") == 0 || strcmp(content, "</s>") == 0 || strcmp(content, "<|end_of_text|>") == 0)) t->eos_id = id;
                } else {
                    free(content);
                }
                if (j.p < j.end && *j.p == ',') j.p++;
            }
        } else {
            j_skip(&j);
        }
        free(key);
        if (j.p < j.end && *j.p == ',') j.p++;
    }

    free(buf);
    ensure_bl();
    if (t->vocab_size == 0) { wubu_tok_hf_free(t); return NULL; }
    /* build the pair->rank hash: O(merges) once, O(1) per lookup. */
    t->pair_hash = (uint32_t *)calloc(PAIR_CAP, sizeof(uint32_t));
    t->pair_rank = (int *)calloc(PAIR_CAP, sizeof(int));
    if (!t->pair_hash || !t->pair_rank) { wubu_tok_hf_free(t); return NULL; }
    for (int i = 0; i < PAIR_CAP; i++) t->pair_hash[i] = PAIR_EMPTY;
    for (int i = 0; i < t->n_merges; i++)
        pair_insert(t, t->merge_a[i], t->merge_b[i], i);
    return t;
}

int wubu_tok_hf_encode(const wubu_tok_hf_t *t, const char *text, int *out_ids, int max_ids) {
    if (!t || !text) return 0;
    ensure_bl();
    /* byte-level pre-tokenize: each input byte -> byte-level symbol */
    char **syms = (char **)malloc((strlen(text) + 1) * sizeof(char *));
    int n = 0;
    for (const unsigned char *q = (const unsigned char *)text; *q; q++) {
        const char *bl = BYTE_TO_BL[*q];
        if (!bl) { static char tmp[2]; tmp[0] = (char)*q; tmp[1] = 0; bl = tmp; }
        syms[n++] = local_strdup(bl);
    }
    n = apply_bpe(t, syms, n);
    int cnt = 0;
    for (int i = 0; i < n && cnt < max_ids; i++) {
        int id = vocab_find(t, syms[i]);
        if (id >= 0) out_ids[cnt++] = id;
        free(syms[i]);
    }
    free(syms);
    return cnt;
}

char *wubu_tok_hf_decode(const wubu_tok_hf_t *t, const int *ids, int n) {
    if (!t) return NULL;
    ensure_bl();
    size_t cap = 4096, len = 0;
    char *out = (char *)malloc(cap);
    for (int i = 0; i < n; i++) {
        const char *s = (i < t->vocab_size && t->vocab_str[i]) ? t->vocab_str[ids[i]] : "";
        /* reverse ByteLevel: each vocab char is a byte-level unicode codepoint;
           map it back to the original byte. */
        for (const unsigned char *q = (const unsigned char *)s; *q; ) {
            int byte;
            if (*q < 0x80) { byte = *q; q += 1; }
            else if ((*q & 0xE0) == 0xC0) {
                int cp = ((*q & 0x1F) << 6) | (q[1] & 0x3F);
                int b = bl_cp_to_byte(cp);
                byte = (b >= 0) ? b : *q;   /* fallback: keep raw byte */
                q += 2;
            } else if ((*q & 0xF0) == 0xE0) {
                int cp = ((*q & 0x0F) << 12) | ((q[1] & 0x3F) << 6) | (q[2] & 0x3F);
                int b = bl_cp_to_byte(cp);
                byte = (b >= 0) ? b : *q;
                q += 3;
            } else { byte = *q; q += 1; }
            if (len + 1 >= cap) { cap = len + 4096; out = (char *)realloc(out, cap); }
            out[len++] = (char)byte;
        }
    }
    out[len] = '\0';
    return out;
}

void wubu_tok_hf_free(wubu_tok_hf_t *t) {
    if (!t) return;
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab_str[i]);
    free(t->vocab_str);
    for (int i = 0; i < t->n_merges; i++) { free(t->merge_a[i]); free(t->merge_b[i]); }
    free(t->merge_a); free(t->merge_b);
    free(t->pair_hash); free(t->pair_rank);
    free(t);
}

int wubu_tok_hf_bos_id(const wubu_tok_hf_t *t) { return t ? t->bos_id : -1; }
int wubu_tok_hf_eos_id(const wubu_tok_hf_t *t) { return t ? t->eos_id : -1; }
int wubu_tok_hf_vocab_size(const wubu_tok_hf_t *t) { return t ? t->vocab_size : 0; }
const char *wubu_tok_hf_id_to_str(const wubu_tok_hf_t *t, int id) {
    return (t && id >= 0 && id < t->vocab_size) ? t->vocab_str[id] : NULL;
}
