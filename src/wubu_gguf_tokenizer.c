/* wubu_gguf_tokenizer.c — GGUF-embedded tokenizer extraction (C11).
 *
 * Walks the GGUF KV section (after magic/version/n_tensors/n_kv) and
 * captures the tokenizer arrays:
 *   tokenizer.ggml.tokens       arr[str]  — the vocab
 *   tokenizer.ggml.merges       arr[str]  — "left right" merge pairs
 *   tokenizer.ggml.eos_token_id u32
 *   tokenizer.ggml.bos_token_id u32
 *   tokenizer.ggml.padding_token_id u32
 * The caller (wubu_tokenizer.c) builds the real tokenizer from these.
 *
 * GGUF KV layout (see gguf_reader.c gguf_open):
 *   header: magic(4) version(4) n_tensors(8) n_kv(8)
 *   per KV: u64 key_len, key bytes, i32 type, value
 *   type 8  = string:  u64 len + bytes
 *   type 9  = array:   i32 elem_type + u64 count + elems
 *     elem_type 8 = str array: per elem u64 len + bytes
 *   type 4  = u32 (4 bytes), type 7 = bool (1 byte)
 */

#include "wubu_gguf_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_KV_STRING  8
#define GGUF_KV_ARRAY   9
#define GGUF_KV_U32     4
#define GGUF_KV_BOOL    7

static uint64_t read_u64_le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}
static uint32_t read_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static int32_t read_i32_le(const uint8_t *p) {
    return (int32_t)read_u32_le(p);
}
static int kv_match(const uint8_t *key, uint64_t key_len, const char *want) {
    size_t wl = strlen(want);
    return key_len == wl && memcmp(key, want, wl) == 0;
}

void wubu_gguf_tokenizer_free(wubu_gguf_tokenizer_data_t *d) {
    if (!d) return;
    if (d->tokens) { for (int i = 0; i < d->n_tokens; i++) free(d->tokens[i]); free(d->tokens); }
    if (d->merges) { for (int i = 0; i < d->n_merges; i++) free(d->merges[i]); free(d->merges); }
    memset(d, 0, sizeof(*d));
}

bool wubu_gguf_tokenizer_extract(const char *gguf_path,
                                 wubu_gguf_tokenizer_data_t *out) {
    memset(out, 0, sizeof(*out));
    out->bos_id = out->eos_id = out->pad_id = -1;
    if (!gguf_path) return false;

    FILE *f = fopen(gguf_path, "rb");
    if (!f) { fprintf(stderr, "gguf_tokenizer: cannot open %s\n", gguf_path); return false; }

    uint8_t hdr[24];
    if (fread(hdr, 1, 24, f) != 24) { fclose(f); return false; }
    int64_t n_kv = (int64_t)read_u64_le(hdr + 16);

    for (int64_t i = 0; i < n_kv; i++) {
        uint8_t kl[8]; if (fread(kl, 1, 8, f) != 8) goto bad;
        uint64_t key_len = read_u64_le(kl);
        char *key = (char *)malloc((size_t)key_len + 1);
        if (!key) goto bad;
        if (fread(key, 1, key_len, f) != key_len) { free(key); goto bad; }
        key[key_len] = '\0';

        uint8_t tb[4]; if (fread(tb, 1, 4, f) != 4) { free(key); goto bad; }
        int32_t typ = read_i32_le(tb);

        if (typ == GGUF_KV_STRING) {
            uint8_t sl[8]; if (fread(sl, 1, 8, f) != 8) { free(key); goto bad; }
            if (fseek(f, (long)read_u64_le(sl), SEEK_CUR) != 0) { free(key); goto bad; }
        } else if (typ == GGUF_KV_ARRAY) {
            uint8_t atb[4]; if (fread(atb, 1, 4, f) != 4) { free(key); goto bad; }
            int32_t arr_type = read_i32_le(atb);
            uint8_t alb[8]; if (fread(alb, 1, 8, f) != 8) { free(key); goto bad; }
            uint64_t arr_len = read_u64_le(alb);
            if (arr_type == GGUF_KV_STRING) {
                char ***dst = NULL; int *dst_n = NULL;
                if (kv_match((const uint8_t *)key, key_len, "tokenizer.ggml.tokens")) {
                    dst = &out->tokens; dst_n = &out->n_tokens;
                } else if (kv_match((const uint8_t *)key, key_len, "tokenizer.ggml.merges")) {
                    dst = &out->merges; dst_n = &out->n_merges;
                }
                if (dst) {
                    *dst = (char **)calloc((size_t)arr_len + 1, sizeof(char *));
                    if (!*dst) { free(key); goto bad; }
                    for (uint64_t j = 0; j < arr_len; j++) {
                        uint8_t sl[8]; if (fread(sl, 1, 8, f) != 8) { free(key); goto bad; }
                        uint64_t slen = read_u64_le(sl);
                        char *s = (char *)malloc((size_t)slen + 1);
                        if (!s) { free(key); goto bad; }
                        if (fread(s, 1, slen, f) != slen) { free(s); free(key); goto bad; }
                        s[slen] = '\0';
                        (*dst)[j] = s;
                    }
                    *dst_n = (int)arr_len;
                } else {
                    for (uint64_t j = 0; j < arr_len; j++) {
                        uint8_t sl[8]; if (fread(sl, 1, 8, f) != 8) { free(key); goto bad; }
                        if (fseek(f, (long)read_u64_le(sl), SEEK_CUR) != 0) { free(key); goto bad; }
                    }
                }
            } else {
                int elem_size = 4;
                if (arr_type == 0 || arr_type == 1 || arr_type == 7) elem_size = 1;
                else if (arr_type == 2 || arr_type == 3) elem_size = 2;
                else if (arr_type == 10 || arr_type == 11 || arr_type == 12) elem_size = 8;
                if (fseek(f, (long)(arr_len * elem_size), SEEK_CUR) != 0) { free(key); goto bad; }
            }
        } else if (typ == GGUF_KV_U32) {
            uint8_t vb[4]; if (fread(vb, 1, 4, f) != 4) { free(key); goto bad; }
            int32_t v = (int32_t)read_u32_le(vb);
            if (kv_match((const uint8_t *)key, key_len, "tokenizer.ggml.eos_token_id")) out->eos_id = v;
            else if (kv_match((const uint8_t *)key, key_len, "tokenizer.ggml.bos_token_id")) out->bos_id = v;
            else if (kv_match((const uint8_t *)key, key_len, "tokenizer.ggml.padding_token_id")) out->pad_id = v;
        } else if (typ == GGUF_KV_BOOL) {
            if (fseek(f, 1, SEEK_CUR) != 0) { free(key); goto bad; }
        } else {
            int sz = (typ == 0 || typ == 1) ? 1 : (typ == 2 || typ == 3) ? 2 :
                     (typ == 6 || typ == 5) ? 4 : 8;
            if (fseek(f, sz, SEEK_CUR) != 0) { free(key); goto bad; }
        }
        free(key);
    }
    fclose(f);

    if (!out->tokens || out->n_tokens <= 0) {
        fprintf(stderr, "gguf_tokenizer: no tokenizer.ggml.tokens in %s\n", gguf_path);
        wubu_gguf_tokenizer_free(out);
        return false;
    }
    return true;

bad:
    fclose(f);
    wubu_gguf_tokenizer_free(out);
    return false;
}
