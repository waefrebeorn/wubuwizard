/*
 * wubu_pretoken.c -- EG-13 family registry + minimal NFC. Pure C11.
 */
#include "wubu_pretoken.h"

#include <stdlib.h>
#include <string.h>

enum { F_DEFAULT = 0, F_LLAMA3, F_GPT2, F_QWEN2, F_DEEPSEEK, F_COUNT };

struct wubu_pretoken {
    int families[F_COUNT];
    int n;
};

static int family_id(const char *name) {
    if (!name) return -1;
    if (!strcmp(name, "default")) return F_DEFAULT;
    if (!strcmp(name, "llama3")) return F_LLAMA3;
    if (!strcmp(name, "gpt2")) return F_GPT2;
    if (!strcmp(name, "qwen2")) return F_QWEN2;
    if (!strcmp(name, "deepseek")) return F_DEEPSEEK;
    return -1;
}

wubu_pretoken_t *wubu_pretoken_create(int max_families) {
    wubu_pretoken_t *p;
    if (max_families < 1) return NULL;
    p = (wubu_pretoken_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n = max_families > F_COUNT ? F_COUNT : max_families;
    return p;
}

void wubu_pretoken_free(wubu_pretoken_t *p) { free(p); }

int wubu_pretoken_register_builtin(wubu_pretoken_t *p, const char *family) {
    int id;
    if (!p || (id = family_id(family)) < 0) return -1;
    if (p->families[id]) return 0;
    p->families[id] = 1;
    return 0;
}

int wubu_pretoken_supported(const wubu_pretoken_t *p, const char *family) {
    int id;
    if (!p || (id = family_id(family)) < 0) return 0;
    return p->families[id] ? 1 : 0;
}

static int is_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static int is_punct(unsigned char c) {
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}

/* 0 = skip, negative = single-character piece, positive = run class. */
static int char_class(unsigned char c, int id) {
    if (is_space(c)) return 0;
    if (id == F_DEFAULT) return 1;
    if (is_punct(c)) return -1;
    if (c >= '0' && c <= '9') {
        if (id == F_DEEPSEEK) return -2;
        if (id == F_QWEN2) return 2;
    }
    return 1;
}

int wubu_pretoken_split(const wubu_pretoken_t *p, const char *family,
                        const char *text, size_t len,
                        size_t *offsets, size_t *lens, int max_pieces) {
    int id, pieces = 0;
    size_t i = 0;
    if (!p || !text || !offsets || !lens || max_pieces <= 0) return -1;
    id = family_id(family);
    if (id < 0 || !p->families[id]) return -1;

    while (i < len && pieces < max_pieces) {
        int cls = char_class((unsigned char)text[i], id);
        size_t start = i;
        if (cls == 0) { i++; continue; }
        if (cls < 0) {
            i++;
        } else {
            while (i < len &&
                   char_class((unsigned char)text[i], id) == cls) i++;
        }
        offsets[pieces] = start;
        lens[pieces] = i - start;
        pieces++;
    }
    return pieces;
}

typedef struct { unsigned char base; unsigned char mark; unsigned char out[2]; } nfc_entry;
static const nfc_entry kNFC[] = {
    { 'a', 0x81, { 0xC3, 0xA1 } }, { 'a', 0x80, { 0xC3, 0xA0 } },
    { 'e', 0x81, { 0xC3, 0xA9 } }, { 'e', 0x80, { 0xC3, 0xA8 } },
    { 'i', 0x81, { 0xC3, 0xAD } }, { 'i', 0x80, { 0xC3, 0xAC } },
    { 'o', 0x81, { 0xC3, 0xB3 } }, { 'o', 0x80, { 0xC3, 0xB2 } },
    { 'u', 0x81, { 0xC3, 0xBA } }, { 'u', 0x80, { 0xC3, 0xB9 } },
    { 'A', 0x81, { 0xC3, 0x81 } }, { 'A', 0x80, { 0xC3, 0x80 } },
    { 'E', 0x81, { 0xC3, 0x89 } }, { 'E', 0x80, { 0xC3, 0x88 } },
    { 'I', 0x81, { 0xC3, 0x8D } }, { 'I', 0x80, { 0xC3, 0x8C } },
    { 'O', 0x81, { 0xC3, 0x93 } }, { 'O', 0x80, { 0xC3, 0x92 } },
    { 'U', 0x81, { 0xC3, 0x9A } }, { 'U', 0x80, { 0xC3, 0x99 } },
};

int wubu_pretoken_nfc(const char *in, size_t len, unsigned char *out,
                      size_t out_cap) {
    size_t i = 0, o = 0;
    if (!in || !out) return -1;
    while (i < len) {
        unsigned char c = (unsigned char)in[i];
        if (i + 2 < len && c < 128 && (unsigned char)in[i + 1] == 0xCC &&
            ((unsigned char)in[i + 2] == 0x81 || (unsigned char)in[i + 2] == 0x80)) {
            const nfc_entry *hit = NULL;
            for (size_t k = 0; k < sizeof(kNFC) / sizeof(kNFC[0]); k++) {
                if (kNFC[k].base == c && kNFC[k].mark == (unsigned char)in[i + 2]) {
                    hit = &kNFC[k];
                    break;
                }
            }
            if (hit) {
                if (o + 2 > out_cap) return -1;
                out[o++] = hit->out[0];
                out[o++] = hit->out[1];
                i += 3;
                continue;
            }
        }
        if (o + 1 > out_cap) return -1;
        out[o++] = c;
        i++;
    }
    return (int)o;
}
