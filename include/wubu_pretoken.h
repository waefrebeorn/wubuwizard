#ifndef WUBU_PRETOKEN_H
#define WUBU_PRETOKEN_H

/*
 * wubu_pretoken.h -- EG-13: family registry keyed on tokenizer.ggml.pre
 * (llama.cpp llama-vocab lineage), reimplemented in C11.
 *
 * A registry maps a pre-tokenizer id to a deterministic splitter. Unknown
 * ids fail closed. A small NFC composition fallback handles decomposed
 * Latin accents so byte streams normalize the same way across families.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_pretoken wubu_pretoken_t;

wubu_pretoken_t *wubu_pretoken_create(int max_families);
void wubu_pretoken_free(wubu_pretoken_t *p);

/* Built-in families: "default", "llama3", "gpt2", "qwen2", "deepseek". */
int wubu_pretoken_register_builtin(wubu_pretoken_t *p, const char *family);
int wubu_pretoken_supported(const wubu_pretoken_t *p, const char *family);

/* Split text into pieces. offsets[i]/lens[i] index into text. Returns the
 * number of pieces or -1 on bad args/unknown family. */
int wubu_pretoken_split(const wubu_pretoken_t *p, const char *family,
                        const char *text, size_t len,
                        size_t *offsets, size_t *lens, int max_pieces);

/* Minimal NFC: compose a base ASCII letter followed by U+0301 (acute) or
 * U+0300 (grave) into the matching 2-byte UTF-8 codepoint. Other bytes are
 * copied unchanged. Returns output length or -1 if out_cap is too small. */
int wubu_pretoken_nfc(const char *in, size_t len, unsigned char *out,
                      size_t out_cap);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_PRETOKEN_H */
