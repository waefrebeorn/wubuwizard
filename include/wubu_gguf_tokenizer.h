#ifndef WUBU_GGUF_TOKENIZER_H
#define WUBU_GGUF_TOKENIZER_H

/* wubu_gguf_tokenizer.h — GGUF-embedded tokenizer extraction (C11).
 *
 * Reads tokenizer.ggml.tokens / tokenizer.ggml.merges / special-token
 * KV pairs straight out of a .gguf file so the model file is
 * self-contained (no data/*.bin extraction step). Produces RAW arrays;
 * wubu_tokenizer.c builds the wubu_tokenizer_t from them using its own
 * helpers (no duplicated logic).
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Raw extracted tokenizer data. tokens[i] and merges[j] are malloc'd
 * NUL-terminated strings owned by the caller (free each + the arrays). */
typedef struct {
    char **tokens;     /* vocab strings, [n_tokens] */
    int n_tokens;
    char **merges;     /* "left right" pairs, [n_merges] */
    int n_merges;
    int32_t bos_id, eos_id, pad_id;
} wubu_gguf_tokenizer_data_t;

/* Walk the GGUF KV section and fill out. Returns true on success.
 * On failure all pointers are NULL / counts 0. */
bool wubu_gguf_tokenizer_extract(const char *gguf_path,
                                 wubu_gguf_tokenizer_data_t *out);

/* Free the arrays returned by extract. */
void wubu_gguf_tokenizer_free(wubu_gguf_tokenizer_data_t *d);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_GGUF_TOKENIZER_H */
