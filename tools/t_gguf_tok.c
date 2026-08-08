/* t_gguf_tok.c — probe the GGUF-embedded tokenizer path.
 * Usage: t_gguf_tok <model.gguf> <text>
 * Prints token ids + decode, mirroring the llama.cpp oracle.
 */
#include "wubu_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <text>\n", argv[0]); return 2; }
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init_from_gguf(&tok, argv[1])) {
        fprintf(stderr, "gguf tokenizer init failed\n");
        return 1;
    }
    int toks[256];
    int n = wubu_tokenizer_encode(&tok, argv[2], toks, 256);
    printf("text='%s' n=%d:", argv[2], n);
    for (int i = 0; i < n; i++) printf(" %d", toks[i]);
    printf("\n");
    /* decode back */
    for (int i = 0; i < n; i++) {
        if (toks[i] >= 0 && toks[i] < tok.vocab_size)
            printf("[%d]=%.*s\n", toks[i], tok.vocab[toks[i]].byte_len, tok.vocab[toks[i]].bytes);
    }
    wubu_tokenizer_free(&tok);
    return 0;
}
