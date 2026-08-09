/* tok_decode.c — decode token ids with a model's tokenizer. */
#include "wubu_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <gguf> <tok> [tok...]\n", argv[0]); return 1; }
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init_from_gguf(&tok, argv[1])) { fprintf(stderr, "tokenizer init failed\n"); return 1; }
    int ids[256]; int n = 0;
    for (int i = 2; i < argc && n < 256; i++) ids[n++] = atoi(argv[i]);
    char buf[8192];
    int plen = wubu_tokenizer_decode(&tok, ids, n, buf, sizeof(buf));
    if (plen < 0) plen = 0;
    buf[plen] = 0;
    printf("DECODE(%d toks, %d chars): %s\n", n, plen, buf);
    wubu_tokenizer_free(&tok);
    return 0;
}
