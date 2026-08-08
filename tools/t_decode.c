/* t_decode.c — decode token ids printed by gen_text (stdout line) */
#include "wubu_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <id1,id2,...>\n", argv[0]); return 2; }
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init_from_gguf(&tok, argv[1])) return 1;
    /* parse comma-separated ids */
    char *s = strdup(argv[2]);
    char *p = s;
    while (*p) {
        char *end = NULL;
        long id = strtol(p, &end, 10);
        if (end == p) break;
        p = end;
        while (*p == ',' || *p == ' ') p++;
        if (id >= 0 && id < tok.vocab_size)
            fwrite(tok.vocab[id].bytes, 1, (size_t)tok.vocab[id].byte_len, stdout);
        else
            printf("[%ld]", id);
    }
    printf("\n");
    free(s);
    wubu_tokenizer_free(&tok);
    return 0;
}
