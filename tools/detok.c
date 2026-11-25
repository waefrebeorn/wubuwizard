/* detok - turn token ids into text using the GGUF's own vocab.
 *
 * Why this exists: every generation A/B in this repo has been judged on token IDs, and "149876 vs 220" says nothing
 * about whether output is coherent. The tokenizer data is already inside the .gguf, so this is the smallest possible
 * way to see what the engine actually wrote.
 *
 * Usage:  detok <gguf> <id> [id ...]
 *         detok <gguf> -f ids.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "wubu_gguf_tokenizer.h"

static const char *DEFAULT_GGUF = "C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf";

static void emit(const char *s)
{
    if (!s) return;
    /* SentencePiece marks a leading space with U+2581; a byte-fallback token is <0xNN>. */
    if (s[0] == '<' && s[1] == '0' && s[2] == 'x') {
        char *end = NULL;
        const long b = strtol(s + 3, &end, 16);
        if (end && *end == '>' && b >= 0 && b < 256) { putchar((int)b); return; }
    }
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        if (p[0] == 0xE2 && p[1] == 0x96 && p[2] == 0x81) { putchar(' '); p += 3; }   /* U+2581 SentencePiece */
        else if (p[0] == 0xC4 && p[1] == 0xA0) { putchar(' '); p += 2; }                /* U+0120 byte-level BPE */
        else { putchar(*p++); }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <gguf> <id>... | -f <file>\n", argv[0]); return 2; }
    const char *path = argv[1];
    if (strstr(path, ".gguf") == NULL) path = DEFAULT_GGUF;

    wubu_gguf_tokenizer_data_t tk;
    memset(&tk, 0, sizeof tk);
    if (!wubu_gguf_tokenizer_extract(path, &tk)) { fprintf(stderr, "tokenizer extract failed\n"); return 3; }
    fflush(stdout);

    FILE *f = NULL;
    int argi = 2;
    if (strcmp(argv[2], "-f") == 0 && argc > 3) { f = fopen(argv[3], "r"); if (!f) { perror("open"); return 4; } argi = argc; }

    int *ids = NULL; size_t n = 0, cap = 0;
    if (f) { int v; while (fscanf(f, "%d", &v) == 1) { if (n == cap) { cap = cap ? cap * 2 : 64; ids = realloc(ids, cap * sizeof(int)); } ids[n++] = v; } fclose(f); }
    else { for (; argi < argc; ++argi) { if (n == cap) { cap = cap ? cap * 2 : 64; ids = realloc(ids, cap * sizeof(int)); } ids[n++] = atoi(argv[argi]); } }

    for (size_t i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= tk.n_tokens) { printf("<%d:OOR>", ids[i]); continue; }
        emit(tk.tokens[ids[i]]);
    }
    putchar('\n');

    /* wubu_gguf_tokenizer_free owns the strings and the arrays - freeing them here as well was a double free
     * (the crash landed after the output, so stdout buffering hid it). */
    wubu_gguf_tokenizer_free(&tk);
    free(ids);
    return 0;
}
