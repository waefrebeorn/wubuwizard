/* tok_encode.c — tokenize text and print the ids (verify the encoder). */
#include "wubu_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <text> [text2 ...]\n", argv[0]); return 1; }
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init_from_gguf(&tok, argv[1])) { fprintf(stderr, "init failed\n"); return 1; }
    if (getenv("WUBU_TOK_DEBUG")) {
        char b1[64], b2[64];
        for (int i = 0; i < 20 && i < tok.n_merges; i++) {
            int l = tok.merges[i].left_id, r = tok.merges[i].right_id, nid = tok.merges[i].new_id;
            int p1 = 0, p2 = 0;
            if (l >= 0 && l < tok.vocab_size) p1 = wubu_tokenizer_decode(&tok, &l, 1, b1, 60);
            if (r >= 0 && r < tok.vocab_size) p2 = wubu_tokenizer_decode(&tok, &r, 1, b2, 60);
            printf("merge[%d] prio=%d: %d(%s) + %d(%s) -> %d\n", i, tok.merges[i].priority,
                   l, p1 > 0 ? b1 : "?", r, p2 > 0 ? b2 : "?", nid);
        }
        printf("byte_token_ids[0x20]=%d [0x41]=%d [0xE2]=%d\n",
               tok.byte_token_ids[0x20], tok.byte_token_ids[0x41], tok.byte_token_ids[0xE2]);
        for (int i = 0; i < 13; i++) {
            char rb[128]; int rl = 0;
            if (i < tok.vocab_size) {
                for (int b = 0; b < tok.vocab[i].byte_len && rl < 120; b++)
                    rb[rl++] = (tok.vocab[i].bytes[b] >= 32 && tok.vocab[i].bytes[b] < 127) ? tok.vocab[i].bytes[b] : '.';
                rb[rl] = 0;
            }
            printf("vocab[%d] blen=%d raw='%s' byte_tok=%d\n", i,
                   i < tok.vocab_size ? tok.vocab[i].byte_len : -1, rb,
                   i < 256 ? tok.byte_token_ids[i] : -1);
        }
    }
    for (int a = 2; a < argc; a++) {
        int ids[512];
        int n = wubu_tokenizer_encode(&tok, argv[a], ids, 512);
        printf("'%s' -> n=%d:", argv[a], n);
        if (n < 0) { printf(" encode error\n"); continue; }
        for (int i = 0; i < n && i < 16; i++) printf(" %d", ids[i]);
        printf("\n");
        char dec[4096]; int plen = 0;
        if (n) plen = wubu_tokenizer_decode(&tok, ids, n, dec, sizeof(dec));
        printf("  decode-back: '%s' (plen=%d)\n", plen > 0 ? dec : "", plen);
    }
    wubu_tokenizer_free(&tok);
    return 0;
}
