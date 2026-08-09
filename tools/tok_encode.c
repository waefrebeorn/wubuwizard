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
        printf("byte_token_ids[0x20]=%d [0x41]=%d [0xE2]=%d [0x0A]=%d [0x00]=%d [0xA0]=%d\n",
               tok.byte_token_ids[0x20], tok.byte_token_ids[0x41], tok.byte_token_ids[0xE2],
               tok.byte_token_ids[0x0A], tok.byte_token_ids[0x00], tok.byte_token_ids[0xA0]);
        for (int b = 0; b < 8; b++)
            printf("  byte_text[%d] len=%d hex=%02x %02x\n", b, tok.byte_text_len[b],
                   tok.byte_text_bytes[b][0], tok.byte_text_bytes[b][1]);
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
            if (i < 4 && i < tok.vocab_size) {
                printf("  hex:");
                for (int b = 0; b < tok.vocab[i].byte_len && b < 24; b++)
                    printf(" %02x", tok.vocab[i].bytes[b]);
                printf("\n");
            }
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
        if (getenv("WUBU_TOK_DEBUG")) {
            /* trace the byte-scan on token 0 */
            int id0 = 0;
            const uint8_t *b0 = (const uint8_t *)tok.vocab[id0].bytes;
            int bl0 = tok.vocab[id0].byte_len;
            printf("  tok0 bytes: %02x %02x %02x blen=%d\n", b0[0], b0[1], b0[2], bl0);
            printf("  byte_text[0x0A] len=%d hex=%02x %02x\n", tok.byte_text_len[0x0A],
                   tok.byte_text_bytes[0x0A][0], tok.byte_text_bytes[0x0A][1]);
            printf("  byte_text[0x7F] len=%d hex=%02x %02x\n", tok.byte_text_len[0x7F],
                   tok.byte_text_bytes[0x7F][0], tok.byte_text_bytes[0x7F][1]);
            printf("  byte_ids[0x0A]=%d [0x7F]=%d\n", tok.byte_token_ids[0x0A], tok.byte_token_ids[0x7F]);
        }
    }
    wubu_tokenizer_free(&tok);
    return 0;
}
