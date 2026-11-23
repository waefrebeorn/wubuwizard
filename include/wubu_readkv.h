#ifndef WUBU_READKV_H
#define WUBU_READKV_H

/*
 * wubu_readkv.h -- EG-49: progressive KV code, store 8 bits / fetch 4 or 8
 * per query (ReadKV lineage), reimplemented in C11.
 *
 * One 8-bit code per value is split into a nibble-pair coarse array (two
 * values per byte) and a refinement array. A 4-bit query fetches only the
 * coarse array; an 8-bit query fetches both. Stored bytes are identical for
 * both paths; fetched bytes differ.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n;
    float scale;
    unsigned char *coarse; /* n/2 rounded up: two nibbles per byte */
    unsigned char *fine;   /* n bytes: low nibble of each code */
} wubu_readkv_t;

int wubu_readkv_store(const float *kv, int n, wubu_readkv_t *out);
int wubu_readkv_fetch(const wubu_readkv_t *b, int bits, float *out);

/* Bytes actually fetched from storage for this query width. */
size_t wubu_readkv_fetched_bytes(const wubu_readkv_t *b, int bits);

void wubu_readkv_free(wubu_readkv_t *b);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_READKV_H */
