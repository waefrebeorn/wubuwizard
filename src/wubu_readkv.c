/*
 * wubu_readkv.c -- EG-49 progressive KV code. Pure C11.
 */
#include "wubu_readkv.h"

#include <math.h>
#include <stdlib.h>

int wubu_readkv_store(const float *kv, int n, wubu_readkv_t *out) {
    float absmax = 0.0f;
    size_t half;
    if (!kv || !out || n <= 0) return -1;
    half = ((size_t)n + 1u) / 2u;
    out->coarse = (unsigned char *)calloc(half, 1);
    out->fine = (unsigned char *)calloc((size_t)n, 1);
    if (!out->coarse || !out->fine) {
        wubu_readkv_free(out);
        return -1;
    }
    out->n = n;
    for (int i = 0; i < n; i++) {
        float a = fabsf(kv[i]);
        if (a > absmax) absmax = a;
    }
    out->scale = absmax > 0.0f ? absmax / 127.0f : 1.0f;
    for (int i = 0; i < n; i++) {
        float q = kv[i] / out->scale;
        int v = (int)(q >= 0.0f ? floorf(q + 0.5f) : ceilf(q - 0.5f));
        unsigned code;
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        code = (unsigned)(v + 128); /* [1..255] */
        out->fine[i] = (unsigned char)(code & 15u);
        {
            unsigned c = code >> 4;
            unsigned char *p = &out->coarse[(size_t)i / 2u];
            if ((i & 1) == 0) *p = (unsigned char)(c << 4);
            else *p = (unsigned char)(*p | c);
        }
    }
    return 0;
}

int wubu_readkv_fetch(const wubu_readkv_t *b, int bits, float *out) {
    if (!b || !out || !b->coarse || !b->fine || b->n <= 0 ||
        (bits != 4 && bits != 8)) return -1;
    for (int i = 0; i < b->n; i++) {
        unsigned c = (unsigned)(b->coarse[(size_t)i / 2u] >> (((i & 1) == 0) ? 4 : 0)) & 15u;
        int q;
        if (bits == 4) {
            /* Midpoint of the 16-value bucket represented by this nibble. */
            q = (int)((c << 4) + 8) - 128;
        } else {
            q = (int)((c << 4) | b->fine[i]) - 128;
        }
        out[i] = (float)q * b->scale;
    }
    return 0;
}

size_t wubu_readkv_fetched_bytes(const wubu_readkv_t *b, int bits) {
    size_t half;
    if (!b || b->n <= 0 || (bits != 4 && bits != 8)) return 0;
    half = ((size_t)b->n + 1u) / 2u;
    return bits == 4 ? half : half + (size_t)b->n;
}

void wubu_readkv_free(wubu_readkv_t *b) {
    if (!b) return;
    free(b->coarse);
    free(b->fine);
    b->coarse = NULL;
    b->fine = NULL;
    b->n = 0;
    b->scale = 0.0f;
}
