/*
 * wubu_tensorfold.c -- EG-66 disk prefix snapshot. Pure C11.
 */
#include "wubu_tensorfold.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TF_MAGIC 0x464C4553u /* "SELF" */
#define TF_VERSION 1u
#define TF_HEADER 24u

static uint64_t fnv(const unsigned char *p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static void put32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16);
    p[3] = (unsigned char)(v >> 24);
}

static uint32_t get32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

void wubu_tensorfold_serial_kv(const int *tokens, int n_tokens, int dim,
                               float *kv) {
    for (int i = 0; i < n_tokens; i++)
        for (int d = 0; d < dim; d++)
            kv[(size_t)i * dim + d] =
                (float)(tokens[i] * 0.125 + (d + 1) * 0.03125);
}

int wubu_tensorfold_snapshot(const char *path, const int *tokens, int n_tokens,
                             const float *kv, int dim) {
    unsigned char *buf;
    size_t npayload, n;
    FILE *f;
    uint64_t digest;
    if (!path || !tokens || !kv || n_tokens <= 0 || dim <= 0) return -1;
    npayload = (size_t)n_tokens * (sizeof(int) + (size_t)dim * sizeof(float));
    if (npayload > (size_t)1 << 28) return -1;
    buf = (unsigned char *)malloc(TF_HEADER + npayload + 8);
    if (!buf) return -1;
    put32(buf, TF_MAGIC);
    put32(buf + 4, TF_VERSION);
    put32(buf + 8, (uint32_t)n_tokens);
    put32(buf + 12, (uint32_t)dim);
    put32(buf + 16, (uint32_t)npayload);
    put32(buf + 20, 0);
    for (int i = 0; i < n_tokens; i++) put32(buf + TF_HEADER + (size_t)i * 4,
                                             (uint32_t)tokens[i]);
    memcpy(buf + TF_HEADER + (size_t)n_tokens * 4, kv,
           (size_t)n_tokens * dim * sizeof(float));
    digest = fnv(buf + TF_HEADER, npayload);
    for (int i = 0; i < 8; i++)
        buf[TF_HEADER + npayload + i] = (unsigned char)(digest >> (8 * i));
    f = fopen(path, "wb");
    if (!f) { free(buf); return -1; }
    n = fwrite(buf, 1, TF_HEADER + npayload + 8, f);
    fclose(f);
    free(buf);
    return n == TF_HEADER + npayload + 8 ? 0 : -1;
}

int wubu_tensorfold_restore(const char *path, int *tokens, int n_tokens,
                            float *kv, int dim) {
    FILE *f;
    long fsize;
    unsigned char *buf;
    size_t npayload, n;
    uint64_t digest, stored = 0;
    if (!path || !tokens || !kv || n_tokens <= 0 || dim <= 0) return -1;
    f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0 || (fsize = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    npayload = (size_t)n_tokens * (sizeof(int) + (size_t)dim * sizeof(float));
    if ((size_t)fsize != TF_HEADER + npayload + 8) { fclose(f); return -1; }
    buf = (unsigned char *)malloc((size_t)fsize);
    if (!buf) { fclose(f); return -1; }
    n = fread(buf, 1, (size_t)fsize, f);
    fclose(f);
    if (n != (size_t)fsize) { free(buf); return -1; }
    if (get32(buf) != TF_MAGIC || get32(buf + 4) != TF_VERSION ||
        get32(buf + 8) != (uint32_t)n_tokens ||
        get32(buf + 12) != (uint32_t)dim ||
        get32(buf + 16) != (uint32_t)npayload) { free(buf); return -1; }
    digest = fnv(buf + TF_HEADER, npayload);
    for (int i = 0; i < 8; i++)
        stored |= (uint64_t)buf[TF_HEADER + npayload + i] << (8 * i);
    if (digest != stored) { free(buf); return -1; }
    for (int i = 0; i < n_tokens; i++)
        tokens[i] = (int)get32(buf + TF_HEADER + (size_t)i * 4);
    memcpy(kv, buf + TF_HEADER + (size_t)n_tokens * 4,
           (size_t)n_tokens * dim * sizeof(float));
    free(buf);
    return 0;
}
