/*
 * wubu_blob_store.c -- EG-29 content-addressed store. Pure C11.
 */
#include "wubu_blob_store.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static const unsigned K[64] = {
0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

static unsigned rotr(unsigned x, int n) { return (x >> n) | (x << (32 - n)); }

int wubu_sha256(const void *data, size_t len, wubu_digest_t *out) {
    unsigned h[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                     0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    unsigned char buf[64];
    const unsigned char *p = (const unsigned char *)data;
    size_t remain = len, total = len;
    int i;
    if (!out || (!data && len)) return -1;

    while (remain >= 64) {
        unsigned w[64], a,b,c,d,e,f,g,hh;
        for (i = 0; i < 16; i++)
            w[i] = ((unsigned)p[i*4] << 24) | ((unsigned)p[i*4+1] << 16) |
                   ((unsigned)p[i*4+2] << 8) | (unsigned)p[i*4+3];
        for (i = 16; i < 64; i++) {
            unsigned s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
            unsigned s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        a=h[0]; b=h[1]; c=h[2]; d=h[3]; e=h[4]; f=h[5]; g=h[6]; hh=h[7];
        for (i = 0; i < 64; i++) {
            unsigned S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            unsigned ch = (e & f) ^ ((~e) & g);
            unsigned t1 = hh + S1 + ch + K[i] + w[i];
            unsigned S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            unsigned maj = (a & b) ^ (a & c) ^ (b & c);
            unsigned t2 = S0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
        h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
        p += 64; remain -= 64;
    }

    memset(buf, 0, sizeof(buf));
    memcpy(buf, p, remain);
    buf[remain] = 0x80;
    if (remain >= 56) {
        unsigned w[64], a,b,c,d,e,f,g,hh;
        int j;
        for (i = 0; i < 16; i++)
            w[i] = ((unsigned)buf[i*4] << 24) | ((unsigned)buf[i*4+1] << 16) |
                   ((unsigned)buf[i*4+2] << 8) | (unsigned)buf[i*4+3];
        for (i = 16; i < 64; i++) {
            unsigned s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
            unsigned s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        a=h[0]; b=h[1]; c=h[2]; d=h[3]; e=h[4]; f=h[5]; g=h[6]; hh=h[7];
        for (j = 0; j < 64; j++) {
            unsigned S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            unsigned ch = (e & f) ^ ((~e) & g);
            unsigned t1 = hh + S1 + ch + K[j] + w[j];
            unsigned S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            unsigned maj = (a & b) ^ (a & c) ^ (b & c);
            unsigned t2 = S0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
        h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
        memset(buf, 0, sizeof(buf));
    }
    {
        unsigned long long bits = (unsigned long long)total * 8ULL;
        for (i = 0; i < 8; i++)
            buf[63 - i] = (unsigned char)(bits >> (8 * i));
    }
    {
        unsigned w[64], a,b,c,d,e,f,g,hh;
        for (i = 0; i < 16; i++)
            w[i] = ((unsigned)buf[i*4] << 24) | ((unsigned)buf[i*4+1] << 16) |
                   ((unsigned)buf[i*4+2] << 8) | (unsigned)buf[i*4+3];
        for (i = 16; i < 64; i++) {
            unsigned s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
            unsigned s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        a=h[0]; b=h[1]; c=h[2]; d=h[3]; e=h[4]; f=h[5]; g=h[6]; hh=h[7];
        for (i = 0; i < 64; i++) {
            unsigned S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            unsigned ch = (e & f) ^ ((~e) & g);
            unsigned t1 = hh + S1 + ch + K[i] + w[i];
            unsigned S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            unsigned maj = (a & b) ^ (a & c) ^ (b & c);
            unsigned t2 = S0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
        h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    for (i = 0; i < 8; i++) {
        out->bytes[i*4]   = (unsigned char)(h[i] >> 24);
        out->bytes[i*4+1] = (unsigned char)(h[i] >> 16);
        out->bytes[i*4+2] = (unsigned char)(h[i] >> 8);
        out->bytes[i*4+3] = (unsigned char)h[i];
    }
    for (i = 0; i < 32; i++) sprintf(out->hex + i*2, "%02x", out->bytes[i]);
    out->hex[64] = '\0';
    return 0;
}

typedef struct {
    char path[256];
    char hex[65];
    size_t bytes;
} entry_t;

struct wubu_blob_store {
    char dir[256];
    entry_t entries[64];
    int n_entries;
};

wubu_blob_store_t *wubu_blob_store_create(const char *dir) {
    wubu_blob_store_t *s;
    if (!dir || !*dir || strlen(dir) >= sizeof(s->dir)) return NULL;
    s = (wubu_blob_store_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    snprintf(s->dir, sizeof(s->dir), "%s", dir);
    return s;
}

void wubu_blob_store_free(wubu_blob_store_t *s) { free(s); }

static int blob_path(const wubu_blob_store_t *s, const char *hex,
                     char *out, size_t cap) {
    if (!hex || strlen(hex) != 64) return -1;
    for (int i = 0; i < 64; i++)
        if (!((hex[i] >= '0' && hex[i] <= '9') ||
              (hex[i] >= 'a' && hex[i] <= 'f'))) return -1;
    snprintf(out, cap, "%s/%s.blob", s->dir, hex);
    return 0;
}

int wubu_blob_put(wubu_blob_store_t *s, const void *data, size_t len,
                  wubu_digest_t *digest) {
    char path[512], tmp[540], hex[65];
    FILE *f;
    wubu_digest_t d;
    if (!s || !data || !digest) return -1;
    if (wubu_sha256(data, len, &d) != 0) return -1;
    if (blob_path(s, d.hex, path, sizeof(path)) != 0) return -1;
    f = fopen(path, "rb");
    if (f) { fclose(f); *digest = d; return 0; } /* dedup */

    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    f = fopen(tmp, "wb");
    if (!f) return -1;
    if (len && fwrite(data, 1, len, f) != len) { fclose(f); remove(tmp); return -1; }
    if (fclose(f) != 0) { remove(tmp); return -1; }
    if (rename(tmp, path) != 0) { remove(tmp); return -1; }
    snprintf(hex, sizeof(hex), "%s", d.hex);
    (void)hex;
    *digest = d;
    return 0;
}

long wubu_blob_get(wubu_blob_store_t *s, const char *hex, void *buf, size_t cap) {
    char path[512];
    FILE *f;
    long n;
    wubu_digest_t d;
    if (!s || !buf || blob_path(s, hex, path, sizeof(path)) != 0) return -1;
    f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    n = ftell(f);
    if (n < 0 || (size_t)n > cap || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); return -1; }
    fclose(f);
    if (wubu_sha256(buf, (size_t)n, &d) != 0 || strcmp(d.hex, hex) != 0) return -1;
    return n;
}

int wubu_blob_manifest_add(wubu_blob_store_t *s, const char *path,
                           const char *hex, size_t bytes) {
    entry_t *e;
    int i;
    {
        char blob[512];
        if (!s || !path || !*path ||
            blob_path(s, hex, blob, sizeof(blob)) != 0) return -1;
    }
    for (i = 0; i < s->n_entries; i++)
        if (strcmp(s->entries[i].path, path) == 0) {
            e = &s->entries[i];
            snprintf(e->hex, sizeof(e->hex), "%s", hex);
            e->bytes = bytes;
            return 0;
        }
    if (s->n_entries >= (int)(sizeof(s->entries)/sizeof(s->entries[0]))) return -1;
    e = &s->entries[s->n_entries++];
    snprintf(e->path, sizeof(e->path), "%s", path);
    snprintf(e->hex, sizeof(e->hex), "%s", hex);
    e->bytes = bytes;
    return 0;
}

int wubu_blob_manifest_get(wubu_blob_store_t *s, const char *path,
                           char *hex, size_t hex_cap, size_t *bytes) {
    for (int i = 0; s && i < s->n_entries; i++)
        if (strcmp(s->entries[i].path, path) == 0) {
            if (hex) {
                if (hex_cap < 65) return -1;
                snprintf(hex, hex_cap, "%s", s->entries[i].hex);
            }
            if (bytes) *bytes = s->entries[i].bytes;
            return 0;
        }
    return -1;
}
