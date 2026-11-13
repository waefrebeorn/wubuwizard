/*
 * wubu_sha1.c -- SHA-1 (FIPS 180-4), self-contained.
 *
 * Freestanding C11: no malloc, no hosted APIs.  The context is an opaque blob
 * the caller allocates; its size is part of the contract (static-asserted
 * here).  The 80-word schedule is built on the stack per block rather than
 * stored in the context, which keeps the caller's blob at 96 bytes -- the
 * size is what makes this usable from a kernel or firmware path that cannot
 * hand back a pointer into its own globals.
 */
#include "wubu_sha1.h"

#include <string.h>

/* The real layout.  Never exposed: callers hold WUBU_SHA1_CTX_SZ bytes. */
struct wubu_sha1_ctx_impl {
    uint64_t        total;      /* bytes fed so far: first, so its 8-byte
                                 * alignment does not leave a hole */
    uint32_t        h[5];
    uint8_t         buf[64];    /* partial block */
    unsigned        buflen;
};                              /* exactly WUBU_SHA1_CTX_SZ (96), asserted */

_Static_assert(sizeof(struct wubu_sha1_ctx_impl) <= WUBU_SHA1_CTX_SZ,
               "the SHA-1 context outgrew its documented size");

static uint32_t rol(uint32_t v, int n)
{
    return (v << n) | (v >> (32 - n));
}

static void sha1_block(uint32_t h[5], const uint8_t block[64])
{
    uint32_t w[80];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               ((uint32_t)block[i * 4 + 3]);
    for (int i = 16; i < 80; i++)
        w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20)      { f = (b & c) | (~b & d);        k = 0x5A827999u; }
        else if (i < 40) { f = b ^ c ^ d;                k = 0x6ED9EBA1u; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCu; }
        else             { f = b ^ c ^ d;                k = 0xCA62C1D6u; }
        uint32_t t = rol(a, 5) + f + e + k + w[i];
        e = d; d = c; c = rol(b, 30); b = a; a = t;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
}

void wubu_sha1_init(wubu_sha1_ctx *c)
{
    struct wubu_sha1_ctx_impl *s = (struct wubu_sha1_ctx_impl *)c;
    s->h[0] = 0x67452301u; s->h[1] = 0xEFCDAB89u;
    s->h[2] = 0x98BADCFEu; s->h[3] = 0x10325476u;
    s->h[4] = 0xC3D2E1F0u;
    s->total = 0;
    s->buflen = 0;
}

void wubu_sha1_update(wubu_sha1_ctx *c, const void *data, size_t len)
{
    struct wubu_sha1_ctx_impl *s = (struct wubu_sha1_ctx_impl *)c;
    const uint8_t *p = (const uint8_t *)data;

    s->total += len;

    /* Fill the partial block first, and FLUSH it as soon as it is full --
     * otherwise a full buffer is never hashed and later writes run off the
     * end of it. */
    while (len && s->buflen < 64) {
        s->buf[s->buflen++] = *p++;
        len--;
    }
    if (s->buflen == 64) {
        sha1_block(s->h, s->buf);
        s->buflen = 0;
    }
    while (len >= 64) {
        sha1_block(s->h, p);
        p += 64;
        len -= 64;
    }
    /* Whatever is left becomes the new partial block. */
    while (len--) s->buf[s->buflen++] = *p++;
}

void wubu_sha1_final(wubu_sha1_ctx *c, uint8_t out[WUBU_SHA1_SZ])
{
    struct wubu_sha1_ctx_impl *s = (struct wubu_sha1_ctx_impl *)c;

    uint64_t bits = s->total * 8u;
    /* Padding: a 0x80 byte, zeros to 56 mod 64, then the 64-bit big-endian
       length.  Two blocks when the tail has no room for the length. */
    uint8_t pad[128];
    size_t plen = 0;
    pad[plen++] = 0x80;
    while ((s->total + plen) % 64 != 56) pad[plen++] = 0x00;
    for (int i = 7; i >= 0; i--) pad[plen++] = (uint8_t)(bits >> (i * 8));

    wubu_sha1_update(c, pad, plen);

    for (int i = 0; i < 5; i++) {
        out[i * 4]     = (uint8_t)(s->h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(s->h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(s->h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)(s->h[i]);
    }
}

int wubu_sha1(const void *data, size_t len, uint8_t out[WUBU_SHA1_SZ])
{
    if (!out) return -1;
    struct wubu_sha1_ctx_impl c;      /* this TU owns the layout */
    wubu_sha1_init((wubu_sha1_ctx *)&c);
    if (data && len) wubu_sha1_update((wubu_sha1_ctx *)&c, data, len);
    wubu_sha1_final((wubu_sha1_ctx *)&c, out);
    return 0;
}

int wubu_sha1_hex(const uint8_t digest[WUBU_SHA1_SZ], char out[41])
{
    static const char hex[] = "0123456789abcdef";
    if (!digest || !out) return -1;
    for (int i = 0; i < WUBU_SHA1_SZ; i++) {
        out[i * 2]     = hex[digest[i] >> 4];
        out[i * 2 + 1] = hex[digest[i] & 15];
    }
    out[40] = '\0';
    return 0;
}
