/*
 * test_sha1.c -- SHA-1 against the published FIPS 180-4 vectors, plus the
 * property the info-hash depends on: splitting the input anywhere must give
 * the same digest, because a .torrent's info dict is fed incrementally.
 */
#include "wubu_sha1.h"

#include <stdio.h>
#include <string.h>

static int failures, checks;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

static void hex_of(const uint8_t d[20], char out[41]) { wubu_sha1_hex(d, out); }

int main(void)
{
    printf("=== test_sha1 ===\n");

    /* -- FIPS 180-4 published vectors -- */
    {
        uint8_t d[20]; char h[41];
        wubu_sha1("", 0, d); hex_of(d, h);
        CHECK(strcmp(h, "da39a3ee5e6b4b0d3255bfef95601890afd80709") == 0,
              "the empty string matches FIPS 180-4");
    }
    {
        uint8_t d[20]; char h[41];
        wubu_sha1("abc", 3, d); hex_of(d, h);
        CHECK(strcmp(h, "a9993e364706816aba3e25717850c26c9cd0d89d") == 0,
              "\"abc\" matches FIPS 180-4");
    }
    {
        const char *m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        uint8_t d[20]; char h[41];
        wubu_sha1(m, strlen(m), d); hex_of(d, h);
        CHECK(strcmp(h, "84983e441c3bd26ebaae4aa1f95129e5e54670f1") == 0,
              "the 56-byte FIPS message matches");
    }
    {
        /* The boundary just past one block: 64 bytes exactly, then 65. */
        char m[65];
        memset(m, 'a', sizeof(m));
        uint8_t d[20]; char h[41];
        wubu_sha1(m, 64, d); hex_of(d, h);
        CHECK(strcmp(h, "0098ba824b5c16427bd7a1122a5a442a25ec644d") == 0,
              "exactly one block matches");
        wubu_sha1(m, 65, d); hex_of(d, h);
        CHECK(strcmp(h, "11655326c708d70319be2610e8a57d9a5b959d3b") == 0,
              "one block plus one byte matches");
    }
    {
        /* One million 'a' (FIPS 180-4 third vector). */
        char buf[1000];
        memset(buf, 'a', sizeof(buf));
        uint8_t cblob[WUBU_SHA1_CTX_SZ];
        wubu_sha1_ctx *c = (wubu_sha1_ctx *)cblob;
        wubu_sha1_init(c);
        for (int i = 0; i < 1000; i++)      /* exactly 1,000,000 bytes */
            wubu_sha1_update(c, buf, sizeof(buf));
        uint8_t d[20]; char h[41];
        wubu_sha1_final(c, d); hex_of(d, h);
        CHECK(strcmp(h, "34aa973cd4c4daa4f61eeb2bdbad27316534016f") == 0,
              "one million 'a' matches FIPS 180-4 (streamed)");
    }

    /* -- streaming must equal one-shot at EVERY split point -- */
    {
        char msg[200];
        for (int i = 0; i < 200; i++) msg[i] = (char)(i * 7 + 3);
        uint8_t whole[20];
        wubu_sha1(msg, sizeof(msg), whole);

        int bad = -1;
        for (size_t cut = 0; cut <= sizeof(msg); cut++) {
            uint8_t cblob[WUBU_SHA1_CTX_SZ];
        wubu_sha1_ctx *c = (wubu_sha1_ctx *)cblob;
            wubu_sha1_init(c);
            wubu_sha1_update(c, msg, cut);
            wubu_sha1_update(c, msg + cut, sizeof(msg) - cut);
            uint8_t part[20];
            wubu_sha1_final(c, part);
            if (memcmp(part, whole, 20) != 0) { bad = (int)cut; break; }
        }
        CHECK(bad < 0, "every split point streams to the same digest");
        if (bad >= 0) printf("  first differing split: %d\n", bad);
    }

    /* -- the caller's context really fits the documented size -- */
    {
        uint8_t blob[WUBU_SHA1_CTX_SZ];
        wubu_sha1_ctx *c = (wubu_sha1_ctx *)blob;
        wubu_sha1_init(c);
        wubu_sha1_update(c, "abc", 3);
        uint8_t d[20]; char h[41];
        wubu_sha1_final(c, d); hex_of(d, h);
        CHECK(strcmp(h, "a9993e364706816aba3e25717850c26c9cd0d89d") == 0,
              "a caller-allocated blob works as the context");
    }

    /* -- bad arguments -- */
    {
        CHECK(wubu_sha1("abc", 3, NULL) == -1, "a NULL digest is refused");
        uint8_t d[20];
        char h[41];
        wubu_sha1(NULL, 0, d);
        hex_of(d, h);
        CHECK(strcmp(h, "da39a3ee5e6b4b0d3255bfef95601890afd80709") == 0,
              "NULL data with zero length digests as the empty string");
        CHECK(wubu_sha1_hex(NULL, h) == -1, "a NULL digest is refused by hex()");
        CHECK(wubu_sha1_hex(d, NULL) == -1, "a NULL hex buffer is refused");
    }

    printf("=== %d checks, %d failed ===\n", checks, failures);
    return failures ? 1 : 0;
}
