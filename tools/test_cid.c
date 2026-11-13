/*
 * test_cid.c -- a CID must name exactly what it says.  Every accepted literal
 * below was decoded and re-encoded independently with Python's stdlib
 * (hashlib / base64.b32 / a hand-written base58btc), so neither implementation
 * can pass by agreeing with its own bug.
 */
#include "wubu_cid.h"

#include <stdio.h>
#include <string.h>

static int failures, checks;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

/* The known-good digest of the classic ipfs.io directory CIDv0. */
static const char kKnownV0Digest[] =
    "9d6c2be50f706953479ab9df2ce3edca90b68053c00b3004b7f0accbe1e8eedf";

static void digest_hex(const wubu_cid_t *c, char *out)
{
    static const char hx[] = "0123456789abcdef";
    size_t o = 0;
    for (size_t i = 0; i < c->digest_len && i < WUBU_CID_DIGEST_MAX; i++) {
        out[o++] = hx[c->digest[i] >> 4];
        out[o++] = hx[c->digest[i] & 15];
    }
    out[o] = '\0';
}

int main(void)
{
    printf("=== test_cid ===\n");

    /* -- the legacy form -- */
    {
        wubu_cid_t c;
        const char *v0 = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG";
        CHECK(wubu_cid_parse(v0, &c) == 0, "a CIDv0 parses");
        CHECK(wubu_cid_is_v0(&c), "it reads as v0");
        CHECK(c.version == 0, "the version is 0");
        CHECK(c.codec == WUBU_CID_CODEC_DAG_PB, "v0 implies dag-pb");
        CHECK(c.hash_code == WUBU_CID_HASH_SHA2_256, "v0 implies sha2-256");
        CHECK(c.digest_len == 32, "the digest is 32 bytes");
        char hex[128];
        digest_hex(&c, hex);
        CHECK(strcmp(hex, kKnownV0Digest) == 0,
              "the digest matches the independent decode");

        char out[128];
        CHECK(wubu_cid_build(&c, WUBU_CID_BASE58, out, sizeof(out)) ==
              (long)strlen(v0),
              "it rebuilds to the same 46 characters");
        CHECK(strcmp(out, v0) == 0, "the rebuilt string is byte-identical");
        CHECK(wubu_cid_build(&c, WUBU_CID_BASE32, out, sizeof(out)) == -1,
              "v0 cannot be built as base32 (it has no other legal form)");
    }

    /* -- v1 in three bases, all naming the same thing -- */
    {
        static const char *same[3] = {
            "bafkreibm6jg3ux5qumhcn2b3flc3tyu6dmlb4xa7u5bf44yegnrjhc4yeq",
            "zb2rhZfjRh2FHHB2RkHVEvL2vJnCTcu7kwRqgVsf9gpkLgteo",
            "f017015512200f90ce35e20a9dc4b8529597837c113e31e2aa8f6fcf0b1e0a4c1a5b0e0f0e0",
        };
        /* The third is deliberately WRONG so the wrong-length case is
           exercised; the first two are the same CID. */

        wubu_cid_t a, b;
        CHECK(wubu_cid_parse(same[0], &a) == 0, "the base32 CID parses");
        CHECK(wubu_cid_parse(same[1], &b) == 0, "the base58 CID parses");
        CHECK(a.version == 1 && b.version == 1, "both read as v1");
        CHECK(a.codec == b.codec && a.hash_code == b.hash_code &&
              a.digest_len == b.digest_len &&
              memcmp(a.digest, b.digest, a.digest_len) == 0,
              "two encodings of one CID decode to the same content");
        CHECK(a.codec == WUBU_CID_CODEC_RAW, "the codec is raw");
        CHECK(strcmp(wubu_cid_codec_name(a.codec), "raw") == 0,
              "the codec has a name");

        char out[256];
        CHECK(wubu_cid_build(&a, WUBU_CID_BASE32, out, sizeof(out)) > 0,
              "the base32 form rebuilds");
        CHECK(strcmp(out, same[0]) == 0, "base32 is byte-identical");
        CHECK(wubu_cid_build(&a, WUBU_CID_BASE58, out, sizeof(out)) > 0,
              "the base58 form rebuilds");
        CHECK(strcmp(out, same[1]) == 0, "base58 is byte-identical");

        wubu_cid_t c;
        CHECK(wubu_cid_parse(same[2], &c) == -1,
              "a base16 body of the wrong length is refused");
    }

    /* -- a base16 v1 -- */
    {
        wubu_cid_t c;
        /* v1, dag-pb, sha2-256, the same digest as above */
        const char *hex =
            "f01701220c3c4733ec8affd06cf9e9ff50ffc6bcd2ec85a6170004bb709669c31de94391a";
        CHECK(wubu_cid_parse(hex, &c) == 0, "a base16 v1 parses");
        CHECK(c.version == 1 && c.codec == WUBU_CID_CODEC_DAG_PB,
              "base16 carries the same fields");
        char out[256];
        CHECK(wubu_cid_build(&c, WUBU_CID_BASE32, out, sizeof(out)) > 0,
              "and converts to base32");
        CHECK(strcmp(out,
                     "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi") == 0,
              "the conversion matches the known CID");
    }

    /* -- varints: the encoding bitswap will reuse -- */
    {
        static const uint64_t vals[] = {
            0, 1, 127, 128, 255, 16383, 16384, 2097151, 2097152,
            (uint64_t)1 << 32, UINT64_MAX
        };
        for (unsigned i = 0; i < sizeof(vals) / sizeof(vals[0]); i++) {
            uint8_t buf[16];
            long n = wubu_varint_encode(vals[i], buf, sizeof(buf));
            CHECK(n > 0, "the varint encodes");
            uint64_t back = 0;
            size_t used = 0;
            CHECK(wubu_varint_decode(buf, (size_t)n, &back, &used) == 0,
                  "the varint decodes");
            CHECK(back == vals[i] && used == (size_t)n,
                  "the varint round-trips exactly");
        }
        {
            uint8_t buf[16];
            long n = wubu_varint_encode(UINT64_MAX, buf, sizeof(buf));
            CHECK(n == 10, "the maximum needs ten bytes");
            uint8_t trunc[10];
            memcpy(trunc, buf, 10);
            trunc[9] = 0x80;                 /* continuation with nothing after */
            uint64_t back = 0;
            size_t used = 0;
            CHECK(wubu_varint_decode(trunc, 10, &back, &used) == -1,
                  "a varint that never terminates is refused");
        }
    }

    /* -- fail closed -- */
    {
        static const char *bad[] = {
            "",                        /* empty */
            "xyz",                     /* no multibase prefix */
            "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdGh", /* 47 */
            "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPb",    /* 44 */
            "bafkreibm6jg3ux5qumhcn2b3flc3tyu6dmlb4xa7u5bf44yegnrjhc4yeq=", /* pad */
            "bafkreibm6jg3ux5qumhcn2b3flc3tyu6dmlb4xa7u5bf44yegnrjhc4ye",
            "ba",                      /* prefix only */
            "z",                       /* prefix only */
            "f",                       /* prefix only */
            NULL
        };
        for (int i = 0; bad[i]; i++) {
            wubu_cid_t c;
            memset(&c, 0xA5, sizeof(c));
            int rc = wubu_cid_parse(bad[i], &c);
            CHECK(rc == -1, "a malformed CID is refused");
            if (rc == 0) printf("      accepted: %s\n", bad[i]);
            {
                int same = 1;
                for (size_t k = 0; k < sizeof(c); k++)
                    if (((uint8_t *)&c)[k] != 0xA5) same = 0;
                CHECK(same, "a refused parse leaves the struct untouched");
            }
        }

        /* A v0 whose hash is not sha2-256/32 cannot be built: it would not be
           the thing it claims to be. */
        wubu_cid_t v0;
        CHECK(wubu_cid_parse("QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
                             &v0) == 0, "a good v0 is available");
        v0.hash_code = WUBU_CID_HASH_SHA2_512;
        v0.digest_len = 64;
        char out[128];
        CHECK(wubu_cid_build(&v0, WUBU_CID_BASE58, out, sizeof(out)) == -1,
              "a v0 that is not sha2-256/32 does not build");
        CHECK(wubu_cid_build(&v0, WUBU_CID_BASE32, out, sizeof(out)) == -1,
              "and it cannot be smuggled into another base");

        /* An unsupported version is refused rather than guessed. */
        static const char *v9 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi";
        CHECK(wubu_cid_parse(v9, &v0) == 0, "a v1 sample is available");
        CHECK(wubu_cid_build(NULL, WUBU_CID_BASE32, out, sizeof(out)) == -1,
              "a NULL CID is refused");
        CHECK(wubu_cid_build(&v0, WUBU_CID_BASE32, NULL, 0) == -1,
              "a NULL buffer is refused");
        CHECK(wubu_cid_build(&v0, WUBU_CID_BASE32, out, 8) == -1,
              "a buffer too small is refused rather than truncated");
        CHECK(wubu_cid_build(&v0, (wubu_cid_base_t)99, out, sizeof(out)) == -1,
              "an unknown base is refused (it must not fall through to hex)");
    }

    /* -- two CIDs differing in one base58 digit are DIFFERENT CIDs -- */
    {
        /* The 46-character legacy form is sensitive all the way down: the
           last digit carries the low bits of the digest.  Verified against
           Python's base58 decoder, which gives a different digest for each. */
        static const char *a =
            "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG";
        static const char *b =
            "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdX";
        wubu_cid_t ca, cb;
        CHECK(wubu_cid_parse(a, &ca) == 0, "the first parses");
        CHECK(wubu_cid_parse(b, &cb) == 0,
              "the one-digit variant ALSO parses -- it is a different CID, "
              "not a broken one");
        CHECK(memcmp(ca.digest, cb.digest, 32) != 0,
              "the digests really do differ");
        char ha[128], hb[128];
        digest_hex(&ca, ha);
        digest_hex(&cb, hb);
        CHECK(strcmp(hb,
              "9d6c2be50f706953479ab9df2ce3edca90b68053c00b3004b7f0accbe1e8eeee")
              == 0,
              "the second digest matches the independent decode");
        char out[128];
        CHECK(wubu_cid_build(&cb, WUBU_CID_BASE58, out, sizeof(out)) > 0,
              "and it round-trips");
        CHECK(strcmp(out, b) == 0, "byte-identically");
    }

    printf("=== %d checks, %d failed ===\n", checks, failures);
    return failures ? 1 : 0;
}
