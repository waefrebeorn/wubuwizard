/*
 * test_multistream.c -- protocol negotiation must pick exactly what both peers
 * offered, and nothing else.  A mis-framed handshake either connects to the
 * wrong protocol or fails loudly; there is no third outcome worth testing.
 */
#include "wubu_multistream.h"

#include <stdio.h>
#include <string.h>

static int failures, checks;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

/* The multistream header on the wire: varint 19 then the text plus '\n'. */
static const uint8_t kHeaderWire[] = {
    0x13,
    '/', 'm', 'u', 'l', 't', 'i', 's', 't', 'r', 'e', 'a', 'm',
    '/', '1', '.', '0', '.', '0', '\n'
};

int main(void)
{
    printf("=== test_multistream ===\n");

    /* ---- framing constants are self-consistent ---- */
    CHECK(strlen(WUBU_MS_HEADER) == WUBU_MS_HEADER_LEN,
          "the header length constant matches the string");
    CHECK(strlen(WUBU_MS_NA) == WUBU_MS_NA_LEN,
          "the na length constant matches the string");

    /* ---- the header ---- */
    {
        uint8_t out[64];
        long n = wubu_ms_encode_header(out, sizeof(out));
        CHECK(n == (long)sizeof(kHeaderWire),
              "the header encodes to its known length");
        CHECK(memcmp(out, kHeaderWire, sizeof(kHeaderWire)) == 0,
              "the header is byte-identical to the known wire form");

        size_t pos = 0;
        const char *str = NULL;
        const uint8_t *data = NULL;
        size_t dl = 0;
        char scratch[64];
        CHECK(wubu_ms_read(out, (size_t)n, &pos, &str, scratch,
                           sizeof(scratch), &data, &dl) == WUBU_MS_STRING,
              "the header reads back as a string");
        CHECK(str && memcmp(str, WUBU_MS_HEADER, WUBU_MS_HEADER_LEN + 1) == 0,
              "and it is the header text");
        CHECK(pos == (size_t)n, "the position advanced past the whole frame");
        CHECK(wubu_ms_is_header(str, strlen(str)),
              "is_header recognises it");
    }

    /* ---- proposals ---- */
    {
        uint8_t out[64];
        long n = wubu_ms_encode("/ipfs/bitswap/1.2.0", out, sizeof(out));
        CHECK(n == 21, "a bare proposal gets its newline appended");
        CHECK(out[0] == 20, "the frame length counts the newline");

        uint8_t again[64];
        long m = wubu_ms_encode("/ipfs/bitswap/1.2.0\n", again, sizeof(again));
        CHECK(m == n && memcmp(out, again, (size_t)n) == 0,
              "an already-terminated proposal is not doubled");

        static const char *bad[] = {
            "",               /* empty */
            "/",              /* just a slash */
            "//x/1.0.0",      /* empty segment */
            "/x//1.0.0",      /* empty middle segment */
            "/x/",            /* trailing slash, no newline */
            "ipfs/1.0.0",     /* missing leading slash */
            "/x/y\ty",        /* a tab is not printable here */
            NULL
        };
        for (int i = 0; bad[i]; i++) {
            uint8_t b[64];
            memset(b, 0xA5, sizeof(b));
            CHECK(wubu_ms_encode(bad[i], b, sizeof(b)) == -1,
                  "a malformed proposal is refused");
        }
    }

    /* ---- na ---- */
    {
        uint8_t out[16];
        long n = wubu_ms_encode_na(out, sizeof(out));
        CHECK(n == 4 && out[0] == 3 && memcmp(out + 1, "na\n", 3) == 0,
              "'na' encodes to 03 6e 61 0a");
        size_t pos = 0;
        const char *str = NULL;
        const uint8_t *data = NULL;
        size_t dl = 0;
        char scratch[16];
        CHECK(wubu_ms_read(out, (size_t)n, &pos, &str, scratch,
                           sizeof(scratch), &data, &dl) == WUBU_MS_STRING &&
              str && strcmp(str, "na\n") == 0,
              "'na' reads back as the string");
    }

    /* ---- opaque data frames survive ---- */
    {
        uint8_t payload[8] = { 0x00, 0xff, 0x80, 0x0a, 0x7f, 0xc3, 0x28, 0x00 };
        uint8_t out[64];
        long n = wubu_ms_encode_data(payload, 8, out, sizeof(out));
        CHECK(n == 9, "a data frame is varint length plus payload");

        size_t pos = 0;
        const char *str = NULL;
        const uint8_t *data = NULL;
        size_t dl = 0;
        char scratch[16];
        CHECK(wubu_ms_read(out, (size_t)n, &pos, &str, scratch,
                           sizeof(scratch), &data, &dl) == WUBU_MS_DATA,
              "binary content is classified as DATA, not text");
        CHECK(dl == 8 && memcmp(data, payload, 8) == 0,
              "and the payload is bit-exact");
        CHECK(str == NULL, "a data frame yields no string");
    }

    /* ---- consecutive frames walk forward ---- */
    {
        uint8_t stream[128];
        size_t o = 0;
        uint8_t one[64];
        long n1 = wubu_ms_encode_header(one, sizeof(one));
        memcpy(stream + o, one, (size_t)n1);
        o += (size_t)n1;
        long n2 = wubu_ms_encode("/ipfs/id/1.0.0", one, sizeof(one));
        memcpy(stream + o, one, (size_t)n2);
        o += (size_t)n2;

        size_t pos = 0;
        const char *str = NULL;
        const uint8_t *data = NULL;
        size_t dl = 0;
        char scratch[64];

        CHECK(wubu_ms_read(stream, o, &pos, &str, scratch, sizeof(scratch),
                           &data, &dl) == WUBU_MS_STRING &&
              str && wubu_ms_is_header(str, strlen(str)),
              "the first frame of a stream is the header");
        CHECK(wubu_ms_read(stream, o, &pos, &str, scratch, sizeof(scratch),
                           &data, &dl) == WUBU_MS_STRING &&
              str && strcmp(str, "/ipfs/id/1.0.0\n") == 0,
              "the second frame is the proposal");
        CHECK(pos == o, "both frames were consumed");
    }

    /* ---- fail closed on reading ---- */
    {
        uint8_t good[64];
        long gn = wubu_ms_encode("/ipfs/id/1.0.0", good, sizeof(good));
        char scratch[64];
        const char *str = NULL;
        const uint8_t *data = NULL;
        size_t dl = 0;
        size_t pos = 0;
        memset(scratch, 0xA5, sizeof(scratch));

        /* Truncated: the declared length runs past the buffer. */
        CHECK(wubu_ms_read(good, (size_t)gn - 1, &pos, &str, scratch,
                           sizeof(scratch), &data, &dl) == -1,
              "a frame cut short is refused");

        /* A lying length prefix. */
        uint8_t lie[4] = { 0x63, 0x00, 0x00, 0x00 };   /* says 99, has 3 */
        pos = 0;
        CHECK(wubu_ms_read(lie, sizeof(lie), &pos, &str, scratch,
                           sizeof(scratch), &data, &dl) == -1,
              "a length larger than the buffer is refused");

        /* A body with no trailing newline is a legal DATA frame, so nothing
           is invented from it. */
        uint8_t raw[3] = { 0x02, 'A', 'B' };
        pos = 0;
        CHECK(wubu_ms_read(raw, sizeof(raw), &pos, &str, scratch,
                           sizeof(scratch), &data, &dl) == WUBU_MS_DATA,
              "a newline-free body is data, not a corrupt frame");

        /* No scratch buffer means a string frame cannot be handed over. */
        pos = 0;
        CHECK(wubu_ms_read(good, (size_t)gn, &pos, &str, NULL, 0,
                           &data, &dl) == -1,
              "a string frame without a scratch buffer is refused");

        /* NULL and empty inputs. */
        CHECK(wubu_ms_read(NULL, 0, &pos, &str, scratch, sizeof(scratch),
                           &data, &dl) == -1,
              "NULL is refused");
        pos = 5;
        CHECK(wubu_ms_read(good, (size_t)gn, &pos, &str, scratch,
                           sizeof(scratch), &data, &dl) == -1,
              "a position past the end is refused");
    }

    /* ---- negotiation ---- */
    {
        static const char *wanted[] = {
            "/ipfs/id/1.0.0",
            "/ipfs/bitswap/1.2.0",
            "/ipfs/kad/1.0.0"
        };

        CHECK(wubu_ms_select(wanted, 3, "/ipfs/bitswap/1.2.0\n",
                             strlen("/ipfs/bitswap/1.2.0\n")) == 1,
              "a proposed protocol selects its index");
        CHECK(wubu_ms_select(wanted, 3, "/ipfs/id/1.0.0\n",
                             strlen("/ipfs/id/1.0.0\n")) == 0,
              "the first match wins");
        CHECK(wubu_ms_select(wanted, 3, "/ipfs/garbage/9.9.9\n",
                             strlen("/ipfs/garbage/9.9.9\n")) == -1,
              "an unknown protocol selects nothing");
        CHECK(wubu_ms_select(wanted, 0, "/ipfs/id/1.0.0\n",
                             strlen("/ipfs/id/1.0.0\n")) == -1,
              "an empty wanted list selects nothing");

        /* A proposal that is malformed is not negotiable at all. */
        CHECK(wubu_ms_select(wanted, 3, "//bad\n", strlen("//bad\n")) == -1,
              "a malformed proposal selects nothing");
        CHECK(wubu_ms_select(wanted, 3, "no-slash\n", strlen("no-slash\n")) == -1,
              "a proposal without a leading slash selects nothing");
        CHECK(wubu_ms_select(wanted, 3, "/x\n", 0) == -1,
              "a zero-length proposal is refused");
        CHECK(wubu_ms_select(NULL, 3, "/ipfs/id/1.0.0\n",
                             strlen("/ipfs/id/1.0.0\n")) == -1,
              "a NULL wanted list is refused");

        /* An embedded NUL means the caller's string is lying about its length. */
        char sneaky[8] = { '/', 'x', '\n', '\0', 'y', '\n', '\0', '\0' };
        CHECK(wubu_ms_select(wanted, 3, sneaky, 7) == -1,
              "a proposal with an embedded NUL is refused");
    }

    /* ---- is_header is exact ---- */
    {
        CHECK(wubu_ms_is_header(WUBU_MS_HEADER, WUBU_MS_HEADER_LEN),
              "the real header is recognised");
        CHECK(!wubu_ms_is_header("/multistream/1.0.1\n", WUBU_MS_HEADER_LEN),
              "a one-version-off string of the same length is not the header");
        CHECK(!wubu_ms_is_header("/multistream/1.0.0", WUBU_MS_HEADER_LEN),
              "the same text without its newline is not the header");
        CHECK(!wubu_ms_is_header(WUBU_MS_HEADER, WUBU_MS_HEADER_LEN - 1),
              "a truncated header is not the header");
        CHECK(!wubu_ms_is_header(NULL, WUBU_MS_HEADER_LEN),
              "NULL is not the header");
        CHECK(!wubu_ms_is_header(WUBU_MS_HEADER, WUBU_MS_HEADER_LEN - 1),
              "a short length is not enough even when the text matches");
    }

    printf("=== %d checks, %d failed ===\n", checks, failures);
    return failures ? 1 : 0;
}
