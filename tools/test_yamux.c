/*
 * test_yamux.c -- a frame must describe exactly the bytes that follow it, and
 * nothing may be invented from a malformed one.  A mux that mis-frames splits
 * streams apart silently, which is worse than refusing the connection.
 */
#include "wubu_yamux.h"

#include <stdio.h>
#include <string.h>

static int failures, checks;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

/* Hand-checked: version 0, type DATA(1), flags SYN, stream 1, length 5,
 * payload "hello". */
static const uint8_t kDataHello[] = {
    0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x05, 'h', 'e', 'l', 'l', 'o'
};

/* PING with SYN and a 7-byte opaque code, no payload. */
static const uint8_t kPing[] = {
    0x00, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x07
};

int main(void)
{
    printf("=== test_yamux ===\n");

    /* ---- golden vectors ---- */
    {
        wubu_yamux_hdr_t h;
        memset(&h, 0, sizeof(h));
        h.type = WUBU_YAMUX_DATA;
        h.flags = WUBU_YAMUX_SYN;
        h.stream_id = 1;

        uint8_t out[64];
        long n = wubu_yamux_encode(&h, (const uint8_t *)"hello", 5,
                                   out, sizeof(out));
        CHECK(n == (long)sizeof(kDataHello),
              "a DATA frame encodes to its known length");
        CHECK(memcmp(out, kDataHello, sizeof(kDataHello)) == 0,
              "and is byte-identical to the hand-checked vector");

        memset(&h, 0, sizeof(h));
        h.type = WUBU_YAMUX_PING;
        h.flags = WUBU_YAMUX_SYN;
        h.length = 7;
        n = wubu_yamux_encode(&h, NULL, 0, out, sizeof(out));
        CHECK(n == (long)sizeof(kPing) && memcmp(out, kPing,
                                                 sizeof(kPing)) == 0,
              "a PING frame matches its hand-checked vector");
    }

    /* ---- every type round-trips ---- */
    {
        static const int types[] = {
            WUBU_YAMUX_DATA, WUBU_YAMUX_WINDOW_UPDATE,
            WUBU_YAMUX_PING, WUBU_YAMUX_GOAWAY
        };
        static const char *names[] = { "DATA", "WINDOW_UPDATE",
                                       "PING", "GOAWAY" };
        for (unsigned i = 0; i < 4; i++) {
            wubu_yamux_hdr_t h;
            memset(&h, 0, sizeof(h));
            h.type = types[i];
            h.flags = WUBU_YAMUX_SYN | WUBU_YAMUX_ACK;
            h.stream_id = 3;
            h.length = 4096;

            uint8_t out[64];
            const uint8_t *pl = NULL;
            size_t pl_len = 0;
            if (types[i] == WUBU_YAMUX_DATA) {
                pl = (const uint8_t *)"abcd";
                pl_len = 4;
            }
            long n = wubu_yamux_encode(&h, pl, pl_len, out, sizeof(out));
            CHECK(n > 0, names[i]);
            if (n <= 0) continue;

            size_t pos = 0;
            wubu_yamux_hdr_t g;
            memset(&g, 0xA5, sizeof(g));
            const uint8_t *back = NULL;
            size_t blen = 0;
            CHECK(wubu_yamux_decode(out, (size_t)n, &pos, &g, &back, &blen) == 0,
                  names[i]);
            CHECK(g.type == types[i], names[i]);
            CHECK(g.flags == (WUBU_YAMUX_SYN | WUBU_YAMUX_ACK), names[i]);
            CHECK(g.stream_id == 3, names[i]);
            CHECK(pos == (size_t)n, names[i]);

            if (types[i] == WUBU_YAMUX_DATA) {
                CHECK(g.length == 4 && blen == 4 &&
                      memcmp(back, "abcd", 4) == 0,
                      "the DATA payload survives");
            } else {
                CHECK(blen == 0 && back == NULL,
                      "a control frame yields no payload");
            }
        }
    }

    /* ---- consecutive frames walk forward ---- */
    {
        uint8_t stream[128];
        size_t o = 0;
        wubu_yamux_hdr_t h;
        memset(&h, 0, sizeof(h));
        h.type = WUBU_YAMUX_DATA;
        h.stream_id = 1;

        h.flags = WUBU_YAMUX_SYN;
        o += (size_t)wubu_yamux_encode(&h, (const uint8_t *)"ab", 2,
                                       stream + o, sizeof(stream) - o);
        h.flags = 0;
        o += (size_t)wubu_yamux_encode(&h, (const uint8_t *)"cd", 2,
                                       stream + o, sizeof(stream) - o);
        h.flags = WUBU_YAMUX_FIN;
        o += (size_t)wubu_yamux_encode(&h, (const uint8_t *)"ef", 2,
                                       stream + o, sizeof(stream) - o);

        size_t pos = 0;
        wubu_yamux_hdr_t g;
        const uint8_t *back = NULL;
        size_t blen = 0;

        CHECK(wubu_yamux_decode(stream, o, &pos, &g, &back, &blen) == 0 &&
              g.flags == WUBU_YAMUX_SYN,
              "frame 1 is the SYN");
        CHECK(wubu_yamux_decode(stream, o, &pos, &g, &back, &blen) == 0 &&
              g.flags == 0,
              "frame 2 is plain data");
        CHECK(wubu_yamux_decode(stream, o, &pos, &g, &back, &blen) == 0 &&
              g.flags == WUBU_YAMUX_FIN,
              "frame 3 is the FIN");
        CHECK(pos == o, "every byte of the stream was consumed");
        CHECK(wubu_yamux_decode(stream, o, &pos, &g, &back, &blen) == -1,
              "a fourth read past the end is refused");
    }

    /* ---- fail closed ---- */
    {
        uint8_t out[64];
        memcpy(out, kDataHello, sizeof(kDataHello));

        /* Wrong version. */
        out[0] = 1;
        size_t pos = 0;
        wubu_yamux_hdr_t g;
        const uint8_t *back = NULL;
        size_t blen = 0;
        CHECK(wubu_yamux_decode(out, sizeof(kDataHello), &pos, &g, &back,
                                &blen) == -1,
              "a non-zero version is refused");

        /* Unknown type. */
        memcpy(out, kDataHello, sizeof(kDataHello));
        out[1] = 9;
        pos = 0;
        CHECK(wubu_yamux_decode(out, sizeof(kDataHello), &pos, &g, &back,
                                &blen) == -1,
              "an unknown type is refused");

        /* Undefined flag bits. */
        memcpy(out, kDataHello, sizeof(kDataHello));
        out[3] = 0x10;                       /* bit 4 of the flags field */
        pos = 0;
        CHECK(wubu_yamux_decode(out, sizeof(kDataHello), &pos, &g, &back,
                                &blen) == -1,
              "undefined flag bits are refused");

        /* Truncated header. */
        pos = 0;
        CHECK(wubu_yamux_decode(kDataHello, WUBU_YAMUX_HEADER - 1, &pos, &g,
                                &back, &blen) == -1,
              "a truncated header is refused");

        /* A declared payload longer than the buffer. */
        uint8_t lie[12];
        memcpy(lie, kDataHello, sizeof(lie));
        lie[11] = 200;                       /* says 200 bytes follow */
        pos = 0;
        CHECK(wubu_yamux_decode(lie, sizeof(lie), &pos, &g, &back,
                                &blen) == -1,
              "a length past the end is refused");

        /* A control frame carrying payload. */
        wubu_yamux_hdr_t h;
        memset(&h, 0, sizeof(h));
        h.type = WUBU_YAMUX_PING;
        h.flags = WUBU_YAMUX_SYN;
        CHECK(wubu_yamux_encode(&h, (const uint8_t *)"x", 1,
                                out, sizeof(out)) == -1,
              "a PING with a payload is refused");

        /* A DATA frame's length is derived, so a caller cannot lie about it. */
        memset(&h, 0, sizeof(h));
        h.type = WUBU_YAMUX_DATA;
        h.stream_id = 1;
        h.length = 999;                      /* deliberately wrong */
        long n = wubu_yamux_encode(&h, (const uint8_t *)"hi", 2,
                                   out, sizeof(out));
        CHECK(n == WUBU_YAMUX_HEADER + 2 && out[11] == 2,
              "a DATA frame's wire length comes from the payload, not the field");

        /* Buffer too small. */
        CHECK(wubu_yamux_encode(&h, (const uint8_t *)"hi", 2, out, 8) == -1,
              "a buffer too small is refused rather than truncated");

        /* NULL arguments. */
        CHECK(wubu_yamux_encode(NULL, NULL, 0, out, sizeof(out)) == -1,
              "a NULL header is refused");
        CHECK(wubu_yamux_encode(&h, NULL, 0, NULL, 0) == -1,
              "a NULL output is refused");
        CHECK(wubu_yamux_decode(NULL, 12, &pos, &g, &back, &blen) == -1,
              "a NULL buffer is refused");
        pos = 5;
        CHECK(wubu_yamux_decode(kDataHello, sizeof(kDataHello), &pos, &g,
                                &back, &blen) == -1,
              "a position inside the header is refused");
    }

    /* ---- stream id parity ---- */
    {
        CHECK(wubu_yamux_is_client(1) && !wubu_yamux_is_server(1),
              "odd stream IDs are client-initiated");
        CHECK(wubu_yamux_is_server(2) && !wubu_yamux_is_client(2),
              "even stream IDs are server-initiated");
        CHECK(wubu_yamux_is_client(0xFFFFFFFFu) &&
              wubu_yamux_is_server(0),
              "the extremes keep their parity");
    }

    printf("=== %d checks, %d failed ===\n", checks, failures);
    return failures ? 1 : 0;
}
