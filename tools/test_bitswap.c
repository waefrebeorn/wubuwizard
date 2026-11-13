/*
 * test_bitswap.c -- the block-exchange messages must carry exactly what they
 * say, and refuse anything whose structure cannot be validated.
 *
 * bitswap rides protobuf, so the primitives are tested directly as well as the
 * message shapes: a wrong tag or a length that runs past the end would put a
 * peer on the wrong block forever.
 */
#include "wubu_bitswap.h"

#include <stdio.h>
#include <string.h>

static int failures, checks;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

/* A two-entry wantlist plus one DONT_HAVE presence.  This exact byte sequence
 * was checked by hand against the go-bitswap schema:
 *   Message{ wantlist: Wantlist{ entries:[Entry{cid,priority},
 *                                            Entry{cid,cancel,sendDontHave}],
 *                               full:true },
 *            blockPresences:[ BlockPresence{cid,type:DONT_HAVE} ] }          */
static const uint8_t kGolden[] = {
    0x0a, 0x20,              /* field 1 (wantlist), len 32 */
    0x0a, 0x10,              /*   field 1 (entry), len 16 */
    0x0a, 0x03, 0x01, 0x55, 0x12,
    0x10, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01,
                                 /* priority -1, sign-extended to 64 bits */
    0x0a, 0x0a,              /*   field 1 (entry), len 10 */
    0x0a, 0x04, 0x01, 0x70, 0x12, 0x20,
    0x18, 0x01,              /*     cancel = true (field 3) */
    0x28, 0x01,              /*     sendDontHave = true (field 5) */
    0x10, 0x01,              /*   full = true (field 2) */
    0x1a, 0x07,              /* field 3 (blockPresences), len 7 */
    0x0a, 0x03, 0x01, 0x55, 0x12,
    0x10, 0x01               /*   type = DONT_HAVE */
};

static void fill_wantlist(wubu_bitswap_wantlist_t *wl)
{
    memset(wl, 0, sizeof(*wl));
    uint8_t cid1[3] = { 0x01, 0x55, 0x12 };
    uint8_t cid2[4] = { 0x01, 0x70, 0x12, 0x20 };
    memcpy(wl->entries[0].cid, cid1, 3);
    wl->entries[0].cid_len = 3;
    wl->entries[0].priority = -1;
    memcpy(wl->entries[1].cid, cid2, 4);
    wl->entries[1].cid_len = 4;
    wl->entries[1].cancel = 1;
    wl->entries[1].send_dont_have = 1;
    wl->count = 2;
    wl->full = 1;
}

int main(void)
{
    printf("=== test_bitswap ===\n");

    /* ---- protobuf primitives ---- */
    {
        uint8_t b[16];
        CHECK(wubu_pb_tag(b, sizeof(b), 1, WUBU_PB_VARINT) == 1 && b[0] == 0x08,
              "field 1 / varint tags as 0x08");
        CHECK(wubu_pb_tag(b, sizeof(b), 2, WUBU_PB_BYTES) == 1 && b[0] == 0x12,
              "field 2 / bytes tags as 0x12");
        CHECK(wubu_pb_tag(b, sizeof(b), 0, WUBU_PB_VARINT) == -1,
              "field number zero is refused");
        CHECK(wubu_pb_tag(b, sizeof(b), 0x20000000u, WUBU_PB_VARINT) == -1,
              "a field number past the spec range is refused");

        CHECK(wubu_pb_varint_field(b, sizeof(b), 2, 300) == 3 &&
              b[0] == 0x10 && b[1] == 0xAC && b[2] == 0x02,
              "a varint field encodes 300 as 0xAC 0x02");
        CHECK(wubu_pb_bytes_field(b, sizeof(b), 1, (const uint8_t *)"ab", 2)
              == 4 && b[0] == 0x0a && b[1] == 0x02,
              "a bytes field carries its length");
        CHECK(wubu_pb_bytes_field(b, sizeof(b), 1,
                                  (const uint8_t *)"abcdef", 6) == -1 ||
              wubu_pb_bytes_field(NULL, 0, 1, NULL, 0) == -1,
              "bad buffer arguments are refused");

        size_t pos = 0;
        unsigned field = 0;
        int wire = 0;
        uint64_t val = 0;
        const uint8_t *bytes = NULL;
        size_t blen = 0;
        static const uint8_t msg[] = { 0x0a, 0x02, 'h', 'i' };
        CHECK(wubu_pb_read_header(msg, sizeof(msg), &pos, &field, &wire,
                                  &val, &bytes, &blen) == 0 &&
              field == 1 && wire == WUBU_PB_BYTES && blen == 2 &&
              memcmp(bytes, "hi", 2) == 0 && pos == 4,
              "a length-delimited header reads its span");
        CHECK(wubu_pb_read_header(msg, 1, &pos, &field, &wire, &val,
                                  &bytes, &blen) == -1,
              "a header past the end is refused");
        static const uint8_t fixed64[] = { 0x09, 1, 2, 3, 4, 5, 6, 7, 8 };
        pos = 0;
        CHECK(wubu_pb_read_header(fixed64, sizeof(fixed64), &pos, &field,
                                  &wire, &val, &bytes, &blen) == -1,
              "a 64-bit field is refused (bitswap never uses one)");
    }

    /* ---- the golden message byte-for-byte ---- */
    {
        wubu_bitswap_wantlist_t wl;
        fill_wantlist(&wl);
        wubu_bitswap_block_presence_t bp;
        memset(&bp, 0, sizeof(bp));
        uint8_t cid[3] = { 0x01, 0x55, 0x12 };
        memcpy(bp.cid, cid, 3);
        bp.cid_len = 3;
        bp.presence = WUBU_BITSWAP_DONT_HAVE;

        uint8_t out[WUBU_BITSWAP_MAX_MESSAGE];
        long n = wubu_bitswap_encode_message(&wl, NULL, 0, &bp, 1,
                                             out, sizeof(out));
        CHECK(n == (long)sizeof(kGolden),
              "the encoded message matches the hand-checked length");
        CHECK(memcmp(out, kGolden, sizeof(kGolden)) == 0,
              "the encoded message matches the hand-checked bytes");
    }

    /* ---- round trip: everything survives ---- */
    {
        wubu_bitswap_wantlist_t wl;
        fill_wantlist(&wl);
        wubu_bitswap_block_presence_t bp;
        memset(&bp, 0, sizeof(bp));
        uint8_t cid[3] = { 0x01, 0x55, 0x12 };
        memcpy(bp.cid, cid, 3);
        bp.cid_len = 3;
        bp.presence = WUBU_BITSWAP_DONT_HAVE;

        uint8_t out[WUBU_BITSWAP_MAX_MESSAGE];
        long n = wubu_bitswap_encode_message(&wl, NULL, 0, &bp, 1,
                                             out, sizeof(out));
        CHECK(n > 0, "the message encodes");

        wubu_bitswap_message_t m;
        CHECK(wubu_bitswap_decode(out, (size_t)n, &m) == 0,
              "the message decodes");
        CHECK(m.have_wantlist, "the wantlist is present");
        CHECK(m.wantlist.count == 2, "both entries arrive");
        CHECK(m.wantlist.full, "the full flag arrives");
        CHECK(m.wantlist.entries[0].cid_len == 3 &&
              memcmp(m.wantlist.entries[0].cid,
                     wl.entries[0].cid, 3) == 0,
              "entry 0 keeps its CID");
        CHECK(m.wantlist.entries[0].priority == -1,
              "a negative priority survives (sign-extended on the wire)");
        CHECK(!m.wantlist.entries[0].cancel &&
              !m.wantlist.entries[0].send_dont_have,
              "absent flags stay false");
        CHECK(m.wantlist.entries[1].cancel &&
              m.wantlist.entries[1].send_dont_have,
              "entry 1's flags arrive");
        CHECK(m.presence_count == 1 && m.presences[0].presence ==
              WUBU_BITSWAP_DONT_HAVE,
              "the presence arrives with its type");
        CHECK(m.presences[0].cid_len == 3,
              "the presence keeps its CID");
    }

    /* ---- an empty wantlist and an absent wantlist both decode ---- */
    {
        wubu_bitswap_message_t m;
        CHECK(wubu_bitswap_decode(NULL, 0, &m) == 0 && !m.have_wantlist,
              "a zero-length buffer decodes as an empty message");

        wubu_bitswap_wantlist_t wl;
        memset(&wl, 0, sizeof(wl));
        uint8_t out[64];
        long n = wubu_bitswap_encode_message(&wl, NULL, 0, NULL, 0,
                                             out, sizeof(out));
        /* A present-but-empty wantlist is still a field: 0a 00. */
        CHECK(n == 2 && out[0] == 0x0a && out[1] == 0x00,
              "an empty wantlist still emits its field header");
        CHECK(wubu_bitswap_decode(out, (size_t)n, &m) == 0 &&
              m.have_wantlist && m.wantlist.count == 0 && !m.wantlist.full,
              "the empty wantlist decodes back to empty");
    }

    /* ---- blocks ---- */
    {
        wubu_bitswap_block_t b;
        memset(&b, 0, sizeof(b));
        uint8_t pre[2] = { 0x12, 0x20 };
        uint8_t data[5] = { 'h', 'e', 'l', 'l', 'o' };
        memcpy(b.prefix, pre, 2);
        b.prefix_len = 2;
        b.data = data;
        b.data_len = 5;

        uint8_t sub[64];
        CHECK(wubu_bitswap_encode_block(&b, sub, sizeof(sub)) > 0,
              "a block encodes");
        CHECK(wubu_bitswap_encode_block(NULL, sub, sizeof(sub)) == -1,
              "a NULL block is refused");
        b.data = NULL;
        CHECK(wubu_bitswap_encode_block(&b, sub, sizeof(sub)) == -1,
              "a block with no payload is refused");
    }

    /* ---- fail closed ---- */
    {
        wubu_bitswap_message_t untouched;
        memset(&untouched, 0xA5, sizeof(untouched));

        /* A present-but-empty wantlist: the smallest valid message. */
        static const uint8_t good[] = { 0x0a, 0x00 };

        CHECK(wubu_bitswap_decode(NULL, 0, &untouched) == 0,
              "a NULL buffer with zero length is an empty message");
        CHECK(wubu_bitswap_decode(NULL, 4, &untouched) == -1,
              "a NULL buffer WITH a length is refused");

        CHECK(wubu_bitswap_decode(good, sizeof(good), &untouched) == 0,
              "a minimal message decodes");

        /* Truncated inside a length-delimited field. */
        static const uint8_t trunc[] = { 0x0a, 0x20, 0x08, 0x01 };
        CHECK(wubu_bitswap_decode(trunc, sizeof(trunc), &untouched) == -1,
              "a length running past the end is refused");

        /* An unknown field in Message. */
        static const uint8_t unknown_msg[] = { 0x32, 0x01, 0x00 };
        CHECK(wubu_bitswap_decode(unknown_msg, sizeof(unknown_msg),
                                  &untouched) == -1,
              "an unknown Message field is refused");

        /* An unknown field inside an Entry. */
        static const uint8_t unknown_entry[] = {
            0x0a, 0x04, 0x0a, 0x02, 0x01, 0x55, 0x38, 0x01
        };
        CHECK(wubu_bitswap_decode(unknown_entry, sizeof(unknown_entry),
                                  &untouched) == -1,
              "an unknown Entry field is refused");

        /* A repeated wantlist. */
        static const uint8_t twice[] = {
            0x0a, 0x02, 0x0a, 0x00,   /* wantlist with one empty entry */
            0x0a, 0x02, 0x0a, 0x00
        };
        CHECK(wubu_bitswap_decode(twice, sizeof(twice), &untouched) == -1,
              "a repeated wantlist is refused");

        /* An entry with no CID is not an entry. */
        static const uint8_t no_cid[] = { 0x0a, 0x02, 0x18, 0x01 };
        CHECK(wubu_bitswap_decode(no_cid, sizeof(no_cid), &untouched) == -1,
              "an entry without a CID is refused");

        /* Trailing bytes after a complete message. */
        uint8_t tail[sizeof(good) + 1];
        memcpy(tail, good, sizeof(good));
        tail[sizeof(good)] = 0xFF;
        CHECK(wubu_bitswap_decode(tail, sizeof(tail), &untouched) == -1,
              "trailing bytes are refused");

        /* Every refusal above must leave the caller's struct alone. */
        wubu_bitswap_message_t fresh;
        memset(&fresh, 0xA5, sizeof(fresh));
        CHECK(wubu_bitswap_decode(trunc, sizeof(trunc), &fresh) == -1,
              "a truncated message is refused again");
        {
            int same = 1;
            for (size_t i = 0; i < sizeof(fresh); i++)
                if (((uint8_t *)&fresh)[i] != 0xA5) same = 0;
            CHECK(same, "a refused decode leaks nothing into the caller's struct");
        }
    }

    /* ---- build-side refusals ---- */
    {
        wubu_bitswap_wantlist_t wl;
        fill_wantlist(&wl);
        uint8_t out[256];
        CHECK(wubu_bitswap_encode_message(NULL, NULL, 0, NULL, 0,
                                          out, sizeof(out)) == 0,
              "an entirely empty message encodes");
        CHECK(wubu_bitswap_encode_message(&wl, NULL, 0, NULL, 0,
                                          NULL, 0) == -1,
              "a NULL buffer is refused");
        CHECK(wubu_bitswap_encode_message(NULL, NULL, 65, NULL, 0,
                                          out, sizeof(out)) == -1,
              "more blocks than the table holds are refused");
        CHECK(wubu_bitswap_encode_message(NULL, NULL, 0, NULL, -1,
                                          out, sizeof(out)) == -1,
              "a negative count is refused");
        CHECK(wubu_bitswap_encode_message(&wl, NULL, 0, NULL, 0,
                                          out, 8) == -1,
              "a buffer too small is refused rather than truncated");
    }

    printf("=== %d checks, %d failed ===\n", checks, failures);
    return failures ? 1 : 0;
}
