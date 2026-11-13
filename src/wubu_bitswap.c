/*
 * wubu_bitswap.c -- bitswap message shapes.
 *
 * C11, libc only.  The protobuf writer appends into a caller buffer and never
 * allocates; the reader walks with an explicit position and refuses to run
 * past the end or past the message.
 */
#include "wubu_bitswap.h"

#include <string.h>
#include <stdio.h>

/* ---- protobuf primitives --------------------------------------- */

/* protobuf varints differ from multiformats ones in exactly one way that
 * matters here: a negative int32 is sign-extended to a 64-bit value, so -1 is
 * written as ten bytes (ff x9, 01).  The multiformats 9-byte limit would
 * refuse that, and refusing it would emit a protobuf peer cannot read.
 * Minimality is still enforced -- no wasted bytes either way. */
long wubu_pb_varint(uint8_t *out, size_t n, uint64_t v)
{
    if (!out) return -1;
    uint8_t tmp[10];
    size_t o = 0;
    do {
        if (o >= sizeof(tmp)) return -1;
        uint8_t b = (uint8_t)(v & 0x7F);
        v >>= 7;
        if (v) b |= 0x80;
        tmp[o++] = b;
    } while (v);
    if (o > n) return -1;
    memcpy(out, tmp, o);
    return (long)o;
}

long wubu_pb_tag(uint8_t *out, size_t n, unsigned field, int wire)
{
    if (field == 0 || field > 0x1FFFFFFFu) return -1;
    return wubu_pb_varint(out, n, ((uint64_t)field << 3) | (uint64_t)(wire & 7));
}

long wubu_pb_varint_field(uint8_t *out, size_t n, unsigned field, uint64_t v)
{
    long t = wubu_pb_tag(out, n, field, WUBU_PB_VARINT);
    if (t < 0) return -1;
    long b = wubu_pb_varint(out + t, n - (size_t)t, v);
    if (b < 0) return -1;
    return t + b;
}

long wubu_pb_bytes_field(uint8_t *out, size_t n, unsigned field,
                         const uint8_t *data, size_t len)
{
    long t = wubu_pb_tag(out, n, field, WUBU_PB_BYTES);
    if (t < 0) return -1;
    long l = wubu_pb_varint(out + t, n - (size_t)t, (uint64_t)len);
    if (l < 0) return -1;
    size_t used = (size_t)t + (size_t)l;
    if (used + len > n) return -1;
    if (len && data) memcpy(out + used, data, len);
    return (long)(used + len);
}

int wubu_pb_read_header(const uint8_t *in, size_t n, size_t *pos,
                        unsigned *field, int *wire, uint64_t *value,
                        const uint8_t **bytes, size_t *len)
{
    if (!in || !pos || !field || !wire || !value || !bytes || !len) return -1;

    size_t p = *pos;
    uint64_t tag = 0;
    size_t used = 0;
    if (wubu_varint_decode(in + p, n - p, &tag, &used) != 0) return -1;
    p += used;

    *field = (unsigned)((tag >> 3) & 0x0FFFFFFFu);
    *wire  = (int)(tag & 7);
    if (*field == 0) return -1;

    switch (*wire) {
    case WUBU_PB_VARINT:
        if (wubu_varint_decode(in + p, n - p, value, &used) != 0) return -1;
        p += used;
        break;
    case WUBU_PB_BYTES: {
        uint64_t blen = 0;
        if (wubu_varint_decode(in + p, n - p, &blen, &used) != 0) return -1;
        p += used;
        if (blen > n - p) return -1;
        *bytes = in + p;
        *len   = (size_t)blen;
        p += (size_t)blen;
        break;
    }
    default:
        return -1;                   /* 64-bit / start-group / 32-bit: unused */
    }

    *pos = p;
    return 0;
}

/* ---- bitswap shapes -------------------------------------------- */

long wubu_bitswap_encode_entry(const wubu_bitswap_entry_t *e,
                               uint8_t *out, size_t n)
{
    if (!e || !out || !e->cid_len) return -1;

    size_t o = 0;
    long k;

#define APPEND(call) do {                                            \
        k = (call);                                                  \
        if (k < 0) return -1;                                        \
        o += (size_t)k;                                              \
    } while (0)

    APPEND(wubu_pb_bytes_field(out + o, n - o, 1, e->cid, e->cid_len));
    if (e->priority != 0)
        APPEND(wubu_pb_varint_field(out + o, n - o, 2, (uint64_t)e->priority));
    if (e->cancel)
        APPEND(wubu_pb_varint_field(out + o, n - o, 3, 1));
    if (e->want_type != WUBU_BITSWAP_WANT_BLOCK)
        APPEND(wubu_pb_varint_field(out + o, n - o, 4, (uint64_t)e->want_type));
    if (e->send_dont_have)
        APPEND(wubu_pb_varint_field(out + o, n - o, 5, 1));

#undef APPEND
    return (long)o;
}

long wubu_bitswap_encode_wantlist(const wubu_bitswap_wantlist_t *wl,
                                  uint8_t *out, size_t n)
{
    if (!wl || !out || wl->count > WUBU_BITSWAP_MAX_ENTRIES) return -1;

    size_t o = 0;
    for (int i = 0; i < wl->count; i++) {
        uint8_t entry[512];
        long en = wubu_bitswap_encode_entry(&wl->entries[i], entry,
                                            sizeof(entry));
        if (en < 0) return -1;
        long k = wubu_pb_bytes_field(out + o, n - o, 1, entry, (size_t)en);
        if (k < 0) return -1;
        o += (size_t)k;
    }
    if (wl->full) {
        long k = wubu_pb_varint_field(out + o, n - o, 2, 1);
        if (k < 0) return -1;
        o += (size_t)k;
    }
    return (long)o;
}

long wubu_bitswap_encode_block(const wubu_bitswap_block_t *b,
                               uint8_t *out, size_t n)
{
    if (!b || !out || !b->data_len || !b->data) return -1;
    if (b->prefix_len >= WUBU_BITSWAP_MAX_PREFIX) return -1;

    size_t o = 0;
    if (b->prefix_len) {
        long k = wubu_pb_bytes_field(out, n, 1, b->prefix, b->prefix_len);
        if (k < 0) return -1;
        o += (size_t)k;
    }
    long k = wubu_pb_bytes_field(out + o, n - o, 2, b->data, b->data_len);
    if (k < 0) return -1;
    o += (size_t)k;
    return (long)o;
}

long wubu_bitswap_encode_presence(const wubu_bitswap_block_presence_t *p,
                                  uint8_t *out, size_t n)
{
    if (!p || !out || !p->cid_len) return -1;

    size_t o = 0;
    long k = wubu_pb_bytes_field(out, n, 1, p->cid, p->cid_len);
    if (k < 0) return -1;
    o += (size_t)k;
    if (p->presence != WUBU_BITSWAP_HAVE) {
        k = wubu_pb_varint_field(out + o, n - o, 2, (uint64_t)p->presence);
        if (k < 0) return -1;
        o += (size_t)k;
    }
    return (long)o;
}

/* ---- decoding --------------------------------------------------- */

static int decode_entry(const uint8_t *in, size_t n, wubu_bitswap_entry_t *e)
{
    memset(e, 0, sizeof(*e));
    size_t pos = 0;
    while (pos < n) {
        unsigned field = 0;
        int wire = 0;
        uint64_t value = 0;
        const uint8_t *bytes = NULL;
        size_t blen = 0;

        if (wubu_pb_read_header(in, n, &pos, &field, &wire, &value,
                                &bytes, &blen) != 0)
            return -1;
        switch (field) {
        case 1:
            if (wire != WUBU_PB_BYTES || blen > WUBU_BITSWAP_MAX_CID) return -1;
            memcpy(e->cid, bytes, blen);
            e->cid_len = blen;
            break;
        case 2: e->priority = (int32_t)(uint32_t)value; break;
        case 3: e->cancel = value != 0;                  break;
        case 4: e->want_type = (int)value;               break;
        case 5: e->send_dont_have = value != 0;          break;
        default: return -1;      /* unknown field inside an Entry */
        }
    }
    return e->cid_len ? 0 : -1;
}

static int decode_wantlist(const uint8_t *in, size_t n,
                           wubu_bitswap_wantlist_t *wl)
{
    size_t pos = 0;
    while (pos < n) {
        unsigned field = 0;
        int wire = 0;
        uint64_t value = 0;
        const uint8_t *bytes = NULL;
        size_t blen = 0;

        if (wubu_pb_read_header(in, n, &pos, &field, &wire, &value,
                                &bytes, &blen) != 0)
            return -1;
        if (field == 1) {
            if (wire != WUBU_PB_BYTES) return -1;
            if (wl->count >= WUBU_BITSWAP_MAX_ENTRIES) return -1;
            if (decode_entry(bytes, blen,
                             &wl->entries[wl->count]) != 0)
                return -1;
            wl->count++;
        } else if (field == 2) {
            if (wire != WUBU_PB_VARINT) return -1;
            wl->full = value != 0;
        } else {
            return -1;
        }
    }
    return 0;
}

long wubu_bitswap_encode_message(const wubu_bitswap_wantlist_t *wl,
                                 const wubu_bitswap_block_t *blocks,
                                 int block_count,
                                 const wubu_bitswap_block_presence_t *presences,
                                 int presence_count,
                                 uint8_t *out, size_t n)
{
    if (!out || block_count < 0 || presence_count < 0) return -1;
    if (block_count > WUBU_BITSWAP_MAX_ENTRIES ||
        presence_count > WUBU_BITSWAP_MAX_ENTRIES) return -1;

    size_t o = 0;

    /* Message.wantlist = field 1 */
    if (wl) {
        uint8_t sub[4096];
        long sn = wubu_bitswap_encode_wantlist(wl, sub, sizeof(sub));
        if (sn < 0) return -1;
        long k = wubu_pb_bytes_field(out, n, 1, sub, (size_t)sn);
        if (k < 0) return -1;
        o += (size_t)k;
    }

    /* Message.block = field 2 (deprecated) */
    for (int i = 0; i < block_count; i++) {
        uint8_t sub[2048];
        long sn = wubu_bitswap_encode_block(&blocks[i], sub, sizeof(sub));
        if (sn < 0) return -1;
        long k = wubu_pb_bytes_field(out + o, n - o, 2, sub, (size_t)sn);
        if (k < 0) return -1;
        o += (size_t)k;
    }

    /* Message.blockPresences = field 3 */
    for (int i = 0; i < presence_count; i++) {
        uint8_t sub[256];
        long sn = wubu_bitswap_encode_presence(&presences[i],
                                               sub, sizeof(sub));
        if (sn < 0) return -1;
        long k = wubu_pb_bytes_field(out + o, n - o, 3, sub, (size_t)sn);
        if (k < 0) return -1;
        o += (size_t)k;
    }
    return (long)o;
}

int wubu_bitswap_decode(const uint8_t *buf, size_t n,
                        wubu_bitswap_message_t *out)
{
    if (!out) return -1;
    /* An empty buffer is a valid, empty protobuf message -- not an error.
       Refusing it would make an absent payload indistinguishable from a
       corrupt one.  A NULL buffer with a non-zero length is still a lie. */
    /* n == 0 needs no buffer at all, so NULL is acceptable here. */
    if (n == 0) {
        memset(out, 0, sizeof(*out));
        return 0;
    }
    if (!buf) return -1;

    wubu_bitswap_message_t m;
    memset(&m, 0, sizeof(m));

    size_t pos = 0;
    while (pos < n) {
        unsigned field = 0;
        int wire = 0;
        uint64_t value = 0;
        const uint8_t *bytes = NULL;
        size_t blen = 0;

        if (wubu_pb_read_header(buf, n, &pos, &field, &wire, &value,
                                &bytes, &blen) != 0)
            return -1;

        switch (field) {
        case 1:                                   /* Message.wantlist */
            if (m.have_wantlist || wire != WUBU_PB_BYTES) return -1;
            if (decode_wantlist(bytes, blen, &m.wantlist) != 0) return -1;
            m.have_wantlist = 1;
            break;

        case 2:                                   /* Message.block: the
                                                     deprecated repeated-bytes
                                                     form carries raw blocks
                                                     with no structure we can
                                                     validate, so it is refused
                                                     rather than half-read. */
            return -1;

        case 3: {                                 /* Message.blockPresences */
            if (wire != WUBU_PB_BYTES) return -1;
            if (m.presence_count >= WUBU_BITSWAP_MAX_ENTRIES) return -1;

            wubu_bitswap_block_presence_t bp;
            memset(&bp, 0, sizeof(bp));
            size_t pp = 0;
            while (pp < blen) {
                unsigned pf = 0;
                int pw = 0;
                uint64_t pv = 0;
                const uint8_t *pb = NULL;
                size_t pl = 0;
                if (wubu_pb_read_header(bytes, blen, &pp, &pf, &pw, &pv,
                                        &pb, &pl) != 0)
                    return -1;
                switch (pf) {
                case 1:
                    if (pw != WUBU_PB_BYTES || pl > WUBU_BITSWAP_MAX_CID)
                        return -1;
                    memcpy(bp.cid, pb, pl);
                    bp.cid_len = pl;
                    break;
                case 2: bp.presence = (int)pv; break;
                default: return -1;
                }
            }
            if (!bp.cid_len) return -1;
            m.presences[m.presence_count++] = bp;
            break;
        }

        default:
            return -1;                            /* unknown Message field */
        }
    }

    memcpy(out, &m, sizeof(m));
    return 0;
}
