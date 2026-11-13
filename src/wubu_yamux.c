/*
 * wubu_yamux.c -- yamux frame framing.
 *
 * C11, libc only.  All multi-byte integers are big-endian on the wire; this
 * module converts explicitly rather than assuming host order.
 */
#include "wubu_yamux.h"

#include <string.h>
#include <stdio.h>

static void put_be16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v >> 8);
    p[1] = (uint8_t)(v & 0xFF);
}

static void put_be32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)(v & 0xFF);
}

static uint16_t get_be16(const uint8_t *p)
{
    return (uint16_t)(((unsigned)p[0] << 8) | p[1]);
}

static uint32_t get_be32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

int wubu_yamux_is_client(uint32_t stream_id)
{
    return (stream_id & 1u) != 0;
}

int wubu_yamux_is_server(uint32_t stream_id)
{
    return !wubu_yamux_is_client(stream_id);
}

long wubu_yamux_encode(const wubu_yamux_hdr_t *h, const uint8_t *payload,
                       size_t payload_len, uint8_t *out, size_t n)
{
    if (!h || !out) return -1;

    switch (h->type) {
    case WUBU_YAMUX_DATA:
    case WUBU_YAMUX_WINDOW_UPDATE:
    case WUBU_YAMUX_PING:
    case WUBU_YAMUX_GOAWAY:
        break;
    default:
        return -1;                       /* unknown type */
    }
    if (h->flags & ~WUBU_YAMUX_KNOWN_FLAGS) return -1;

    int carries_payload = (h->type == WUBU_YAMUX_DATA);
    if (!carries_payload && payload_len != 0) return -1;

    /* For a DATA frame the wire length IS the payload length.  Deriving it
       here rather than trusting h->length means a caller cannot claim a length
       that does not match the bytes it actually handed over. */
    uint32_t len_field = carries_payload ? (uint32_t)payload_len : h->length;

    if (payload_len > WUBU_YAMUX_MAX_LEN) return -1;
    if ((size_t)WUBU_YAMUX_HEADER + payload_len > n) return -1;

    out[0] = WUBU_YAMUX_VERSION;
    out[1] = (uint8_t)h->type;
    put_be16(out + 2, (uint16_t)h->flags);
    put_be32(out + 4, h->stream_id);
    put_be32(out + 8, len_field);
    if (payload_len) memcpy(out + WUBU_YAMUX_HEADER, payload, payload_len);

    return WUBU_YAMUX_HEADER + (long)payload_len;
}

int wubu_yamux_decode(const uint8_t *buf, size_t n, size_t *pos,
                      wubu_yamux_hdr_t *h, const uint8_t **payload,
                      size_t *payload_len)
{
    if (!buf || !pos || !h || !payload || !payload_len) return -1;
    if (*pos >= n) return -1;
    if (n - *pos < WUBU_YAMUX_HEADER) return -1;   /* truncated header */

    size_t p = *pos;
    if (buf[p] != WUBU_YAMUX_VERSION) return -1;

    int type = buf[p + 1];
    unsigned flags = get_be16(buf + p + 2);
    uint32_t stream_id = get_be32(buf + p + 4);
    uint32_t length = get_be32(buf + p + 8);

    switch (type) {
    case WUBU_YAMUX_DATA:
    case WUBU_YAMUX_WINDOW_UPDATE:
    case WUBU_YAMUX_PING:
    case WUBU_YAMUX_GOAWAY:
        break;
    default:
        return -1;
    }
    if (flags & ~WUBU_YAMUX_KNOWN_FLAGS) return -1;

    size_t body;
    int carries_payload = (type == WUBU_YAMUX_DATA);
    if (carries_payload) {
        if (length > n - p - WUBU_YAMUX_HEADER) return -1;
        body = (size_t)length;
    } else {
        body = 0;
    }

    *pos = p + WUBU_YAMUX_HEADER + body;
    h->type = type;
    h->flags = flags;
    h->stream_id = stream_id;
    h->length = length;
    *payload = carries_payload ? buf + p + WUBU_YAMUX_HEADER : NULL;
    *payload_len = body;
    return 0;
}
