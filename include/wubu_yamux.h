/*
 * wubu_yamux.h -- yamux stream framing.
 *
 * One TCP connection carries many logical streams.  Each frame is a fixed
 * 12-byte header followed by an optional payload:
 *
 *     version    1 byte    always 0
 *     type       1 byte    DATA / WINDOW_UPDATE / PING / GOAWAY
 *     flags      2 bytes   BE: SYN, ACK, FIN, RST
 *     streamID   4 bytes   BE: odd = client-initiated, even = server
 *     length     4 bytes   BE: payload length, or the window delta
 *
 * This module frames and parses headers.  It does not open sockets, track
 * windows, or reassemble streams -- framing is its whole job, and the wire
 * belongs to the caller.
 *
 * Fail-closed: a version other than 0, an unknown type, undefined flag bits, a
 * payload longer than the buffer, or a truncated header is refused with -1 and
 * leaves the caller's output untouched.
 *
 * C11, libc only.
 */
#ifndef WUBU_YAMUX_H
#define WUBU_YAMUX_H

#include <stddef.h>
#include <stdint.h>

#define WUBU_YAMUX_VERSION   0
#define WUBU_YAMUX_HEADER    12
#define WUBU_YAMUX_MAX_LEN   (1 << 20)

/* Type numbers are the ones from the libp2p yamux specification:
 *   0x0 Data, 0x1 Window Update, 0x2 Ping, 0x3 Go Away.
 * They are written explicitly rather than left implicit -- an enum that starts
 * at 1 would put a Window Update on the wire where a peer reads Data. */
typedef enum {
    WUBU_YAMUX_DATA = 0,
    WUBU_YAMUX_WINDOW_UPDATE = 1,
    WUBU_YAMUX_PING = 2,
    WUBU_YAMUX_GOAWAY = 3
} wubu_yamux_type_t;

enum {
    WUBU_YAMUX_SYN = 0x01,
    WUBU_YAMUX_ACK = 0x02,
    WUBU_YAMUX_FIN = 0x04,
    WUBU_YAMUX_RST = 0x08
};
#define WUBU_YAMUX_KNOWN_FLAGS 0x0F

typedef struct {
    int      type;          /* wubu_yamux_type_t */
    unsigned flags;         /* SYN | ACK | FIN | RST */
    uint32_t stream_id;
    uint32_t length;        /* payload bytes, or the window delta */
} wubu_yamux_hdr_t;

/* Encode a frame (header + payload).  Returns bytes written, or -1. */
long wubu_yamux_encode(const wubu_yamux_hdr_t *h, const uint8_t *payload,
                       size_t payload_len, uint8_t *out, size_t n);

/* Parse ONE frame at buf[*pos..n).  Advances *pos past it.
 * Returns 0, or -1 on a malformed frame. */
int wubu_yamux_decode(const uint8_t *buf, size_t n, size_t *pos,
                      wubu_yamux_hdr_t *h, const uint8_t **payload,
                      size_t *payload_len);

/* True when the stream ID was initiated by this side's peer-role convention:
 * odd IDs are opened by the client, even by the server. */
int wubu_yamux_is_client(uint32_t stream_id);
int wubu_yamux_is_server(uint32_t stream_id);

#endif /* WUBU_YAMUX_H */
