/*
 * wubu_multiaddr.h -- libp2p multiaddresses (the addressing grammar).
 *
 * A multiaddr is a self-describing path to a peer:
 *
 *   /ip4/1.2.3.4/tcp/6881/p2p/<peer-id>
 *   /ip6/::1/udp/53/quic-v1/webtransport
 *
 * It exists so a peer can be named WITHOUT knowing how to reach it yet: the
 * address carries every protocol hop in order, and a caller picks whichever
 * one it speaks.  That is the same property the magnet gives a model -- an
 * address that survives the host -- which is why this leg is here.
 *
 * This module PARSES and BUILDS.  It does not dial anything: producing bytes
 * and validating structure is its whole job, and the socket belongs to the
 * caller, exactly as in the other legs.
 *
 * Fail-closed: a string must start with '/', every segment must name a
 * protocol this module knows, every value must match that protocol's shape,
 * and a trailing empty segment (a bare slash) is refused rather than ignored.
 * Every refusal is -1 and leaves the output untouched.
 *
 * C11, libc only.
 */
#ifndef WUBU_MULTIADDR_H
#define WUBU_MULTIADDR_H

#include <stddef.h>
#include <stdint.h>

#define WUBU_MA_MAX_PARTS  16
#define WUBU_MA_NAME_MAX   256
#define WUBU_MA_PEER_MAX   64

/* The protocols this module understands.  Unknown ones are refused rather
 * than stored: a part we cannot validate is a part we cannot route on. */
typedef enum {
    WUBU_PROTO_IP4 = 1,
    WUBU_PROTO_IP6,
    WUBU_PROTO_TCP,
    WUBU_PROTO_UDP,
    WUBU_PROTO_DNS,
    WUBU_PROTO_DNS4,
    WUBU_PROTO_DNS6,
    WUBU_PROTO_P2P,
    WUBU_PROTO_QUIC,
    WUBU_PROTO_QUIC_V1,
    WUBU_PROTO_TLS,
    WUBU_PROTO_WS,
    WUBU_PROTO_WSS,
    WUBU_PROTO_HTTP,
    WUBU_PROTO_HTTPS,
    WUBU_PROTO_NOISE
} wubu_ma_proto_t;

typedef struct {
    wubu_ma_proto_t proto;
    uint32_t        ip4;               /* network byte order */
    uint8_t         ip6[16];
    uint16_t        port;
    char            name[WUBU_MA_NAME_MAX];
    uint8_t         peer[WUBU_MA_PEER_MAX];
    size_t          peer_len;
} wubu_ma_part_t;

typedef struct {
    wubu_ma_part_t  parts[WUBU_MA_MAX_PARTS];
    int             count;
} wubu_multiaddr_t;

const char *wubu_ma_proto_name(wubu_ma_proto_t p);

/* Parse a multiaddr.  Returns 0 on success, -1 on any malformed input. */
int wubu_multiaddr_parse(const char *s, wubu_multiaddr_t *out);

/* Re-serialise.  Returns bytes written, or -1 on a bad buffer or an empty
 * address. */
long wubu_multiaddr_build(const wubu_multiaddr_t *a, char *out, size_t n);

/* Append a part (used by callers assembling an address programmatically).
 * Returns 0, or -1 when full. */
int wubu_multiaddr_add(wubu_multiaddr_t *a, const wubu_ma_part_t *part);

/* True when the address contains a part of this protocol. */
int wubu_multiaddr_has(const wubu_multiaddr_t *a, wubu_ma_proto_t p);

#endif /* WUBU_MULTIADDR_H */
