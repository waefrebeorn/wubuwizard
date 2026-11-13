/*
 * wubu_multiaddr.c -- libp2p multiaddresses.
 *
 * C11, libc only.  Parsing is strict: the string is split on '/', each segment
 * must be a known protocol followed by a value of exactly the right shape,
 * and no empty segments are tolerated (so "a//b" and a trailing slash fail).
 */
#include "wubu_multiaddr.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

typedef struct { const char *name; wubu_ma_proto_t proto; } proto_entry;

static const proto_entry g_protos[] = {
    { "ip4",     WUBU_PROTO_IP4 },
    { "ip6",     WUBU_PROTO_IP6 },
    { "tcp",     WUBU_PROTO_TCP },
    { "udp",     WUBU_PROTO_UDP },
    { "dns",     WUBU_PROTO_DNS },
    { "dns4",    WUBU_PROTO_DNS4 },
    { "dns6",    WUBU_PROTO_DNS6 },
    { "p2p",     WUBU_PROTO_P2P },
    { "quic",    WUBU_PROTO_QUIC },
    { "quic-v1", WUBU_PROTO_QUIC_V1 },
    { "tls",     WUBU_PROTO_TLS },
    { "ws",      WUBU_PROTO_WS },
    { "wss",     WUBU_PROTO_WSS },
    { "http",    WUBU_PROTO_HTTP },
    { "https",   WUBU_PROTO_HTTPS },
    { "noise",   WUBU_PROTO_NOISE },
};

const char *wubu_ma_proto_name(wubu_ma_proto_t p)
{
    for (unsigned i = 0; i < sizeof(g_protos) / sizeof(g_protos[0]); i++)
        if (g_protos[i].proto == p) return g_protos[i].name;
    return NULL;
}

/* --- value parsers --------------------------------------------- */

/* dotted quad -> network order.  Refuses leading zeros and out-of-range
 * octets, because both make an address ambiguous. */
static int parse_ip4(const char *s, uint32_t *out)
{
    uint8_t b[4];
    size_t i = 0;
    while (i < 4) {
        if (!isdigit((unsigned char)*s)) return -1;
        if (s[0] == '0' && isdigit((unsigned char)s[1])) return -1;
        unsigned v = 0;
        int digits = 0;
        while (isdigit((unsigned char)*s) && digits < 3) {
            v = v * 10 + (unsigned)(*s - '0');
            s++;
            digits++;
        }
        if (v > 255) return -1;
        b[i++] = (uint8_t)v;
        if (i < 4) {
            if (*s != '.') return -1;
            s++;
        }
    }
    if (*s != '\0') return -1;
    *out = ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8)  | (uint32_t)b[3];
    return 0;
}

/* IPv6, including the "::" compression and the IPv4-mapped tail
 * ("::ffff:1.2.3.4").  Rules taken from RFC 4291 section 2.2:
 *
 *   - "::" stands for ONE OR MORE groups of zeros, so a compressed address may
 *     carry at most seven explicit groups.  "::1:2:3:4:5:6:7:8" has eight and
 *     is malformed -- accepting it would silently mean something other than
 *     what was written.
 *   - An IPv4 tail occupies exactly the last two groups, and an address may
 *     have at most 6 groups before it when it is present.
 *   - "::" alone is the unspecified address: syntactically valid.  It names no
 *     peer, but that is a ROUTING judgement, not a shape one, so the parser
 *     accepts it and leaves the decision to the caller. */
static int parse_ipv4_octets(const char **pp, uint8_t out[4])
{
    const char *p = *pp;
    for (int i = 0; i < 4; i++) {
        if (!isdigit((unsigned char)*p)) return -1;
        if (p[0] == '0' && isdigit((unsigned char)p[1])) return -1;
        unsigned v = 0;
        int digits = 0;
        while (isdigit((unsigned char)*p) && digits < 3) {
            v = v * 10 + (unsigned)(*p - '0');
            p++;
            digits++;
        }
        if (v > 255) return -1;
        out[i] = (uint8_t)v;
        if (i < 3) {
            if (*p != '.') return -1;
            p++;
        }
    }
    *pp = p;
    return 0;
}

static int parse_ip6(const char *s, uint8_t out[16])
{
    uint16_t g[8];
    int n = 0, compressed_at = -1;
    const char *p = s;

    if (*p == ':') {
        if (p[1] != ':') return -1;
        compressed_at = 0;
        p += 2;
        if (*p == '\0') {                    /* "::" -- the unspecified address */
            memset(out, 0, 16);
            return 0;
        }
    }

    while (*p) {
        if (n >= 8) return -1;

        /* An IPv4-mapped tail consumes the last two groups. */
        /* Only THIS segment counts: a dot further along belongs to a later
         * group, not to the tail. */
        size_t seg_len = strcspn(p, ":");
        if (isdigit((unsigned char)*p) &&
            memchr(p, '.', seg_len) != NULL) {
            uint8_t o[4];
            if (parse_ipv4_octets(&p, o) != 0) return -1;
            if (*p != '\0') return -1;
            if (compressed_at >= 0 ? (n + 2 >= 8) : (n + 2 != 8)) return -1;
            g[n++] = (uint16_t)((o[0] << 8) | o[1]);
            g[n++] = (uint16_t)((o[2] << 8) | o[3]);
            break;
        }

        unsigned v = 0;
        int digits = 0;
        while (isxdigit((unsigned char)*p) && digits < 4) {
            int c = *p;
            v = v * 16 + (unsigned)(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10);
            p++;
            digits++;
        }
        if (digits == 0) {
            if (*p == ':' && compressed_at < 0) {
                compressed_at = n;
                p++;
                if (*p == '\0') break;       /* a trailing "::" */
                continue;
            }
            return -1;
        }
        g[n++] = (uint16_t)v;

        if (*p == ':') {
            p++;
            if (*p == ':') {
                if (compressed_at >= 0) return -1;   /* a second "::" */
                compressed_at = n;
                p++;
                if (*p == '\0') break;               /* a trailing "::" */
            }
            continue;
        }
        break;
    }
    if (*p != '\0') return -1;
    if (compressed_at >= 0) {
        if (n >= 8) return -1;                       /* "::" needs room */
        int before = compressed_at;
        int after  = n - before;
        memset(out, 0, 16);
        for (int i = 0; i < before; i++) {
            out[i * 2]     = (uint8_t)(g[i] >> 8);
            out[i * 2 + 1] = (uint8_t)(g[i] & 0xFF);
        }
        for (int i = 0; i < after; i++) {
            int idx = before + (8 - n) + i;
            out[idx * 2]     = (uint8_t)(g[before + i] >> 8);
            out[idx * 2 + 1] = (uint8_t)(g[before + i] & 0xFF);
        }
        return 0;
    }
    if (n != 8) return -1;
    for (int i = 0; i < 8; i++) {
        out[i * 2]     = (uint8_t)(g[i] >> 8);
        out[i * 2 + 1] = (uint8_t)(g[i] & 0xFF);
    }
    return 0;
}

static int parse_port(const char *s, uint16_t *out)
{
    if (!isdigit((unsigned char)*s)) return -1;
    if (s[0] == '0' && isdigit((unsigned char)s[1])) return -1;
    unsigned long v = 0;
    int digits = 0;
    while (isdigit((unsigned char)*s) && digits < 5) {
        v = v * 10 + (unsigned long)(*s - '0');
        s++;
        digits++;
    }
    if (*s != '\0' || v > 65535) return -1;
    *out = (uint16_t)v;
    return 0;
}

/* A hostname-ish value: non-empty, no '/' (that would end the segment), and
 * within the buffer.  We do NOT validate the DNS grammar here -- this module
 * routes on shape, not on whether a name resolves. */
static int parse_name(const char *s, char *dst, size_t cap)
{
    if (!s || !*s || !dst || !cap) return -1;
    size_t n = strlen(s);
    if (n >= cap || strchr(s, '/') != NULL) return -1;
    memcpy(dst, s, n + 1);
    return 0;
}

/* A peer id: base58 multihash, arbitrary length within our buffer.  Any
 * non-empty segment of hex-ish or base58 characters is accepted verbatim; the
 * caller that needs the id's structure has its own parser. */
static int parse_peer(const char *s, uint8_t *dst, size_t cap, size_t *len)
{
    if (!s || !*s || !dst || !cap) return -1;
    size_t n = strlen(s);
    if (n >= cap || n >= WUBU_MA_PEER_MAX) return -1;
    memcpy(dst, s, n);
    *len = n;
    return 0;
}

/* --- public API ------------------------------------------------ */

int wubu_multiaddr_add(wubu_multiaddr_t *a, const wubu_ma_part_t *part)
{
    if (!a || !part || a->count >= WUBU_MA_MAX_PARTS) return -1;
    a->parts[a->count++] = *part;
    return 0;
}

int wubu_multiaddr_parse(const char *s, wubu_multiaddr_t *out)
{
    if (!s || !out || s[0] != '/') return -1;
    if (s[1] == '\0') return -1;             /* a bare "/" is not an address */
    size_t slen = strlen(s);
    if (s[slen - 1] == '/') return -1;        /* no trailing slash */

    wubu_multiaddr_t tmp;
    memset(&tmp, 0, sizeof(tmp));

    const char *p = s + 1;
    while (*p) {
        if (tmp.count >= WUBU_MA_MAX_PARTS) return -1;

        /* The protocol runs to the next '/'. */
        const char *seg = p;
        const char *slash = strchr(p, '/');
        size_t seglen = slash ? (size_t)(slash - p) : strlen(p);
        if (seglen == 0) return -1;          /* "//" or a trailing slash */

        char pname[32];
        if (seglen >= sizeof(pname)) return -1;
        memcpy(pname, seg, seglen);
        pname[seglen] = '\0';

        const proto_entry *pe = NULL;
        for (unsigned i = 0; i < sizeof(g_protos) / sizeof(g_protos[0]); i++)
            if (strcmp(g_protos[i].name, pname) == 0) { pe = &g_protos[i]; break; }
        if (!pe) return -1;                  /* unknown protocol */

        /* Value-bearing protocols consume the next segment; flag protocols do
         * not, so the walker stops after the name when the table says "none". */
        const char *value = slash ? slash + 1 : NULL;
        wubu_ma_part_t part;
        memset(&part, 0, sizeof(part));
        part.proto = pe->proto;


        switch (pe->proto) {
        case WUBU_PROTO_IP4:
        case WUBU_PROTO_IP6:
        case WUBU_PROTO_TCP:
        case WUBU_PROTO_UDP:
        case WUBU_PROTO_DNS:
        case WUBU_PROTO_DNS4:
        case WUBU_PROTO_DNS6:
        case WUBU_PROTO_P2P:
            if (!value || !*value) return -1;
            {
                const char *vend = strchr(value, '/');
                size_t vlen = vend ? (size_t)(vend - value) : strlen(value);
                char vbuf[WUBU_MA_NAME_MAX + 8];
                if (vlen >= sizeof(vbuf)) return -1;
                memcpy(vbuf, value, vlen);
                vbuf[vlen] = '\0';

                switch (pe->proto) {
                case WUBU_PROTO_IP4:
                    if (parse_ip4(vbuf, &part.ip4) != 0) return -1;
                    break;
                case WUBU_PROTO_IP6:
                    if (parse_ip6(vbuf, part.ip6) != 0) return -1;
                    break;
                case WUBU_PROTO_TCP:
                case WUBU_PROTO_UDP:
                    if (parse_port(vbuf, &part.port) != 0) return -1;
                    break;
                case WUBU_PROTO_P2P:
                    if (parse_peer(vbuf, part.peer, sizeof(part.peer),
                                   &part.peer_len) != 0) return -1;
                    break;
                default:
                    if (parse_name(vbuf, part.name, sizeof(part.name)) != 0)
                        return -1;
                    break;
                }
                if (!vend) { p = value + vlen; }
                else       { p = vend + 1; }
            }
            break;

        default:                             /* flags: quic, tls, ws, ... */
            if (!slash) { p = seg + seglen; }
            else        { p = slash + 1; }
            break;
        }

        tmp.parts[tmp.count++] = part;
    }

    if (tmp.count == 0) return -1;
    memcpy(out, &tmp, sizeof(tmp));
    return 0;
}

long wubu_multiaddr_build(const wubu_multiaddr_t *a, char *out, size_t n)
{
    if (!a || !out || a->count <= 0) return -1;

    size_t o = 0;
    for (int i = 0; i < a->count; i++) {
        const wubu_ma_part_t *p = &a->parts[i];
        if (o + 1 >= n) return -1;
        const char *pname = wubu_ma_proto_name(p->proto);
        if (!pname) return -1;
        out[o++] = '/';
        int w = snprintf(out + o, n - o, "%s", pname);
        if (w < 0 || (size_t)w >= n - o) return -1;
        o += (size_t)w;

        char val[64];
        switch (p->proto) {
        case WUBU_PROTO_IP4:
            snprintf(val, sizeof(val), "%u.%u.%u.%u",
                     (unsigned)(p->ip4 >> 24) & 0xFF,
                     (unsigned)(p->ip4 >> 16) & 0xFF,
                     (unsigned)(p->ip4 >> 8) & 0xFF,
                     (unsigned)p->ip4 & 0xFF);
            break;
        case WUBU_PROTO_IP6: {
            /* Emit in full form when any low group is zero; otherwise use
             * "::" for the longest run of zero groups. */
            unsigned g[8];
            for (int k = 0; k < 8; k++)
                g[k] = ((unsigned)p->ip6[k * 2] << 8) | p->ip6[k * 2 + 1];
            int best_s = -1, best_len = 0;
            for (int k = 0; k < 8; ) {
                if (g[k] == 0) {
                    int s = k, len = 0;
                    while (k < 8 && g[k] == 0) { k++; len++; }
                    if (len > best_len) { best_len = len; best_s = s; }
                } else k++;
            }
            size_t vi = 0;
            for (int k = 0; k < 8;) {
                if (best_len > 1 && k == best_s) {
                    val[vi++] = (k == 0) ? ':' : ':';
                    if (best_s + best_len >= 8) val[vi++] = ':';
                    k += best_len;
                    continue;
                }
                vi += (size_t)snprintf(val + vi, sizeof(val) - vi,
                                       (k == 0) ? "%x" : ":%x", g[k]);
                k++;
            }
            val[vi] = '\0';
            break;
        }
        case WUBU_PROTO_TCP:
        case WUBU_PROTO_UDP:
            snprintf(val, sizeof(val), "%u", (unsigned)p->port);
            break;
        case WUBU_PROTO_DNS:
        case WUBU_PROTO_DNS4:
        case WUBU_PROTO_DNS6:
            snprintf(val, sizeof(val), "%s", p->name);
            break;
        case WUBU_PROTO_P2P:
            memcpy(val, p->peer, p->peer_len);
            val[p->peer_len < sizeof(val) - 1 ? p->peer_len : sizeof(val) - 1]
                = '\0';
            break;
        default:
            val[0] = '\0';                   /* a flag carries no value */
            break;
        }

        if (val[0]) {
            w = snprintf(out + o, n - o, "/%s", val);
            if (w < 0 || (size_t)w >= n - o) return -1;
            o += (size_t)w;
        }
    }
    out[o] = '\0';
    return (long)o;
}

int wubu_multiaddr_has(const wubu_multiaddr_t *a, wubu_ma_proto_t p)
{
    if (!a) return 0;
    for (int i = 0; i < a->count; i++)
        if (a->parts[i].proto == p) return 1;
    return 0;
}
