/*
 * wubu_magnet.h -- BitTorrent magnet URIs (BEP 9) for model addressing.
 *
 * Why: an open-weights model needs an address that survives the place it was
 * published.  A magnet URI names a model by its info-hash and carries the
 * trackers, so the bytes can come from the swarm rather than from one host --
 * the "Pirate Face / swarm permanence" leg of the distributed model plan.
 *
 * This module is the SHAPE only: it builds and parses the URI, and validates
 * the info-hash.  It does not fetch anything, and it deliberately does not
 * invent a hash -- the btih must be a real SHA-1 of a real bencoded info dict
 * (see docs/DISTRIBUTED_MODELS.md).  A magnet that names nothing is worse
 * than no magnet, so parse() refuses a URI without an xt.
 *
 * C11, libc only.
 */
#ifndef WUBU_MAGNET_H
#define WUBU_MAGNET_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define WUBU_MAGNET_BTIH_LEN   40   /* SHA-1 hex, no nul */
#define WUBU_MAGNET_NAME_MAX   256
#define WUBU_MAGNET_TRACKER_MAX 256
#define WUBU_MAGNET_TRACKERS   8

typedef struct {
    char     btih[WUBU_MAGNET_BTIH_LEN + 1];   /* lowercase hex, nul-ended */
    char     name[WUBU_MAGNET_NAME_MAX];
    char     trackers[WUBU_MAGNET_TRACKERS][WUBU_MAGNET_TRACKER_MAX];
    int      tracker_count;
    uint64_t length;                            /* xl; 0 = not stated */
} wubu_magnet_t;

/* Is `btih` a well-formed v1 info-hash: exactly 40 hex chars? */
bool wubu_magnet_valid_btih(const char *btih);

/* Build a magnet URI.  `name` may be NULL/empty (omitted).  Trackers beyond
 * WUBU_MAGNET_TRACKERS are ignored.  Returns bytes written, or -1 when the
 * btih is malformed or the buffer is too small. */
long wubu_magnet_build(const wubu_magnet_t *m, char *out, size_t n);

/* Parse a magnet URI.  On success fills `out` and returns 0.  EVERY failure --
 * not a magnet, no xt, a bad btih, a bad percent escape, an over-long part --
 * returns -1 and leaves `out` untouched, so a caller cannot learn WHY. */
int wubu_magnet_parse(const char *uri, wubu_magnet_t *out);

/* The xt exact-topic prefixes we understand. */
#define WUBU_MAGNET_XT_BTIH "urn:btih:"
#define WUBU_MAGNET_XT_BTMH "urn:btmh:"   /* v2 multihash; parsed, not hashed */

#endif /* WUBU_MAGNET_H */
