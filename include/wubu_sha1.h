/*
 * wubu_sha1.h -- SHA-1 (FIPS 180-4), self-contained, for info-hashes.
 *
 * Why this exists: a BitTorrent v1 info-hash is the SHA-1 of the BENCODED INFO
 * DICT of a .torrent.  A magnet URI whose btih was invented addresses nothing
 * -- it looks like it works and resolves to nobody.  The magnet module
 * deliberately computes no hash for exactly that reason; this module is the
 * one that makes the address real.
 *
 * SHA-1 is broken for collision resistance and MUST NOT be used where a
 * collision matters (signatures, certs, integrity against an adversary).  An
 * info-hash is not such a use: it is an ADDRESS, its job is to be derived
 * identically by every peer, and the swarm verifies the bytes against the
 * piece hashes anyway.  Do not carry this primitive into anything security
 * critical without saying so out loud.
 *
 * Freestanding C11: no malloc, no hosted APIs.  Streaming API (init/update/
 * final) so a large info dict never has to be buffered whole.
 */
#ifndef WUBU_SHA1_H
#define WUBU_SHA1_H

#include <stddef.h>
#include <stdint.h>

#define WUBU_SHA1_SZ 20   /* digest bytes */

/* Opaque context: callers allocate WUBU_SHA1_CTX_SZ bytes (a byte array works
 * fine) and pass the pointer.  The size is part of the contract. */
#define WUBU_SHA1_CTX_SZ 96
typedef struct wubu_sha1_ctx wubu_sha1_ctx;

/* One-shot: digest of `len` bytes at `data`, written big-endian to `out`
 * (must hold >= WUBU_SHA1_SZ bytes).  Returns 0. */
int wubu_sha1(const void *data, size_t len, uint8_t out[WUBU_SHA1_SZ]);

void wubu_sha1_init(wubu_sha1_ctx *c);
void wubu_sha1_update(wubu_sha1_ctx *c, const void *data, size_t len);
void wubu_sha1_final(wubu_sha1_ctx *c, uint8_t out[WUBU_SHA1_SZ]);

/* The canonical hex form of a digest (lowercase, 40 chars + nul).  Returns 0. */
int wubu_sha1_hex(const uint8_t digest[WUBU_SHA1_SZ], char out[41]);

#endif /* WUBU_SHA1_H */
