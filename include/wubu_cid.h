/*
 * wubu_cid.h -- content identifiers (CIDs), the libp2p naming layer.
 *
 * A CID is an address that says WHAT something is, not WHERE it lives:
 *
 *   <version><codec><multihash>   -- wrapped in a multibase for text
 *
 *   bafkreibm6jg...   v1, raw, sha2-256   (base32)
 *   zb2rhZfjRh2...    v1, raw, sha2-256   (base58btc)
 *   QmYwAPJzv5CZ...   v0, dag-pb, sha2-256 (the 46-char legacy form)
 *
 * The codec names the ENCODING (raw bytes, dag-pb, dag-cbor); the multihash
 * names the HASH and carries the digest.  Splitting those two is what lets a
 * peer verify bytes without knowing how to interpret them.
 *
 * Fail-closed everywhere: a CID that does not decode, has an unsupported
 * version, a truncated digest, or a multibase prefix this module cannot read
 * is refused with -1 and leaves the caller's struct untouched.  A CID that
 * LOOKS like it works is worse than no CID, so nothing here guesses.
 *
 * C11, libc only.
 */
#ifndef WUBU_CID_H
#define WUBU_CID_H

#include <stddef.h>
#include <stdint.h>

#define WUBU_CID_DIGEST_MAX 64

/* multihash codes */
#define WUBU_CID_HASH_SHA2_256 0x12
#define WUBU_CID_HASH_SHA2_512 0x13
#define WUBU_CID_HASH_BLAKE3   0x1e

/* codecs */
#define WUBU_CID_CODEC_RAW      0x55
#define WUBU_CID_CODEC_DAG_PB   0x70
#define WUBU_CID_CODEC_DAG_CBOR 0x71

/* The legacy CIDv0 form: exactly these two bytes before the digest. */
#define WUBU_CID_V0_LEN       34     /* 0x12 0x20 + 32 bytes */
#define WUBU_CID_V0_TEXT_LEN  46     /* base58btc of the above */

typedef struct {
    uint64_t version;                /* 0 or 1 */
    uint64_t codec;                  /* v0 implies dag-pb */
    uint64_t hash_code;
    uint8_t  digest[WUBU_CID_DIGEST_MAX];
    size_t   digest_len;
} wubu_cid_t;

typedef enum {
    WUBU_CID_BASE32 = 1,             /* 'b', lowercase, no padding */
    WUBU_CID_BASE58,                 /* 'z' */
    WUBU_CID_BASE16                  /* 'f' */
} wubu_cid_base_t;

int wubu_cid_is_v0(const wubu_cid_t *c);
const char *wubu_cid_codec_name(uint64_t codec);
const char *wubu_cid_hash_name(uint64_t hash_code);

/* Parse a CID.  Returns 0 on success, -1 on any malformed input. */
int wubu_cid_parse(const char *s, wubu_cid_t *out);

/* Re-encode.  v0 always builds base58btc (it has no other legal form).
 * Returns bytes written, or -1 on a bad buffer or an impossible combination
 * (a v0 whose hash is not sha2-256/32). */
long wubu_cid_build(const wubu_cid_t *in, wubu_cid_base_t base,
                    char *out, size_t n);

/* ---- varints, public because bitswap reuses them ------------------ */

/* Unsigned LEB128.  Returns bytes written (1..10), or -1 when out is small. */
long wubu_varint_encode(uint64_t v, uint8_t *out, size_t n);

/* Decode one varint.  On success returns 0 and reports how many bytes it used
 * (so callers can chain).  -1 on a truncated or over-long encoding. */
int wubu_varint_decode(const uint8_t *in, size_t n, uint64_t *out,
                       size_t *used);

#endif /* WUBU_CID_H */
