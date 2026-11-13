/*
 * wubu_cid.c -- content identifiers (CIDs).
 *
 * C11, libc only.  Three encodings: unsigned LEB128 varints, base32-lower
 * without padding, and base58btc.  Base16 is trivial enough to inline.
 */
#include "wubu_cid.h"

#include <string.h>
#include <stdio.h>

/* ---- varint ---------------------------------------------------- */

/* Multiformats unsigned varints have a practical maximum of NINE bytes (63
 * bits): "Implementations MUST restrict the size of the varint to a max of 9
 * bytes".  Values at or above 2^63 therefore cannot be represented here and
 * are refused rather than silently truncated into ten bytes. */
#define WUBU_VARINT_MAX_BYTES 9

long wubu_varint_encode(uint64_t v, uint8_t *out, size_t n)
{
    if (!out || v >= ((uint64_t)1 << 63)) return -1;

    uint8_t tmp[WUBU_VARINT_MAX_BYTES];
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

/* Decoding tolerates up to TEN bytes because protobuf sign-extends negative
 * int32/int64 to a 64-bit varint, and bitswap carries negative priorities.
 * Callers that need the stricter multiformats limit bound the VALUE instead
 * (a CID or a frame length can never legitimately reach 2^63).
 *
 * Non-minimal encodings are refused: {0x81, 0x00} means 1 but wastes a byte,
 * and accepting it lets one number have two spellings.  The spec requires the
 * minimal form ("Leading zeros must be trimmed when encoding and must be
 * rejected when decoding"). */
int wubu_varint_decode(const uint8_t *in, size_t n, uint64_t *out,
                       size_t *used)
{
    if (!in || !out || !used) return -1;
    uint64_t v = 0;
    int shift = 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t b = in[i];
        if (shift >= 64) return -1;          /* would overflow */
        if (i > 0 && b == 0x00) return -1;   /* not minimally encoded */
        v |= ((uint64_t)(b & 0x7F)) << shift;
        if (!(b & 0x80)) {
            *out = v;
            *used = i + 1;
            return 0;
        }
        shift += 7;
    }
    return -1;                               /* ran out mid-varint */
}

/* ---- base32 lower, no padding (RFC 4648) ----------------------- */

static const char kB32[] = "abcdefghijklmnopqrstuvwxyz234567";

static int b32val(int c)
{
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= '2' && c <= '7') return c - '2' + 26;
    return -1;
}

static long base32_encode(const uint8_t *in, size_t len, char *out)
{
    size_t o = 0;
    for (size_t i = 0; i < len; i += 5) {
        uint64_t chunk = 0;
        int have = (int)((len - i) < 5 ? (len - i) : 5);
        for (int k = 0; k < 5; k++)
            chunk = (chunk << 8) | (k < have ? in[i + k] : 0);
        int bits = have * 8;
        for (int k = 0; k < 8; k++) {
            if (k * 5 >= bits) break;        /* no padding characters */
            int idx = (int)((chunk >> (35 - k * 5)) & 0x1F);
            out[o++] = kB32[idx];
        }
    }
    out[o] = '\0';
    return (long)o;
}

static int base32_decode(const char *s, uint8_t *out, size_t cap)
{
    size_t o = 0;
    uint64_t acc = 0;
    int bits = 0;
    for (; *s; s++) {
        int v = b32val((unsigned char)*s);
        if (v < 0) return -1;                /* '=' or anything else */
        acc = (acc << 5) | (uint64_t)v;
        bits += 5;
        while (bits >= 8) {
            if (o >= cap) return -1;
            bits -= 8;
            out[o++] = (uint8_t)((acc >> bits) & 0xFF);
        }
    }
    /* Trailing bits must be zero, or the encoding was not canonical. */
    if (bits > 0 && (acc & ((1u << bits) - 1)) != 0) return -1;
    return (int)o;
}

/* ---- base58btc ----------------------------------------------- */

static const char kB58[] =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

static long base58_encode(const uint8_t *in, size_t len, char *out)
{
    /* Count leading zeros: each becomes a leading '1'. */
    size_t zeros = 0;
    while (zeros < len && in[zeros] == 0) zeros++;

    /* Worst case log(256)/log(58) ~= 1.37 bytes per input byte. */
    char tmp[256];
    size_t o = 0;
    for (size_t i = zeros; i < len; i++) {
        unsigned carry = in[i];
        for (size_t k = 0; k < o; k++) {
            unsigned v = (unsigned)((tmp[k] - '0') * 256 + carry);
            tmp[k] = (char)('0' + (v % 58));
            carry = v / 58;
        }
        while (carry) {
            if (o >= sizeof(tmp)) return -1;
            tmp[o++] = (char)('0' + (carry % 58));
            carry /= 58;
        }
    }
    size_t w = 0;
    for (size_t i = 0; i < zeros; i++) out[w++] = '1';
    /* tmp[] holds DIGIT VALUES ('0' + 0..57), least-significant first. */
    for (size_t i = 0; i < o; i++)
        out[w++] = kB58[(unsigned char)(tmp[o - 1 - i] - '0')];
    out[w] = '\0';
    return (long)w;
}

static int base58_value(int c)
{
    if (c < 0) return -1;
    for (int i = 0; i < 58; i++)
        if ((unsigned char)kB58[i] == c) return i;
    return -1;
}

static int base58_decode(const char *s, uint8_t *out, size_t cap)
{
    size_t o = 0;
    for (const char *p = s; *p; p++) {
        int v = base58_value((unsigned char)*p);
        if (v < 0) return -1;
        unsigned carry = (unsigned)v;
        for (size_t k = 0; k < o; k++) {
            unsigned x = (unsigned)out[k] * 58 + carry;
            out[k] = (uint8_t)(x & 0xFF);
            carry = x >> 8;
        }
        while (carry) {
            if (o >= cap) return -1;
            out[o++] = (uint8_t)(carry & 0xFF);
            carry >>= 8;
        }
    }
    /* The digits accumulated most-significant-first; reverse them into place
     * and re-prefix the leading '1's as zero bytes. */
    size_t ones = 0;
    while (s[ones] == '1') ones++;
    /* out currently holds the payload REVERSED and without the leading zeros. */
    for (size_t i = 0; i < o / 2; i++) {
        uint8_t t = out[i];
        out[i] = out[o - 1 - i];
        out[o - 1 - i] = t;
    }
    if (ones) {
        if (o + ones > cap) return -1;
        memmove(out + ones, out, o);
        memset(out, 0, ones);
        o += ones;
    }
    return (int)o;
}

/* ---- names ----------------------------------------------------- */

int wubu_cid_is_v0(const wubu_cid_t *c)
{
    return c && c->version == 0;
}

const char *wubu_cid_codec_name(uint64_t codec)
{
    switch (codec) {
    case WUBU_CID_CODEC_RAW:      return "raw";
    case WUBU_CID_CODEC_DAG_PB:   return "dag-pb";
    case WUBU_CID_CODEC_DAG_CBOR: return "dag-cbor";
    default: return NULL;
    }
}

const char *wubu_cid_hash_name(uint64_t hash_code)
{
    switch (hash_code) {
    case WUBU_CID_HASH_SHA2_256: return "sha2-256";
    case WUBU_CID_HASH_SHA2_512: return "sha2-512";
    case WUBU_CID_HASH_BLAKE3:   return "blake3";
    default: return NULL;
    }
}

/* ---- parse ----------------------------------------------------- */

/* Parse the CID body after any multibase prefix has been stripped. */
static int parse_body(const uint8_t *buf, size_t len, wubu_cid_t *out)
{
    if (len == 0) return -1;

    /* The legacy binary form: 0x12 0x20 <32 bytes>. */
    if (len == WUBU_CID_V0_LEN && buf[0] == 0x12 && buf[1] == 0x20) {
        out->version = 0;
        out->codec = WUBU_CID_CODEC_DAG_PB;         /* implied by v0 */
        out->hash_code = WUBU_CID_HASH_SHA2_256;
        out->digest_len = 32;
        memcpy(out->digest, buf + 2, 32);
        return 0;
    }

    size_t used = 0;
    uint64_t version = 0;
    if (wubu_varint_decode(buf, len, &version, &used) != 0) return -1;
    if (version != 1) return -1;                     /* only v0 and v1 exist */
    size_t p = used;

    uint64_t codec = 0;
    if (p >= len || wubu_varint_decode(buf + p, len - p, &codec, &used) != 0)
        return -1;
    p += used;

    uint64_t hash_code = 0;
    if (p >= len || wubu_varint_decode(buf + p, len - p, &hash_code, &used) != 0)
        return -1;
    p += used;

    uint64_t digest_len = 0;
    if (p >= len || wubu_varint_decode(buf + p, len - p, &digest_len, &used) != 0)
        return -1;
    p += used;

    if (digest_len > WUBU_CID_DIGEST_MAX) return -1;
    if (len - p != digest_len) return -1;            /* exact length required */

    out->version = 1;
    out->codec = codec;
    out->hash_code = hash_code;
    out->digest_len = (size_t)digest_len;
    memcpy(out->digest, buf + p, (size_t)digest_len);
    return 0;
}

int wubu_cid_parse(const char *s, wubu_cid_t *out)
{
    if (!s || !out) return -1;

    /* The 46-character legacy form needs no multibase prefix. */
    if (strlen(s) == WUBU_CID_V0_TEXT_LEN && strncmp(s, "Qm", 2) == 0) {
        uint8_t body[WUBU_CID_V0_LEN + 8];
        int n = base58_decode(s, body, sizeof(body));
        if (n != WUBU_CID_V0_LEN) return -1;
        return parse_body(body, (size_t)n, out);
    }

    /* Anything else must name its encoding with a multibase prefix. */
    char prefix = s[0];
    const char *rest = s + 1;
    uint8_t body[512];
    int n = -1;

    if (prefix == 'b' || prefix == 'B') {
        n = base32_decode(rest, body, sizeof(body));
    } else if (prefix == 'z') {
        n = base58_decode(rest, body, sizeof(body));
    } else if (prefix == 'f' || prefix == 'F') {
        size_t rl = strlen(rest);
        if (rl % 2) return -1;
        n = 0;
        for (size_t i = 0; i < rl; i += 2) {
            int hi = (rest[i] >= 'a') ? rest[i] - 'a' + 10 : rest[i] - '0';
            int lo = (rest[i + 1] >= 'a') ? rest[i + 1] - 'a' + 10
                                          : rest[i + 1] - '0';
            if ((unsigned)hi > 15 || (unsigned)lo > 15) return -1;
            if ((size_t)n >= sizeof(body)) return -1;
            body[n++] = (uint8_t)((hi << 4) | lo);
        }
    } else {
        return -1;
    }
    if (n <= 0) return -1;
    return parse_body(body, (size_t)n, out);
}

/* ---- build ----------------------------------------------------- */

static int append_multihash(const wubu_cid_t *c, uint8_t *body, size_t *p,
                            size_t cap)
{
    uint8_t vb[10];
    size_t need = c->digest_len;
    long n;
    n = wubu_varint_encode(c->hash_code, vb, sizeof(vb));
    if (n < 0 || *p + (size_t)n > cap) return -1;
    memcpy(body + *p, vb, (size_t)n);
    *p += (size_t)n;
    n = wubu_varint_encode((uint64_t)need, vb, sizeof(vb));
    if (n < 0 || *p + (size_t)n > cap) return -1;
    memcpy(body + *p, vb, (size_t)n);
    *p += (size_t)n;
    if (*p + need > cap) return -1;
    memcpy(body + *p, c->digest, need);
    *p += need;
    return 0;
}

long wubu_cid_build(const wubu_cid_t *c, wubu_cid_base_t base,
                    char *out, size_t n)
{
    if (!c || !out) return -1;
    /* An unknown base must not silently become base16. */
    if (base != WUBU_CID_BASE32 && base != WUBU_CID_BASE58 &&
        base != WUBU_CID_BASE16)
        return -1;

    /* v0 has exactly one legal shape: dag-pb, sha2-256, 32 bytes, base58. */
    if (c->version == 0) {
        if (c->hash_code != WUBU_CID_HASH_SHA2_256 ||
            c->digest_len != 32 ||
            c->codec != WUBU_CID_CODEC_DAG_PB)
            return -1;
        if (base != WUBU_CID_BASE58) return -1;      /* v0 is base58 or nothing */
        uint8_t body[WUBU_CID_V0_LEN];
        body[0] = 0x12;
        body[1] = 0x20;
        memcpy(body + 2, c->digest, 32);
        char txt[80];
        if (base58_encode(body, sizeof(body), txt) != WUBU_CID_V0_TEXT_LEN)
            return -1;
        if ((size_t)WUBU_CID_V0_TEXT_LEN + 1 > n) return -1;
        memcpy(out, txt, WUBU_CID_V0_TEXT_LEN + 1);
        return WUBU_CID_V0_TEXT_LEN;
    }
    if (c->version != 1) return -1;

    uint8_t body[512];
    size_t p = 0;
    uint8_t vb[10];

    long vn = wubu_varint_encode(1, vb, sizeof(vb));
    if (vn < 0) return -1;
    memcpy(body, vb, (size_t)vn);
    p += (size_t)vn;

    long cn = wubu_varint_encode(c->codec, vb, sizeof(vb));
    if (cn < 0) return -1;
    memcpy(body + p, vb, (size_t)cn);
    p += (size_t)cn;

    if (append_multihash(c, body, &p, sizeof(body)) != 0) return -1;

    char txt[600];
    size_t tl = 0;
    const char pre = (base == WUBU_CID_BASE32) ? 'b'
                   : (base == WUBU_CID_BASE58) ? 'z' : 'f';

    if (base == WUBU_CID_BASE32) {
        long k = base32_encode(body, p, txt + 1);
        if (k < 0) return -1;
        txt[0] = pre;
        tl = (size_t)k + 1;
    } else if (base == WUBU_CID_BASE58) {
        long k = base58_encode(body, p, txt + 1);
        if (k < 0) return -1;
        txt[0] = pre;
        tl = (size_t)k + 1;
    } else {
        static const char hx[] = "0123456789abcdef";
        size_t o = 1;
        txt[0] = pre;
        for (size_t i = 0; i < p; i++) {
            txt[o++] = hx[body[i] >> 4];
            txt[o++] = hx[body[i] & 15];
        }
        txt[o] = '\0';
        tl = o;
    }

    if (tl + 1 > n) return -1;
    memcpy(out, txt, tl + 1);
    return (long)tl;
}
