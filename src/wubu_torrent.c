/*
 * wubu_torrent.c -- read a .torrent well enough to name it.
 *
 * C11, libc only.  The parser is structural and single-pass: it never copies
 * the dictionary, it only walks it far enough to bound the `info` value.
 */
#include "wubu_torrent.h"
#include "wubu_sha1.h"

#include <string.h>

typedef struct {
    const uint8_t *buf;
    size_t         len;
    size_t         pos;
} rd_t;

/* A bencode byte-string: <decimal-len>:<bytes>.  Leading zeros are invalid
 * (except "0" itself) because two encodings of the same length would make the
 * format non-canonical.  Returns 0, or -1 on a malformed length. */
static int skip_string(rd_t *r)
{
    if (r->pos >= r->len) return -1;
    if (r->buf[r->pos] < '0' || r->buf[r->pos] > '9') return -1;

    size_t n = 0;
    int digits = 0;
    while (r->pos < r->len && r->buf[r->pos] >= '0' && r->buf[r->pos] <= '9') {
        n = n * 10 + (size_t)(r->buf[r->pos] - '0');
        if (n > r->len) return -1;          /* cannot possibly fit */
        r->pos++;
        digits++;
    }
    if (digits == 0) return -1;
    if (digits > 1 && r->buf[r->pos - digits] == '0') return -1;  /* leading 0 */
    if (r->pos >= r->len || r->buf[r->pos] != ':') return -1;
    r->pos++;
    if (n > r->len - r->pos) return -1;
    r->pos += n;
    return 0;
}

/* The bytes of a key at `at`: returns the start and length, or -1. */
static int key_at(const uint8_t *buf, size_t len, size_t at,
                  size_t *kstart, size_t *klen)
{
    if (at >= len || buf[at] < '0' || buf[at] > '9') return -1;
    size_t n = 0, p = at;
    while (p < len && buf[p] >= '0' && buf[p] <= '9') {
        n = n * 10 + (size_t)(buf[p] - '0');
        p++;
    }
    if (p >= len || buf[p] != ':') return -1;
    if (n > len - (p + 1)) return -1;
    *kstart = p + 1;
    *klen = n;
    return 0;
}

static int skip_int(rd_t *r)
{
    if (r->pos >= r->len || r->buf[r->pos] != 'i') return -1;
    r->pos++;
    if (r->pos < r->len && r->buf[r->pos] == '-') r->pos++;
    size_t digits_start = r->pos;
    while (r->pos < r->len && r->buf[r->pos] >= '0' && r->buf[r->pos] <= '9')
        r->pos++;
    if (r->pos == digits_start) return -1;
    if (r->pos >= r->len || r->buf[r->pos] != 'e') return -1;
    r->pos++;
    return 0;
}

static int skip_value(rd_t *r);

static int skip_list(rd_t *r)
{
    if (r->pos >= r->len || r->buf[r->pos] != 'l') return -1;
    r->pos++;
    while (r->pos < r->len && r->buf[r->pos] != 'e')
        if (skip_value(r) != 0) return -1;
    if (r->pos >= r->len) return -1;
    r->pos++;                                   /* the closing 'e' */
    return 0;
}

static int skip_dict(rd_t *r)
{
    if (r->pos >= r->len || r->buf[r->pos] != 'd') return -1;
    r->pos++;
    while (r->pos < r->len && r->buf[r->pos] != 'e') {
        if (skip_string(r) != 0) return -1;     /* the key */
        if (skip_value(r) != 0) return -1;      /* the value */
    }
    if (r->pos >= r->len) return -1;
    r->pos++;
    return 0;
}

static int skip_value(rd_t *r)
{
    if (r->pos >= r->len) return -1;
    char c = (char)r->buf[r->pos];
    if (c == 'i') return skip_int(r);
    if (c == 'l') return skip_list(r);
    if (c == 'd') return skip_dict(r);
    return skip_string(r);                      /* anything else is a string */
}

int wubu_bencode_validate(const uint8_t *buf, size_t len)
{
    if (!buf || !len) return -1;
    rd_t r = { buf, len, 0 };
    if (skip_value(&r) != 0) return -1;
    /* Trailing bytes after one complete value mean this was not one value. */
    return (r.pos == len) ? 0 : -1;
}

int wubu_torrent_info_span(const uint8_t *buf, size_t len,
                           size_t *start, size_t *end)
{
    if (!buf || !start || !end) return -1;
    rd_t r = { buf, len, 0 };

    /* The top level must be a dictionary. */
    if (len == 0 || buf[0] != 'd') return -1;
    r.pos = 1;

    while (r.pos < len && buf[r.pos] != 'e') {
        size_t kstart, klen;
        size_t key_at_pos = r.pos;
        if (key_at(buf, len, key_at_pos, &kstart, &klen) != 0) return -1;

        /* Skip the key properly so `pos` advances past it. */
        if (skip_string(&r) != 0) return -1;

        size_t vstart = r.pos;
        if (klen == 4 && memcmp(buf + kstart, "info", 4) == 0) {
            if (skip_value(&r) != 0) return -1;
            /* An empty info dict ("de") addresses nothing usable -- no name,
             * no pieces, nothing for a peer to fetch.  Refuse it rather than
             * hand back a btih for a torrent that cannot exist. */
            if (r.pos - vstart <= 2) return -1;
            *start = vstart;
            *end = r.pos;
            return 0;
        }
        if (skip_value(&r) != 0) return -1;
    }
    return -1;                                  /* no `info` key */
}

int wubu_torrent_info_hash(const uint8_t *buf, size_t len, char out[41])
{
    if (!out) return -1;
    size_t start = 0, end = 0;
    if (wubu_torrent_info_span(buf, len, &start, &end) != 0) return -1;

    uint8_t digest[WUBU_SHA1_SZ];
    if (wubu_sha1(buf + start, end - start, digest) != 0) return -1;
    return wubu_sha1_hex(digest, out);
}
