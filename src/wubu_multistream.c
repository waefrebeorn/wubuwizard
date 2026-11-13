/*
 * wubu_multistream.c -- multistream-select framing and negotiation.
 *
 * C11, libc only.  Frames are varint-length-prefixed; the length counts the
 * bytes on the wire, which for strings includes the trailing newline.
 */
#include "wubu_multistream.h"

#include <string.h>
#include <stdio.h>

#include "wubu_cid.h"

/* ---- encode ----------------------------------------------------- */

long wubu_ms_encode_header(uint8_t *out, size_t n)
{
    return wubu_ms_encode_data((const uint8_t *)WUBU_MS_HEADER,
                               WUBU_MS_HEADER_LEN, out, n);
}

long wubu_ms_encode_na(uint8_t *out, size_t n)
{
    return wubu_ms_encode_data((const uint8_t *)WUBU_MS_NA, WUBU_MS_NA_LEN,
                               out, n);
}

static int proto_valid(const char *p, size_t len)
{
    if (!p || len == 0) return 0;
    if (p[0] != '/') return 0;
    /* A protocol name is slash-separated non-empty segments.  Two slashes in a
     * row, a trailing slash without a newline, or an embedded NUL all make it
     * something we cannot address reliably. */
    int prev_slash = 0;   /* the required leading "/" is not a repeat */
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)p[i];
        if (c == '\0') return 0;
        if (c == '\n') {
            if (i + 1 != len) return 0;      /* newline must end it */
            if (prev_slash) return 0;        /* "/" alone is not a protocol */
            return 1;
        }
        if (c < 0x20 || c > 0x7E) return 0;  /* printable ASCII only */
        if (c == '/') { if (prev_slash) return 0; prev_slash = 1; }
        else prev_slash = 0;
    }
    return 0;                                /* no terminating newline */
}

long wubu_ms_encode(const char *proto, uint8_t *out, size_t n)
{
    if (!proto || !out) return -1;

    size_t len = strlen(proto);
    if (len + 1 >= WUBU_MS_MAX_PROTO) return -1;

    /* Accept a bare name and add the newline the wire form requires. */
    char with_nl[WUBU_MS_MAX_PROTO];
    if (len && proto[len - 1] == '\n') {
        memcpy(with_nl, proto, len);
        with_nl[len] = '\0';
    } else {
        memcpy(with_nl, proto, len);
        with_nl[len] = '\n';
        with_nl[len + 1] = '\0';
        len++;
    }
    if (!proto_valid(with_nl, len)) return -1;
    return wubu_ms_encode_data((const uint8_t *)with_nl, len, out, n);
}

long wubu_ms_encode_data(const uint8_t *data, size_t len, uint8_t *out,
                         size_t n)
{
    if (!out) return -1;
    if (len > WUBU_MS_MAX_FRAME) return -1;
    if (data == NULL && len != 0) return -1;

    uint8_t vb[10];
    long vn = wubu_varint_encode(len, vb, sizeof(vb));
    if (vn < 0) return -1;
    if ((size_t)vn + len > n) return -1;
    memcpy(out, vb, (size_t)vn);
    if (len) memcpy(out + vn, data, len);
    return vn + (long)len;
}

/* ---- read ------------------------------------------------------- */

int wubu_ms_is_header(const char *s, size_t len)
{
    if (!s) return 0;
    if (strlen(s) != len) return 0;          /* an embedded NUL is not a header */
    return len == WUBU_MS_HEADER_LEN &&
           memcmp(s, WUBU_MS_HEADER, WUBU_MS_HEADER_LEN) == 0;
}

int wubu_ms_read(const uint8_t *buf, size_t n, size_t *pos,
                 const char **str, char *scratch, size_t scratch_len,
                 const uint8_t **data, size_t *len)
{
    if (!buf || !pos || !str || !data || !len) return -1;
    if (*pos >= n) return -1;

    uint64_t flen = 0;
    size_t used = 0;
    if (wubu_varint_decode(buf + *pos, n - *pos, &flen, &used) != 0)
        return -1;
    if (flen > n - *pos - used) return -1;   /* the frame does not fit */
    if (flen > WUBU_MS_MAX_FRAME) return -1;

    size_t body = *pos + used;
    size_t blen = (size_t)flen;
    *pos = body + blen;

    /* Anything printable ending in a newline is a string message -- the
       handshake phase, or 'na'.  Anything else is an opaque data frame from a
       negotiated protocol, which this layer must not interpret. */
    int is_string = 0;
    if (blen > 0 && buf[body + blen - 1] == '\n') {
        is_string = 1;
        /* The trailing newline is required and is not printable (0x0A), so it
           sits outside this scan rather than failing it. */
        for (size_t i = 0; i + 1 < blen; i++) {
            unsigned char c = buf[body + i];
            if (c < 0x20 || c > 0x7E) { is_string = 0; break; }
        }
    }

    if (!is_string) {
        *data = buf + body;
        *len = blen;
        *str = NULL;
        return WUBU_MS_DATA;
    }

    if (!scratch || scratch_len < blen + 1) return -1;
    memcpy(scratch, buf + body, blen);
    scratch[blen] = '\0';
    *str = scratch;
    *len = blen;
    *data = NULL;
    return WUBU_MS_STRING;
}

int wubu_ms_read_string(const uint8_t *buf, size_t n, size_t *pos,
                        const char **str, char *scratch, size_t scratch_len)
{
    const uint8_t *d = NULL;
    size_t dl = 0;
    int r = wubu_ms_read(buf, n, pos, str, scratch, scratch_len, &d, &dl);
    if (r != WUBU_MS_STRING) return -1;
    return 0;
}

/* ---- negotiate -------------------------------------------------- */

int wubu_ms_select(const char *const *wanted, int nwanted,
                   const char *incoming, size_t ilen)
{
    if (!wanted || !incoming || ilen == 0) return -1;
    if (memchr(incoming, '\0', ilen)) return -1;
    if (!proto_valid(incoming, ilen)) return -1;

    /* The wire form carries a trailing newline; the wanted table carries bare
       names.  Normalise once here rather than forcing every caller to store
       "/ipfs/id/1.0.0\n" in its table. */
    char have[WUBU_MS_MAX_PROTO];
    if (ilen >= sizeof(have)) return -1;
    memcpy(have, incoming, ilen);
    have[ilen] = '\0';
    size_t hl = strlen(have);
    if (hl && have[hl - 1] == '\n') have[hl - 1] = '\0';

    for (int i = 0; i < nwanted; i++) {
        if (!wanted[i]) continue;
        if (strcmp(wanted[i], have) == 0) return i;
    }
    return -1;
}
