/*
 * wubu_magnet.c -- BitTorrent magnet URIs (BEP 9).
 *
 * C11, libc only.  No hash is computed here: a btih must be the SHA-1 of a
 * real bencoded info dict, and inventing one would make the address a lie.
 */
#include "wubu_magnet.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ---- hex ------------------------------------------------------- */

static int hexval(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool wubu_magnet_valid_btih(const char *btih)
{
    if (!btih || strlen(btih) != WUBU_MAGNET_BTIH_LEN) return false;
    for (int i = 0; i < WUBU_MAGNET_BTIH_LEN; i++)
        if (hexval(btih[i]) < 0) return false;
    return true;
}

/* ---- percent encoding (RFC 3986, the unreserved set is left alone) -- */

static bool unreserved(unsigned char c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~';
}

/* Encode src into dst (capacity n, including the nul).  Returns bytes written
 * excluding the nul, or -1 if it does not fit. */
static long pct_encode(const char *src, char *dst, size_t n)
{
    static const char hex[] = "0123456789ABCDEF";
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)src; *p; p++) {
        if (unreserved(*p)) {
            if (o + 1 >= n) return -1;
            dst[o++] = (char)*p;
        } else {
            if (o + 3 >= n) return -1;
            dst[o++] = '%';
            dst[o++] = hex[*p >> 4];
            dst[o++] = hex[*p & 15];
        }
    }
    dst[o] = '\0';
    return (long)o;
}

/* Decode in place.  Returns 0, or -1 on a truncated or non-hex escape. */
static int pct_decode(char *s)
{
    char *o = s;
    for (char *p = s; *p; p++) {
        if (*p == '%') {
            int hi = hexval(p[1]);
            int lo = hexval(p[2]);
            /* p[1]/p[2] may read the nul and one past it -- the literal two
             * chars after a % in a nul-terminated string are always in bounds
             * because the string's own nul terminates the scan. */
            if (hi < 0 || lo < 0 || p[1] == '\0' || p[2] == '\0') return -1;
            *o++ = (char)((hi << 4) | lo);
            p += 2;
        } else {
            *o++ = *p;
        }
    }
    *o = '\0';
    return 0;
}

/* ---- build ----------------------------------------------------- */

long wubu_magnet_build(const wubu_magnet_t *m, char *out, size_t n)
{
    if (!m || !out || !wubu_magnet_valid_btih(m->btih)) return -1;

    char name_enc[WUBU_MAGNET_NAME_MAX * 3];
    if (m->name[0]) {
        if (pct_encode(m->name, name_enc, sizeof(name_enc)) < 0) return -1;
    }

    size_t o = 0;
    int written = snprintf(out + o, n - o, "magnet:?xt=%s%s",
                           WUBU_MAGNET_XT_BTIH, m->btih);
    if (written < 0 || (size_t)written >= n - o) return -1;
    o += (size_t)written;

    if (m->name[0]) {
        written = snprintf(out + o, n - o, "&dn=%s", name_enc);
        if (written < 0 || (size_t)written >= n - o) return -1;
        o += (size_t)written;
    }
    for (int i = 0; i < m->tracker_count && i < WUBU_MAGNET_TRACKERS; i++) {
        if (!m->trackers[i][0]) continue;
        written = snprintf(out + o, n - o, "&tr=%s", m->trackers[i]);
        if (written < 0 || (size_t)written >= n - o) return -1;
        o += (size_t)written;
    }
    if (m->length) {
        written = snprintf(out + o, n - o, "&xl=%llu",
                           (unsigned long long)m->length);
        if (written < 0 || (size_t)written >= n - o) return -1;
        o += (size_t)written;
    }
    return (long)o;
}

/* ---- parse ----------------------------------------------------- */

/* Split the query into key/value pairs in place.  A pair with no '=' is a
 * failure: every BEP 9 parameter is a value.  Returns the pair count or -1. */
static int split_pairs(char *query, char **keys, char **vals, int max)
{
    int n = 0;
    char *p = query;
    while (*p) {
        char *amp = strchr(p, '&');
        if (amp) *amp = '\0';
        char *eq = strchr(p, '=');
        if (!eq) return -1;
        *eq = '\0';
        if (n >= max) return -1;
        keys[n] = p;
        vals[n] = eq + 1;
        n++;
        if (!amp) break;
        p = amp + 1;
    }
    return n;
}

int wubu_magnet_parse(const char *uri, wubu_magnet_t *out)
{
    if (!uri || !out) return -1;
    if (strncmp(uri, "magnet:?", 8) != 0) return -1;

    char body[4096];
    size_t qlen = strlen(uri + 8);
    if (qlen >= sizeof(body)) return -1;
    memcpy(body, uri + 8, qlen + 1);

    /* Work on a copy so a refused URI never touches the caller's struct. */
    char *keys[32], *vals[32];
    int np = split_pairs(body, keys, vals, 32);
    if (np < 0) return -1;

    wubu_magnet_t m;
    memset(&m, 0, sizeof(m));
    int have_xt = 0;

    for (int i = 0; i < np; i++) {
        if (pct_decode(keys[i]) != 0) return -1;
        if (pct_decode(vals[i]) != 0) return -1;

        if (strcmp(keys[i], "xt") == 0) {
            if (strncmp(vals[i], WUBU_MAGNET_XT_BTIH,
                        strlen(WUBU_MAGNET_XT_BTIH)) == 0) {
                const char *h = vals[i] + strlen(WUBU_MAGNET_XT_BTIH);
                if (!wubu_magnet_valid_btih(h)) return -1;
                memcpy(m.btih, h, WUBU_MAGNET_BTIH_LEN);
                m.btih[WUBU_MAGNET_BTIH_LEN] = '\0';
                have_xt = 1;
            } else if (strncmp(vals[i], WUBU_MAGNET_XT_BTMH,
                               strlen(WUBU_MAGNET_XT_BTMH)) == 0) {
                /* v2 multihash: accept the shape, keep it verbatim so a v1
                 * swarm can still see the address.  It is not a v1 btih. */
                const char *h = vals[i] + strlen(WUBU_MAGNET_XT_BTMH);
                if (!*h || strlen(h) > WUBU_MAGNET_BTIH_LEN) return -1;
                memcpy(m.btih, h, strlen(h));
                have_xt = 1;
            } else {
                return -1;   /* an unknown exact topic is not a model address */
            }
        } else if (strcmp(keys[i], "dn") == 0) {
            if (strlen(vals[i]) >= sizeof(m.name)) return -1;
            memcpy(m.name, vals[i], strlen(vals[i]) + 1);
        } else if (strcmp(keys[i], "tr") == 0) {
            if (m.tracker_count >= WUBU_MAGNET_TRACKERS) continue;
            if (strlen(vals[i]) >= sizeof(m.trackers[0])) return -1;
            memcpy(m.trackers[m.tracker_count], vals[i], strlen(vals[i]) + 1);
            m.tracker_count++;
        } else if (strcmp(keys[i], "xl") == 0) {
            char *end = NULL;
            long long v = strtoll(vals[i], &end, 10);
            if (!end || *end != '\0' || v < 0) return -1;
            m.length = (uint64_t)v;
        }
        /* Unknown parameters are ignored: BEP 9 lets senders add them. */
    }

    if (!have_xt) return -1;   /* a magnet that names nothing is not an address */

    memcpy(out, &m, sizeof(m));
    return 0;
}
