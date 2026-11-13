/*
 * wubu_dht.c -- BitTorrent DHT (BEP 5): identity, distance, KRPC.
 *
 * C11, libc only.  The bencode here is deliberately tiny: it writes the exact
 * dictionaries these four queries need and reads back the two response shapes,
 * rather than being a general parser.  A general parser belongs with the
 * .torrent reader once both need one.
 */
#include "wubu_dht.h"
#include "wubu_sha1.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ---- identity and distance ------------------------------------- */

int wubu_dht_node_id(const uint8_t *seed, size_t len,
                     uint8_t id[WUBU_DHT_ID_LEN])
{
    if (!id) return -1;
    if (len && !seed) return -1;
    return wubu_sha1(seed, len, id) == 0 ? 0 : -1;
}

void wubu_dht_distance(const uint8_t a[WUBU_DHT_ID_LEN],
                       const uint8_t b[WUBU_DHT_ID_LEN],
                       uint8_t out[WUBU_DHT_ID_LEN])
{
    for (int i = 0; i < WUBU_DHT_ID_LEN; i++)
        out[i] = (uint8_t)(a[i] ^ b[i]);
}

int wubu_dht_distance_cmp(const uint8_t x[WUBU_DHT_ID_LEN],
                          const uint8_t a[WUBU_DHT_ID_LEN],
                          const uint8_t b[WUBU_DHT_ID_LEN])
{
    for (int i = 0; i < WUBU_DHT_ID_LEN; i++) {
        uint8_t da = (uint8_t)(x[i] ^ a[i]);
        uint8_t db = (uint8_t)(x[i] ^ b[i]);
        if (da != db) return da < db ? -1 : 1;
    }
    return 0;
}

int wubu_dht_shared_prefix_bits(const uint8_t a[WUBU_DHT_ID_LEN],
                                const uint8_t b[WUBU_DHT_ID_LEN])
{
    int bits = 0;
    for (int i = 0; i < WUBU_DHT_ID_LEN; i++) {
        uint8_t x = (uint8_t)(a[i] ^ b[i]);
        if (x == 0) { bits += 8; continue; }
        /* Leading zero bits of this byte are the shared prefix's tail. */
        int z = 0;
        for (uint8_t m = 0x80; m && !(x & m); m = (uint8_t)(m >> 1)) z++;
        return bits + z;
    }
    return bits;                         /* identical ids: all 160 bits */
}

/* ---- query names ----------------------------------------------- */

static const wubu_dht_query_info_t g_queries[] = {
    { WUBU_DHT_Q_PING,          "ping" },
    { WUBU_DHT_Q_FIND_NODE,     "find_node" },
    { WUBU_DHT_Q_GET_PEERS,     "get_peers" },
    { WUBU_DHT_Q_ANNOUNCE_PEER, "announce_peer" },
};

const wubu_dht_query_info_t *wubu_dht_query_info(wubu_dht_query_t q)
{
    for (unsigned i = 0; i < sizeof(g_queries) / sizeof(g_queries[0]); i++)
        if (q == (wubu_dht_query_t)g_queries[i].query) return &g_queries[i];
    return NULL;
}

const wubu_dht_query_info_t *wubu_dht_query_by_name(const char *name)
{
    if (!name) return NULL;
    for (unsigned i = 0; i < sizeof(g_queries) / sizeof(g_queries[0]); i++)
        if (strcmp(g_queries[i].name, name) == 0) return &g_queries[i];
    return NULL;
}

/* ---- a tiny bencode writer ------------------------------------- */

typedef struct { uint8_t *buf; size_t n; size_t o; } wr_t;

static int wr_raw(wr_t *w, const void *p, size_t len)
{
    if (w->o + len > w->n) return -1;
    memcpy(w->buf + w->o, p, len);
    w->o += len;
    return 0;
}

static int wr_str(wr_t *w, const char *s)
{
    int n = snprintf((char *)w->buf + w->o, w->n - w->o, "%s", s);
    if (n < 0 || (size_t)n >= w->n - w->o) return -1;
    w->o += (size_t)n;
    return 0;
}

static int wr_int(wr_t *w, long long v)
{
    char tmp[32];
    int n = snprintf(tmp, sizeof(tmp), "i%llde", v);
    return wr_raw(w, tmp, (size_t)n);
}

static int wr_bstr(wr_t *w, const uint8_t *p, size_t len)
{
    char tmp[32];
    int n = snprintf(tmp, sizeof(tmp), "%zu:", len);
    if (wr_raw(w, tmp, (size_t)n) != 0) return -1;
    return wr_raw(w, p, len);
}

/* ---- request / response encoding ------------------------------- */

long wubu_dht_encode_request(wubu_dht_query_t q, const wubu_dht_args_t *args,
                             uint8_t *out, size_t n)
{
    if (!args || !out) return -1;
    const wubu_dht_query_info_t *qi = wubu_dht_query_info(q);
    if (!qi) return -1;

    wr_t w = { out, n, 0 };
    if (wr_str(&w, "d1:ad2:id") != 0) return -1;
    if (wr_bstr(&w, args->id, WUBU_DHT_ID_LEN) != 0) return -1;

    if (args->have_target) {
        if (wr_str(&w, "6:target") != 0) return -1;
        if (wr_bstr(&w, args->target, WUBU_DHT_ID_LEN) != 0) return -1;
    }
    if (args->have_token) {
        if (wr_str(&w, "5:token") != 0) return -1;
        if (wr_bstr(&w, args->token, WUBU_DHT_TOKEN_LEN) != 0) return -1;
    }
    if (args->have_port) {
        if (wr_str(&w, "4:port") != 0) return -1;
        if (wr_int(&w, args->port) != 0) return -1;
    }
    if (wr_str(&w, "e1:q") != 0) return -1;
    if (wr_bstr(&w, (const uint8_t *)qi->name, strlen(qi->name)) != 0) return -1;
    if (!args->have_txn) return -1;
    if (wr_str(&w, "1:t") != 0) return -1;
    if (wr_bstr(&w, args->txn, sizeof(args->txn)) != 0) return -1;
    if (wr_str(&w, "1:y1:qe") != 0) return -1;
    return (long)w.o;
}

long wubu_dht_encode_response(const uint8_t id[WUBU_DHT_ID_LEN],
                              const uint8_t token[WUBU_DHT_TOKEN_LEN],
                              int have_token, const uint8_t *value,
                              size_t value_len, uint8_t *out, size_t n)
{
    if (!id || !out) return -1;
    if (have_token && !token) return -1;
    if (value && !value_len) return -1;

    wr_t w = { out, n, 0 };
    if (wr_str(&w, "d1:rd2:id") != 0) return -1;
    if (wr_bstr(&w, id, WUBU_DHT_ID_LEN) != 0) return -1;
    if (have_token) {
        if (wr_str(&w, "5:token") != 0) return -1;
        if (wr_bstr(&w, token, WUBU_DHT_TOKEN_LEN) != 0) return -1;
    }
    if (value) {
        if (wr_str(&w, "5:nodes") != 0) return -1;
        if (wr_bstr(&w, value, value_len) != 0) return -1;
    }
    if (wr_str(&w, "e1:y1:re") != 0) return -1;
    return (long)w.o;
}

/* ---- decoding -------------------------------------------------- */

typedef struct { const uint8_t *b; size_t n; size_t p; } rd_t;

static int rd_match(rd_t *r, char c)
{
    if (r->p >= r->n || r->b[r->p] != (uint8_t)c) return -1;
    r->p++;
    return 0;
}

/* A bencoded byte-string: <len>:<bytes>.  Returns the span or -1. */
static long rd_bstr_span(rd_t *r, size_t *start, size_t *len)
{
    if (r->p >= r->n || r->b[r->p] < '0' || r->b[r->p] > '9') return -1;
    size_t v = 0;
    int digits = 0;
    while (r->p < r->n && r->b[r->p] >= '0' && r->b[r->p] <= '9') {
        v = v * 10 + (size_t)(r->b[r->p] - '0');
        if (v > r->n) return -1;
        r->p++;
        digits++;
    }
    if (digits == 0 || (digits > 1 && r->b[r->p - digits] == '0')) return -1;
    if (rd_match(r, ':') != 0) return -1;
    if (v > r->n - r->p) return -1;
    *start = r->p;
    *len = v;
    r->p += v;
    return 0;
}

/* Skip any one value.  Strings are parsed as STRINGS -- counting 'd'/'l'/'e'
 * bytes directly would desync on an 'e' that is part of a payload, which is
 * exactly what a node id and a token contain. */
static int rd_skip_value(rd_t *r)
{
    if (r->p >= r->n) return -1;
    char c = (char)r->b[r->p];
    if (c == 'i') {
        r->p++;
        while (r->p < r->n && r->b[r->p] != 'e') r->p++;
        if (r->p >= r->n) return -1;
        r->p++;
        return 0;
    }
    if (c == 'l') {
        r->p++;
        while (r->p < r->n && r->b[r->p] != 'e')
            if (rd_skip_value(r) != 0) return -1;
        if (r->p >= r->n) return -1;
        r->p++;
        return 0;
    }
    if (c == 'd') {
        r->p++;
        while (r->p < r->n && r->b[r->p] != 'e') {
            size_t s, l;
            if (rd_bstr_span(r, &s, &l) != 0) return -1;
            if (rd_skip_value(r) != 0) return -1;
        }
        if (r->p >= r->n) return -1;
        r->p++;
        return 0;
    }
    size_t s, l;
    return rd_bstr_span(r, &s, &l);
}

/* Read one dictionary entry: the KEY IS CONSUMED, then either its value is
 * copied (key matched `want`) or the value is skipped whole.
 * Returns 1 on a match, 0 on a non-match, -1 on anything malformed.
 *
 * `exact_len` > 0 requires the value to be exactly that many bytes; 0 accepts
 * any length up to `cap`. */
static int rd_entry(rd_t *r, const char *want, size_t want_len,
                    uint8_t *dst, size_t cap, size_t exact_len, size_t *got)
{
    size_t ks, kl;
    if (rd_bstr_span(r, &ks, &kl) != 0) return -1;
    if (kl == want_len && memcmp(r->b + ks, want, want_len) == 0) {
        size_t vs, vl;
        if (rd_bstr_span(r, &vs, &vl) != 0) return -1;
        if (vl > cap) return -1;
        if (exact_len && vl != exact_len) return -1;
        if (dst) memcpy(dst, r->b + vs, vl);
        if (got) *got = vl;
        return 1;
    }
    return rd_skip_value(r) == 0 ? 0 : -1;
}

/* The `r` dictionary of a response: id, then optionally token / nodes.
 * An error body is NOT an r-dict -- BEP 5 puts it under `e` -- so this only
 * ever sees real responses. */
static int rd_read_r_dict(rd_t *r, wubu_dht_decoded_t *d)
{
    if (rd_match(r, 'd') != 0) return -1;

    int saw_id = 0;
    while (r->p < r->n && r->b[r->p] != 'e') {
        size_t ks, kl;
        if (rd_bstr_span(r, &ks, &kl) != 0) return -1;
        const char *k = (const char *)r->b + ks;

        if (kl == 2 && memcmp(k, "id", 2) == 0) {
            size_t vs, vl;
            if (rd_bstr_span(r, &vs, &vl) != 0 || vl != WUBU_DHT_ID_LEN)
                return -1;
            memcpy(d->id, r->b + vs, WUBU_DHT_ID_LEN);
            d->have_id = 1;
            saw_id = 1;
            continue;
        }
        if (kl == 5 && memcmp(k, "token", 5) == 0) {
            size_t vs, vl;
            if (rd_bstr_span(r, &vs, &vl) != 0 || vl != WUBU_DHT_TOKEN_LEN)
                return -1;
            memcpy(d->token, r->b + vs, WUBU_DHT_TOKEN_LEN);
            d->have_token = 1;
            continue;
        }
        if (kl == 5 && memcmp(k, "nodes", 5) == 0) {
            size_t vs, vl;
            if (rd_bstr_span(r, &vs, &vl) != 0 || vl != WUBU_DHT_ID_LEN)
                return -1;
            memcpy(d->value, r->b + vs, WUBU_DHT_ID_LEN);
            d->have_value = 1;
            continue;
        }
        /* Unknown field: its value must still be skipped to stay aligned. */
        if (rd_skip_value(r) != 0) return -1;
    }
    if (rd_match(r, 'e') != 0) return -1;
    return saw_id ? 0 : -1;
}

/* The `a` dictionary of a request: id plus whatever arguments the query
 * carries.  Only the id is surfaced; the rest is validated and skipped. */
static int rd_read_a_dict(rd_t *r, wubu_dht_decoded_t *d)
{
    if (rd_match(r, 'd') != 0) return -1;

    int saw_id = 0;
    while (r->p < r->n && r->b[r->p] != 'e') {
        size_t ks, kl;
        if (rd_bstr_span(r, &ks, &kl) != 0) return -1;
        const char *k = (const char *)r->b + ks;

        if (kl == 2 && memcmp(k, "id", 2) == 0) {
            size_t vs, vl;
            if (rd_bstr_span(r, &vs, &vl) != 0 || vl != WUBU_DHT_ID_LEN)
                return -1;
            memcpy(d->id, r->b + vs, WUBU_DHT_ID_LEN);
            d->have_id = 1;
            saw_id = 1;
            continue;
        }
        if (rd_skip_value(r) != 0) return -1;
    }
    if (rd_match(r, 'e') != 0) return -1;
    return saw_id ? 0 : -1;
}

/* A BEP 5 error body: l i<code>e <message> e.  The code is recovered; the
 * message is validated and dropped. */
static int rd_read_error_list(rd_t *r, wubu_dht_decoded_t *d)
{
    if (rd_match(r, 'l') != 0) return -1;
    if (rd_match(r, 'i') != 0) return -1;
    size_t ds = r->p;
    while (r->p < r->n && r->b[r->p] != 'e') r->p++;
    if (r->p >= r->n) return -1;
    char num[24];
    size_t nl = r->p - ds;
    if (nl >= sizeof(num)) return -1;
    memcpy(num, r->b + ds, nl);
    num[nl] = '\0';
    d->error_code = atoi(num);
    r->p++;                                      /* the integer's 'e' */
    size_t ms, ml;
    if (rd_bstr_span(r, &ms, &ml) != 0) return -1;   /* the message */
    return rd_match(r, 'e');                         /* closes the list */
}

int wubu_dht_decode(const uint8_t *buf, size_t n, wubu_dht_decoded_t *out)
{
    if (!buf || !out) return -1;

    rd_t r = { buf, n, 0 };
    wubu_dht_decoded_t d;
    memset(&d, 0, sizeof(d));

    if (rd_match(&r, 'd') != 0) return -1;

    while (r.p < r.n && buf[r.p] != 'e') {
        size_t ks, kl;
        if (rd_bstr_span(&r, &ks, &kl) != 0) return -1;   /* the key */
        const char *k = (const char *)buf + ks;

        if (kl == 1 && k[0] == 'y') {
            size_t vs, vl;
            if (rd_bstr_span(&r, &vs, &vl) != 0 || vl != 1) return -1;
            char y = (char)buf[vs];
            if (y == 'q')           { d.is_response = 0; d.is_error = 0; }
            else if (y == 'r')      { d.is_response = 1; d.is_error = 0; }
            else if (y == 'e')      { d.is_response = 1; d.is_error = 1; }
            else return -1;
        } else if (kl == 1 && k[0] == 't') {
            size_t vs, vl;
            if (rd_bstr_span(&r, &vs, &vl) != 0 || vl > sizeof(d.token))
                return -1;
            memcpy(d.token, buf + vs, vl);
            d.have_token = 1;
        } else if (kl == 1 && k[0] == 'r') {
            if (rd_read_r_dict(&r, &d) != 0) return -1;
        } else if (kl == 1 && k[0] == 'e') {
            if (rd_read_error_list(&r, &d) != 0) return -1;
        } else if (kl == 1 && k[0] == 'a') {
            if (rd_read_a_dict(&r, &d) != 0) return -1;
        } else {
            if (rd_skip_value(&r) != 0) return -1;   /* unknown argument */
        }
    }
    if (rd_match(&r, 'e') != 0) return -1;       /* closes the outer dict */
    if (r.p != r.n) return -1;                   /* trailing bytes: refuse */

    memcpy(out, &d, sizeof(d));
    return 0;
}
