/* wubu_exact_cert.c -- EG-70 output-exactness certificate. Strict C11. */
#include "wubu_exact_cert.h"

#include <stdlib.h>
#include <string.h>

struct wubu_exact_cert {
    uint64_t identity, generation;
    size_t semantic_bytes;
    int authority, horizon, n, issued;
    float logits[WUBU_CERT_MAX_LOGITS];
};

wubu_exact_cert_t *wubu_exact_cert_create(uint64_t identity,
                                          size_t semantic_bytes,
                                          int authority, int horizon)
{
    wubu_exact_cert_t *p;
    if (identity == 0 || semantic_bytes == 0 || authority < 0 || authority > 3 ||
        horizon <= 0) return NULL;
    p = (wubu_exact_cert_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->identity = identity;
    p->semantic_bytes = semantic_bytes;
    p->authority = authority;
    p->horizon = horizon;
    return p;
}

void wubu_exact_cert_free(wubu_exact_cert_t *p) { free(p); }

int wubu_exact_cert_issue(wubu_exact_cert_t *p, const float *logits, int n,
                          uint64_t generation)
{
    if (!p || !logits || n < 1 || n > WUBU_CERT_MAX_LOGITS || generation == 0)
        return -1;
    p->n = n;
    p->generation = generation;
    memcpy(p->logits, logits, (size_t)n * sizeof(float));
    p->issued = 1;
    return 0;
}

int wubu_exact_cert_verify(const wubu_exact_cert_t *p,
                           const float *logits, int n)
{
    if (!p || !p->issued || !logits || n != p->n) return -1;
    return memcmp(p->logits, logits, (size_t)n * sizeof(float)) == 0 ? 0 : -1;
}

/* Wire layout: magic[8] | identity | semantic | authority | horizon | n |
 * issued | generation | logits[n] | FNV-1a checksum. All fields little-endian
 * memcpy of native integers; this gate checks round-trip on one host. */
#define CERT_HEADER 48
#define CERT_SUFFIX 8

static uint64_t cert_checksum(const unsigned char *buf, size_t n)
{
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= buf[i];
        h *= 1099511628211ULL;
    }
    return h;
}

size_t wubu_exact_cert_encoded_size(const wubu_exact_cert_t *p)
{
    return p ? CERT_HEADER + (size_t)p->n * sizeof(float) + CERT_SUFFIX : 0;
}

int wubu_exact_cert_encode(const wubu_exact_cert_t *p,
                           unsigned char *buf, size_t cap)
{
    size_t need, off = CERT_HEADER;
    uint64_t sum, v;
    int32_t iv;
    if (!p || !buf) return -1;
    need = wubu_exact_cert_encoded_size(p);
    if (cap < need) return -1;
    memcpy(buf, "WUCERT01", 8);
    memcpy(buf + 8, &p->identity, 8);
    v = (uint64_t)p->semantic_bytes; memcpy(buf + 16, &v, 8);
    iv = p->authority; memcpy(buf + 24, &iv, 4);
    iv = p->horizon;   memcpy(buf + 28, &iv, 4);
    iv = p->n;         memcpy(buf + 32, &iv, 4);
    iv = p->issued;    memcpy(buf + 36, &iv, 4);
    memcpy(buf + 40, &p->generation, 8);
    for (int i = 0; i < p->n; ++i) {
        memcpy(buf + off, &p->logits[i], 4);
        off += 4;
    }
    sum = cert_checksum(buf, off);
    memcpy(buf + off, &sum, 8);
    return (int)need;
}

int wubu_exact_cert_decode(const unsigned char *buf, size_t n,
                           wubu_exact_cert_t **out)
{
    wubu_exact_cert_t *p;
    uint64_t sum, v;
    int32_t iv;
    size_t off;
    int count;
    if (!buf || !out || n < CERT_HEADER + CERT_SUFFIX ||
        memcmp(buf, "WUCERT01", 8) != 0) return -1;
    memcpy(&sum, buf + n - CERT_SUFFIX, 8);
    if (cert_checksum(buf, n - CERT_SUFFIX) != sum) return -1;
    memcpy(&v, buf + 16, 8);
    memcpy(&iv, buf + 24, 4);
    if (iv < 0 || iv > 3) return -1;
    memcpy(&iv, buf + 28, 4);
    if (iv <= 0) return -1;
    memcpy(&iv, buf + 32, 4);
    count = iv;
    if (count < 0 || count > WUBU_CERT_MAX_LOGITS ||
        n != CERT_HEADER + (size_t)count * 4 + CERT_SUFFIX) return -1;
    p = (wubu_exact_cert_t *)calloc(1, sizeof(*p));
    if (!p) return -1;
    memcpy(&p->identity, buf + 8, 8);
    p->semantic_bytes = (size_t)v;
    memcpy(&iv, buf + 24, 4); p->authority = iv;
    memcpy(&iv, buf + 28, 4); p->horizon = iv;
    memcpy(&iv, buf + 36, 4); p->issued = iv;
    memcpy(&p->generation, buf + 40, 8);
    p->n = count;
    off = CERT_HEADER;
    for (int i = 0; i < count; ++i) {
        memcpy(&p->logits[i], buf + off, 4);
        off += 4;
    }
    *out = p;
    return 0;
}

int wubu_exact_cert_equal(const wubu_exact_cert_t *p,
                          const wubu_exact_cert_t *q)
{
    if (!p || !q) return -1;
    return p->identity == q->identity && p->generation == q->generation &&
           p->semantic_bytes == q->semantic_bytes &&
           p->authority == q->authority && p->horizon == q->horizon &&
           p->n == q->n && p->issued == q->issued &&
           memcmp(p->logits, q->logits, (size_t)p->n * sizeof(float)) == 0;
}

int wubu_exact_cert_reuse(wubu_exact_cert_t *p, uint64_t new_generation)
{
    if (!p || new_generation <= p->generation) return -1;
    p->generation = new_generation;
    return 0;
}

uint64_t wubu_exact_cert_generation(const wubu_exact_cert_t *p)
{
    return p ? p->generation : 0;
}
