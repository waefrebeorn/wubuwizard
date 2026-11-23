#ifndef WUBU_EXACT_CERT_H
#define WUBU_EXACT_CERT_H

/*
 * wubu_exact_cert.h -- EG-70: falsifiable output-exactness certificate.
 * Memory-Sovereign C=(I,B,A,E,L): identity, semantic-demand bound, authority,
 * exactness equivalence and lifecycle generation. A certificate is valid only
 * for bit-identical logit rows and a strictly advancing reuse generation.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_CERT_MAX_LOGITS 256

typedef struct wubu_exact_cert wubu_exact_cert_t;

/* identity != 0, semantic_bytes > 0, authority in [0,3], horizon > 0. */
wubu_exact_cert_t *wubu_exact_cert_create(uint64_t identity,
                                          size_t semantic_bytes,
                                          int authority, int horizon);
void wubu_exact_cert_free(wubu_exact_cert_t *p);

/* Bind a logit row and generation to the certificate. 0/-1. */
int wubu_exact_cert_issue(wubu_exact_cert_t *p, const float *logits, int n,
                          uint64_t generation);

/* 0 only for a bit-identical row of the same length; -1 otherwise. */
int wubu_exact_cert_verify(const wubu_exact_cert_t *p,
                           const float *logits, int n);

/* Encode/decode round-trip with a checksum; decode allocates *out on success
 * (caller frees). Returns encoded byte count, or -1 on bad input/tampering. */
size_t wubu_exact_cert_encoded_size(const wubu_exact_cert_t *p);
int wubu_exact_cert_encode(const wubu_exact_cert_t *p,
                           unsigned char *buf, size_t cap);
int wubu_exact_cert_decode(const unsigned char *buf, size_t n,
                           wubu_exact_cert_t **out);
int wubu_exact_cert_equal(const wubu_exact_cert_t *p,
                          const wubu_exact_cert_t *q);

/* Async buffer reuse is safe only with a strictly advancing generation. */
int wubu_exact_cert_reuse(wubu_exact_cert_t *p, uint64_t new_generation);
uint64_t wubu_exact_cert_generation(const wubu_exact_cert_t *p);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_EXACT_CERT_H */
