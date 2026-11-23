/* test_eg70_cert.c -- EG-70 gate: output-exactness certificate. */
#include <stdio.h>

#include "wubu_exact_cert.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_exact_cert_t *p = wubu_exact_cert_create(0x1234ULL, 4096, 0, 128);
    float logits[4] = {0.25f, -0.5f, 1.75f, 0.0f};
    float tampered[4] = {0.25f, -0.5f, 1.75f, 0.0001f};
    CHECK(p != NULL);

    CHECK(wubu_exact_cert_issue(p, logits, 4, 1) == 0);
    CHECK(wubu_exact_cert_verify(p, logits, 4) == 0);
    CHECK(wubu_exact_cert_verify(p, tampered, 4) == -1);
    printf("EG-70 certificate id=0x1234 semantic=4096 authority=hard horizon=128\n");
    CHECK(wubu_exact_cert_verify(p, logits, 3) == -1);
    CHECK(wubu_exact_cert_verify(NULL, logits, 4) == -1);

    /* Encode/decode round-trip and tamper detection. */
    {
        unsigned char buf[1024];
        wubu_exact_cert_t *q = NULL, *bad = NULL;
        size_t sz = wubu_exact_cert_encoded_size(p);
        CHECK(sz <= sizeof(buf));
        CHECK(wubu_exact_cert_encode(p, buf, sizeof(buf)) == (int)sz);
        CHECK(wubu_exact_cert_decode(buf, sz, &q) == 0);
        CHECK(q != NULL);
        CHECK(wubu_exact_cert_equal(p, q) == 1);
        CHECK(wubu_exact_cert_verify(q, logits, 4) == 0);
        buf[16] ^= 1;                       /* corrupt the identity field */
        CHECK(wubu_exact_cert_decode(buf, sz, &bad) == -1);
        CHECK(bad == NULL);
        printf("EG-70 encode/decode round-trip=%zu bytes, tamper rejected\n", sz);
        wubu_exact_cert_free(q);
    }

    /* Reuse generations must strictly advance. */
    CHECK(wubu_exact_cert_reuse(p, 2) == 0);
    CHECK(wubu_exact_cert_generation(p) == 2);
    CHECK(wubu_exact_cert_reuse(p, 2) == -1);
    CHECK(wubu_exact_cert_reuse(p, 1) == -1);
    CHECK(wubu_exact_cert_verify(p, logits, 4) == 0);
    CHECK(wubu_exact_cert_create(0, 4096, 0, 128) == NULL);
    CHECK(wubu_exact_cert_create(1, 0, 0, 128) == NULL);
    CHECK(wubu_exact_cert_create(1, 4096, 4, 128) == NULL);
    printf("EG-70 reuse generation 1->2 accepted, stale rejected\n");

    wubu_exact_cert_free(p);
    if (failures) { fprintf(stderr, "test_eg70_cert: FAIL (%d)\n", failures); return 1; }
    printf("test_eg70_cert: PASS\n");
    return 0;
}
