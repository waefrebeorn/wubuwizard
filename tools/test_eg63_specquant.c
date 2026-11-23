/* test_eg63_specquant.c -- EG-63 gate: routed precision variants. */
#include <stdio.h>
#include <math.h>

#include "wubu_specquant.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define N 256

static double doterr(const float *w, const float *x, const float *v, int n) {
    double ref = 0.0, got = 0.0;
    for (int i = 0; i < n; i++) { ref += w[i] * x[i]; got += v[i] * x[i]; }
    return fabs(got - ref) / (fabs(ref) + 1e-9);
}

int main(void) {
    float w[N], q[N], out[N], s;
    unsigned char buf[2 * N];
    int variant, v4, v16;
    double e4, e8, e16, erouted;
    size_t bytes4, bytes8, bytes16;

    for (int i = 0; i < N; i++) {
        w[i] = 0.5f * sinf(0.07f * i) + 0.01f * (float)(i % 3);
        q[i] = 1.0f; /* low-complexity query */
    }
    bytes4 = wubu_specquant_bytes(WUBU_SQ_INT4, N);
    bytes8 = wubu_specquant_bytes(WUBU_SQ_FP8, N);
    bytes16 = wubu_specquant_bytes(WUBU_SQ_FP16, N);
    printf("EG-63 bytes int4=%zu fp8=%zu fp16=%zu f32=%zu\n",
           bytes4, bytes8, bytes16, (size_t)N * 4);
    CHECK(bytes4 == 256 && bytes8 == 256 && bytes16 == 512);

    for (int v = 0; v < 3; v++) {
        CHECK(wubu_specquant_pack(w, N, v, buf, &s) == 0);
        CHECK(wubu_specquant_unpack(buf, N, v, s, out) == 0);
        if (v == WUBU_SQ_INT4) e4 = doterr(w, q, out, N);
        if (v == WUBU_SQ_FP8) e8 = doterr(w, q, out, N);
        if (v == WUBU_SQ_FP16) e16 = doterr(w, q, out, N);
    }
    printf("EG-63 err int4=%.5f fp8=%.5f fp16=%.5f\n", e4, e8, e16);
    CHECK(e16 < e8 && e8 < e4);
    CHECK(e16 < 0.01);

    v4 = wubu_specquant_route(q, N, 0.1f, 0.5f);
    for (int i = 0; i < N; i++) q[i] = (i & 1) ? 2.0f : -1.0f;
    v16 = wubu_specquant_route(q, N, 0.1f, 0.5f);
    CHECK(v4 == WUBU_SQ_INT4);
    CHECK(v16 == WUBU_SQ_FP16);

    for (int i = 0; i < N; i++) q[i] = 1.0f;
    variant = wubu_specquant_route(q, N, 0.1f, 0.5f);
    CHECK(wubu_specquant_pack(w, N, variant, buf, &s) == 0);
    CHECK(wubu_specquant_unpack(buf, N, variant, s, out) == 0);
    erouted = doterr(w, q, out, N);
    printf("EG-63 routed variant=%d relerr=%.5f bytes_saved=%.4f\n",
           variant, erouted,
           1.0 - (double)wubu_specquant_bytes(variant, N) / ((double)N * 4));
    CHECK(erouted < 0.02);
    CHECK((double)wubu_specquant_bytes(variant, N) < 0.5 * N * 4);

    CHECK(wubu_specquant_pack(w, N, 9, buf, &s) == -1);
    CHECK(wubu_specquant_route(NULL, N, 0.1f, 0.5f) == -1);
    CHECK(wubu_specquant_unpack(buf, N, WUBU_SQ_INT4, 0.0f, out) == -1);

    if (failures) {
        fprintf(stderr, "test_eg63_specquant: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg63_specquant: PASS\n");
    return 0;
}
