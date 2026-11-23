/* test_eg36_fp8kv.c -- EG-36a gate: OCP FP8 KV round-trip. */
#include <stdio.h>
#include <math.h>

#include "wubu_fp8_kv.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    float k[8] = { 0.5f, -1.0f, 1.5f, -2.0f, 0.125f, -0.75f, 2.0f, 0.0f };
    float v[8] = { 2.0f, -0.25f, 0.75f, -1.5f, 0.5f, -0.125f, 1.0f, 0.0f };
    uint8_t qk[8], qv[8];
    float kr[8], vr[8], ks, vs, kerr = 0.0f, verr = 0.0f;
    int i;

    CHECK(wubu_fp8_kv_pack(k, 8, qk, &ks) == 0);
    CHECK(wubu_fp8_kv_unpack(qk, 8, ks, kr) == 0);
    CHECK(wubu_fp8_kv_pack(v, 8, qv, &vs) == 0);
    CHECK(wubu_fp8_kv_unpack(qv, 8, vs, vr) == 0);

    for (i = 0; i < 8; i++) {
        float e = fabsf(k[i] - kr[i]);
        float f = fabsf(v[i] - vr[i]);
        if (e > kerr) kerr = e;
        if (f > verr) verr = f;
    }
    printf("EG-36 K scale=%.6f maxerr=%.6f; V scale=%.6f maxerr=%.6f\n",
           ks, kerr, vs, verr);
    printf("EG-36 bytes fp8=%u f32=%u\n",
           (unsigned)(sizeof(qk) + sizeof(qv)),
           (unsigned)(sizeof(k) + sizeof(v)));
    CHECK(kerr <= 0.02f * 2.0f && verr <= 0.02f * 2.0f);
    CHECK(sizeof(qk) * 4 == sizeof(k));

    CHECK(wubu_fp8_kv_pack(NULL, 8, qk, &ks) == -1);
    CHECK(wubu_fp8_kv_pack(k, 0, qk, &ks) == -1);
    CHECK(wubu_fp8_kv_unpack(qk, 8, 0.0f, kr) == -1);
    CHECK(wubu_fp8_kv_unpack(NULL, 8, ks, kr) == -1);

    if (failures) {
        fprintf(stderr, "test_eg36_fp8kv: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg36_fp8kv: PASS\n");
    return 0;
}
