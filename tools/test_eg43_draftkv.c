/* test_eg43_draftkv.c -- EG-43 gate: K4V4 per-channel/per-token draft KV. */
#include <stdio.h>
#include <math.h>

#include "wubu_draft_kv.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define ROWS 4
#define D 4
#define N (ROWS * D)

int main(void) {
    float k[N] = { 1, -2, 0.5, 0,  2, 1.5, -1, 0.25,
                  -0.5, 2, 1, -1.5, 0.75, -2, 0.25, 1.5 };
    float v[N] = { 2, 0.5, -1, 0,  -0.25, 1.5, -2, 0.75,
                  1, -1.5, 0.5, 2,  -0.5, 0.25, -1, 1.5 };
    uint8_t kq[N/2], vq[N/2];
    float ks[D], vs[ROWS], kr[N], vr[N];
    float kerr = 0.0f, verr = 0.0f;
    int i;

    CHECK(wubu_draft_kv_pack(k, v, ROWS, D, kq, vq, ks, vs) == 0);
    CHECK(wubu_draft_kv_unpack_k(kq, ROWS, D, ks, kr) == 0);
    CHECK(wubu_draft_kv_unpack_v(vq, ROWS, D, vs, vr) == 0);
    for (i = 0; i < N; i++) {
        float e = fabsf(k[i] - kr[i]);
        float f = fabsf(v[i] - vr[i]);
        if (e > kerr) kerr = e;
        if (f > verr) verr = f;
    }
    printf("EG-43 K maxerr=%.4f (scale=%.4f) V maxerr=%.4f\n",
           kerr, ks[0], verr);
    CHECK(kerr <= ks[0] / 2.0f + 1e-6f);
    CHECK(verr <= vs[0] / 2.0f + 1e-6f);
    CHECK(kerr < 0.2f && verr < 0.2f);
    CHECK(sizeof(kq) * 8 == sizeof(k)); /* 4 bits per element, K and V */

    /* Per-channel key scales must actually differ across channels. */
    CHECK(fabsf(ks[0] - ks[2]) > 1e-6f);

    CHECK(wubu_draft_kv_pack(NULL, v, ROWS, D, kq, vq, ks, vs) == -1);
    CHECK(wubu_draft_kv_pack(k, v, 3, 3, kq, vq, ks, vs) == -1); /* odd n */
    CHECK(wubu_draft_kv_unpack_k(NULL, ROWS, D, ks, kr) == -1);
    CHECK(wubu_draft_kv_unpack_v(vq, ROWS, D, NULL, vr) == -1);

    if (failures) {
        fprintf(stderr, "test_eg43_draftkv: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg43_draftkv: PASS\n");
    return 0;
}
