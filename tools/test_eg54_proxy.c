/* test_eg54_proxy.c -- EG-54 gate: block summary vs dropped-block baseline. */
#include <stdio.h>
#include <math.h>

#include "wubu_proxy_route.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define NB 6
#define BS 4
#define D 6

static double relerr(const float *a, const float *b) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < D; i++) { num += fabs(a[i] - b[i]); den += fabs(a[i]); }
    return num / den;
}

int main(void) {
    float keys[NB*BS*D], sum[NB*D], q[D], full[D], drop[D], proxy[D];
    int sel[NB];
    double e_drop, e_proxy;

    for (int b = 0; b < NB; b++) {
        int hot = (b % 2 == 0);
        for (int t = 0; t < BS; t++)
            for (int d = 0; d < D; d++)
                keys[((size_t)b * BS + t) * D + d] =
                    (hot ? 0.60f : 0.35f) + 0.02f * (float)((t + d) % 3);
    }
    for (int d = 0; d < D; d++) q[d] = 1.0f;

    CHECK(wubu_proxy_summary(keys, NB, BS, D, sum) == 0);
    /* Full attention reference. */
    {
        float mx = -INFINITY, den = 0.0f;
        for (int i = 0; i < NB*BS; i++) {
            const float *k = keys + (size_t)i * D;
            float s = 0.0f;
            for (int d = 0; d < D; d++) s += q[d] * k[d];
            if (s > mx) mx = s;
        }
        for (int d = 0; d < D; d++) full[d] = 0.0f;
        for (int i = 0; i < NB*BS; i++) {
            const float *k = keys + (size_t)i * D;
            float s = 0.0f;
            for (int d = 0; d < D; d++) s += q[d] * k[d];
            float w = expf(s - mx);
            den += w;
            for (int d = 0; d < D; d++) full[d] += w * k[d];
        }
        for (int d = 0; d < D; d++) full[d] /= den;
    }

    CHECK(wubu_proxy_route(keys, sum, NB, BS, D, q, 3, 0.0f, drop, sel) == 0);
    CHECK(wubu_proxy_route(keys, sum, NB, BS, D, q, 3, 1.0f, proxy, sel) == 0);
    e_drop = relerr(drop, full);
    e_proxy = relerr(proxy, full);
    printf("EG-54 selected=[%d %d %d] dropped_err=%.5f proxy_err=%.5f\n",
           sel[0], sel[1], sel[2], e_drop, e_proxy);
    CHECK(e_proxy < e_drop - 0.01);
    CHECK(e_proxy < 0.05);

    CHECK(wubu_proxy_summary(NULL, NB, BS, D, sum) == -1);
    CHECK(wubu_proxy_route(keys, sum, NB, BS, D, q, NB, 1.0f, proxy, sel) == -1);
    CHECK(wubu_proxy_route(keys, sum, NB, BS, D, q, 3, -1.0f, proxy, sel) == -1);

    if (failures) {
        fprintf(stderr, "test_eg54_proxy: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg54_proxy: PASS\n");
    return 0;
}
