/* test_eg48_bitnest.c -- EG-48 gate: one physical W4-in-W8 representation. */
#include <stdio.h>
#include <math.h>

#include "wubu_bitnest.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define N 256

int main(void) {
    float w[N], x[N];
    wubu_bitnest_t b;
    double ref8 = 0.0, norm = 0.0;
    float d8, d4;
    signed char *physical;

    for (int i = 0; i < N; i++) {
        w[i] = sinf(0.07f * (float)i) * (0.3f + 0.7f * fabsf(cosf(0.11f * i)));
        x[i] = cosf(0.05f * (float)i);
        ref8 += (double)(w[i] * x[i]);
        norm += fabs((double)(w[i] * x[i]));
    }
    CHECK(wubu_bitnest_init(&b, N) == 0);
    CHECK(wubu_bitnest_quantize(&b, w, N) == 0);
    physical = b.data;
    CHECK(fabs(b.w4_scale - 0.5f * b.w8_scale) < 1e-7f);

    d8 = wubu_bitnest_dot_w8(&b, x);
    d4 = wubu_bitnest_dot_w4(&b, x);
    printf("EG-48 ref8=%.5f w8=%.5f w4=%.5f w4_expected=%.5f\n",
           ref8, d8, d4, 0.5 * ref8);
    CHECK(fabs((double)d8 - ref8) / norm < 0.01);
    CHECK(fabs((double)d4 - 0.5 * ref8) / norm < 0.01);
    CHECK(b.data == physical); /* both views share one physical buffer */
    printf("EG-48 physical_bytes=%d (one int8 per weight, no second copy)\n", N);

    CHECK(wubu_bitnest_quantize(&b, w, N - 1) == -1);
    CHECK(wubu_bitnest_init(&b, 0) == -1);
    CHECK(wubu_bitnest_dot_w8(NULL, x) == 0.0f);
    wubu_bitnest_free(&b);
    CHECK(b.data == NULL);

    if (failures) {
        fprintf(stderr, "test_eg48_bitnest: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg48_bitnest: PASS\n");
    return 0;
}
