/* test_eg60_layerskip.c -- EG-60 gate: KV saved vs output parity. */
#include <stdio.h>
#include <math.h>

#include "wubu_layerskip.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define DIM 8
#define LAYERS 12

static double cosine(const float *a, const float *b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < DIM; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-12);
}

int main(void) {
    float x[DIM], zero[LAYERS], inc[LAYERS], full[DIM], nodrop[DIM], skip[DIM];
    size_t saved_zero, saved_skip, full_kv;

    for (int i = 0; i < DIM; i++) x[i] = 0.4f * (float)cos(0.5 * i);
    for (int l = 0; l < LAYERS; l++) {
        zero[l] = 0.0f;
        inc[l] = 0.9f * (float)l / (float)(LAYERS - 1);
    }
    full_kv = (size_t)LAYERS * DIM * sizeof(float);
    CHECK(wubu_layerskip_forward(x, DIM, LAYERS, zero, -1, full, &saved_zero) == 0);
    CHECK(wubu_layerskip_forward(x, DIM, LAYERS, zero, 6, nodrop, &saved_zero) == 0);
    CHECK(wubu_layerskip_forward(x, DIM, LAYERS, inc, 6, skip, &saved_skip) == 0);

    for (int i = 0; i < DIM; i++) CHECK(fabs(nodrop[i] - full[i]) < 1e-6f);
    printf("EG-60 kv full=%zu saved=%zu (%.2f) parity_nodrop exact\n",
           full_kv, saved_skip, (double)saved_skip / full_kv);
    printf("EG-60 parity_skipped cosine=%.5f\n", cosine(full, skip));
    CHECK(saved_skip > full_kv / 3);
    CHECK(cosine(full, skip) > 0.90);

    CHECK(wubu_layerskip_forward(x, 0, LAYERS, zero, -1, full, &saved_zero) == -1);
    CHECK(wubu_layerskip_forward(x, DIM, LAYERS, zero, LAYERS, full, &saved_zero) == -1);

    if (failures) {
        fprintf(stderr, "test_eg60_layerskip: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg60_layerskip: PASS\n");
    return 0;
}
