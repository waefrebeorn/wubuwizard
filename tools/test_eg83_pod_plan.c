/* test_eg83_pod_plan.c -- EG-83 gate: POD split-K workspace + aux reuse. */
#include <stdio.h>
#include <math.h>

#include "wubu_pod_plan.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_pod_plan_t *p = wubu_pod_plan_create(2, 4, 3);
    float kv[8][6], partial[4][6], out[6], direct[6];
    double maxerr = 0.0;
    CHECK(p != NULL);
    CHECK(wubu_pod_plan_workspace_bytes(p) == 4 * 2 * 3 * sizeof(float));
    printf("EG-83 workspace=%zu bytes heads=2 splits=4 dim=3\n",
           wubu_pod_plan_workspace_bytes(p));

    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 6; ++j) kv[i][j] = 0.125f * (float)((i + j) % 7);
    CHECK(wubu_pod_plan_split(p, &kv[0][0], 8, &partial[0][0]) == 4);
    CHECK(wubu_pod_plan_reduce(p, &partial[0][0], out) == 0);
    for (int j = 0; j < 6; ++j) {
        double v = 0.0;
        for (int i = 0; i < 8; ++i) v += kv[i][j];
        direct[j] = (float)v;
        if (fabs(out[j] - direct[j]) > maxerr) maxerr = fabs(out[j] - direct[j]);
    }
    CHECK(maxerr < 1e-4);
    printf("EG-83 split-K partials=4 maxerr=%.8f\n", maxerr);

    /* One aux allocation is rebound across layers. */
    {
        CHECK(wubu_pod_plan_aux_bytes(p) == 4 * sizeof(unsigned int));
        CHECK(wubu_pod_plan_bind_layer(p, 0) == 0);
        CHECK(wubu_pod_plan_bind_layer(p, 1) == 0);
        CHECK(wubu_pod_plan_bind_layer(p, 2) == 0);
        CHECK(wubu_pod_plan_aux_allocations(p) == 1);
        CHECK(wubu_pod_plan_layers_bound(p) == 3);
        printf("EG-83 aux=%zu bytes, allocations=%d for 3 layers\n",
               wubu_pod_plan_aux_bytes(p), wubu_pod_plan_aux_allocations(p));
        CHECK(wubu_pod_plan_split(p, &kv[0][0], 3, &partial[0][0]) == -1);
        CHECK(wubu_pod_plan_reduce(p, NULL, out) == -1);
        CHECK(wubu_pod_plan_create(0, 4, 3) == NULL);
        CHECK(wubu_pod_plan_create(2, 0, 3) == NULL);
        CHECK(wubu_pod_plan_create(2, 4, 0) == NULL);
        CHECK(wubu_pod_plan_aux_bytes(NULL) == 0);
    }

    wubu_pod_plan_free(p);
    if (failures) { fprintf(stderr, "test_eg83_pod_plan: FAIL (%d)\n", failures); return 1; }
    printf("test_eg83_pod_plan: PASS\n");
    return 0;
}
