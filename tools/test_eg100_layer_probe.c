/* test_eg100_layer_probe.c -- EG-100 gate: measured real-layer probe. */
#include <stdio.h>
#include <math.h>

#include "wubu_layer_probe.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define DIM 16

int main(void) {
    wubu_layer_probe_t *p = wubu_layer_probe_create(8, DIM);
    double rate1 = 0.0, rate128 = 0.0;
    CHECK(p != NULL);

    /* Probe a synthetic layer at both decode and prompt widths. */
    CHECK(wubu_layer_probe_run(p, 1, 2000, &rate1) == 0);
    CHECK(rate1 > 0.0 && isfinite(rate1));
    CHECK(wubu_layer_probe_host_out_n(p) == DIM);
    CHECK(wubu_layer_probe_validate(p, wubu_layer_probe_host_out(p), DIM, 1) == 0);
    CHECK(wubu_layer_probe_run(p, 128, 2000, &rate128) == 0);
    CHECK(rate128 > 0.0 && isfinite(rate128));
    CHECK(wubu_layer_probe_validate(p, wubu_layer_probe_host_out(p), DIM, 128) == 0);
    printf("EG-100 measured rates: decode=%.2f prompt=%.2f (synthetic MB/s)\n",
           rate1, rate128);

    /* NMSE bound depends on width: 2% error fails decode, passes prompt. */
    {
        float dev[DIM];
        double measured;
        for (int i = 0; i < DIM; ++i) dev[i] = wubu_layer_probe_host_out(p)[i] * 1.05f;
        CHECK(wubu_layer_probe_validate(p, dev, DIM, 1) == -1);
        CHECK(wubu_layer_probe_validate(p, dev, DIM, 128) == 0);
        CHECK(wubu_layer_probe_choose(p, dev, DIM, 1, 1.0) == 0);
        CHECK(wubu_layer_probe_choose(p, wubu_layer_probe_host_out(p), DIM, 1, 1.0) == 1);
        CHECK(wubu_layer_probe_choose(p, wubu_layer_probe_host_out(p), DIM, 1, 1e12) == 0);
        printf("EG-100 NMSE 5%%: decode refused, prompt accepted; tier chosen only on match\n");
        CHECK(wubu_layer_probe_run(p, 1, 100, &measured) == 0);
        CHECK(wubu_layer_probe_price(p, 50.0) == measured);
        CHECK(wubu_layer_probe_run(NULL, 1, 100, &measured) == -1);
        CHECK(wubu_layer_probe_run(p, 0, 100, &measured) == -1);
        CHECK(wubu_layer_probe_validate(p, NULL, DIM, 1) == -1);
        CHECK(wubu_layer_probe_create(0, DIM) == NULL);
        CHECK(wubu_layer_probe_create(8, 0) == NULL);
    }

    /* Without a probe, pricing falls back to the host rate. */
    {
        wubu_layer_probe_t *fresh = wubu_layer_probe_create(8, DIM);
        CHECK(fresh != NULL);
        CHECK(wubu_layer_probe_price(fresh, 50.0) == 50.0);
        CHECK(wubu_layer_probe_price(NULL, 50.0) == 50.0);
        wubu_layer_probe_free(fresh);
    }

    wubu_layer_probe_free(p);
    if (failures) { fprintf(stderr, "test_eg100_layer_probe: FAIL (%d)\n", failures); return 1; }
    printf("test_eg100_layer_probe: PASS\n");
    return 0;
}
