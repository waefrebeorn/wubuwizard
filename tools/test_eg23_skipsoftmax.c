/* test_eg23_skipsoftmax.c -- EG-23 gate: calibrated attention block pruning. */
#include <stdio.h>
#include <math.h>

#include "wubu_skip_softmax.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_skip_calib_t *c = wubu_skip_calib_create(2, 0.01);
    double learn0[6] = { -6, -5, -4, 4, 5, 6 };
    double learn1[4] = { 5.0, 5.1, 5.2, 5.3 };
    double q0[5] = { -7, -4, 0, 4, 7 };
    double q1[4] = { 5.0, 5.1, 5.2, 5.3 };
    int keep[8], kept, skipped;
    double worst;

    CHECK(c != NULL);
    if (!c) return 1;
    CHECK(wubu_skip_calib_observe(c, 0, learn0, 6) == 0);
    printf("EG-23 threshold0=%.3f\n", wubu_skip_calib_threshold(c, 0));
    CHECK(fabs(wubu_skip_calib_threshold(c, 0) - (-3.0)) < 1e-9);

    kept = wubu_skip_calib_select(c, 0, q0, 5, keep, 8);
    printf("EG-23 selective kept=%d keep=[%d%d%d%d%d]\n", kept,
           keep[0], keep[1], keep[2], keep[3], keep[4]);
    CHECK(kept == 3 && keep[0] == 0 && keep[1] == 0 && keep[4] == 1);

    /* Comparable-mass blocks: accuracy guard must veto pruning. */
    CHECK(wubu_skip_calib_observe(c, 1, learn1, 4) == 0);
    kept = wubu_skip_calib_select(c, 1, q1, 4, keep, 8);
    printf("EG-23 guarded kept=%d\n", kept);
    CHECK(kept == 4);

    wubu_skip_calib_stats(c, &skipped, &kept, &worst);
    printf("EG-23 stats skipped=%d kept=%d worst_err=%.6g\n",
           skipped, kept, worst);
    CHECK(skipped == 2 && kept == 7);
    CHECK(worst <= 0.01 + 1e-12);

    CHECK(wubu_skip_calib_select(c, 0, q0, 5, keep, 4) == -1);
    CHECK(wubu_skip_calib_observe(c, -1, learn0, 6) == -1);
    CHECK(wubu_skip_calib_create(0, 0.01) == NULL);
    CHECK(wubu_skip_calib_create(1, 0.0) == NULL);

    wubu_skip_calib_free(c);
    if (failures) {
        fprintf(stderr, "test_eg23_skipsoftmax: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg23_skipsoftmax: PASS\n");
    return 0;
}
