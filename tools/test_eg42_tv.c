/* test_eg42_tv.c -- EG-42 gate: TV identity and acceptance calibration. */
#include <stdio.h>
#include <math.h>

#include "wubu_tv_accept.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    float p[3] = { 0.5f, 0.3f, 0.2f };
    float q[3] = { 0.4f, 0.4f, 0.2f };
    float same[3] = { 0.5f, 0.3f, 0.2f };
    float one[3] = { 1.0f, 0.0f, 0.0f };
    float other[3] = { 0.0f, 1.0f, 0.0f };
    float a, tv, empirical;
    float bad[3] = { 0.5f, 0.5f, 0.5f };

    a = wubu_tv_accept_prob(p, q, 3);
    tv = wubu_tv_distance(p, q, 3);
    printf("EG-42 accept=%.4f tv=%.4f identity=%.4f\n", a, tv, 1.0f - tv);
    CHECK(fabsf(a - 0.9f) < 1e-5f && fabsf(tv - 0.1f) < 1e-5f);
    CHECK(fabsf(a - (1.0f - tv)) < 1e-5f);

    CHECK(fabsf(wubu_tv_accept_prob(same, same, 3) - 1.0f) < 1e-5f);
    CHECK(fabsf(wubu_tv_distance(same, same, 3)) < 1e-5f);
    CHECK(wubu_tv_accept_prob(one, other, 3) == 0.0f);
    CHECK(wubu_tv_distance(one, other, 3) == 1.0f);

    CHECK(wubu_tv_calibrate(p, q, 3, 12345u, 20000, &empirical) == 0);
    printf("EG-42 calibrated=%.4f target=%.4f\n", empirical, a);
    CHECK(fabsf(empirical - a) < 0.02f);

    CHECK(wubu_tv_accept_prob(bad, q, 3) == -1.0f);
    CHECK(wubu_tv_accept_prob(NULL, q, 3) == -1.0f);
    CHECK(wubu_tv_accept_prob(p, q, 0) == -1.0f);
    CHECK(wubu_tv_calibrate(p, q, 3, 1u, 0, &empirical) == -1);
    CHECK(wubu_tv_calibrate(p, q, 3, 1u, 10, NULL) == -1);

    if (failures) {
        fprintf(stderr, "test_eg42_tv: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg42_tv: PASS\n");
    return 0;
}
