/* test_eg51_spice.c -- EG-51 gate: confidence routing + accuracy-gated surrogate. */
#include <stdio.h>
#include <math.h>

#include "wubu_spice.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define R 8
#define C 6

int main(void) {
    float M[R*C], x[C], approx;
    wubu_spice_surrogate_t s3, s1;
    const float conf[4] = { 0.95f, 0.4f, 0.2f, 0.8f };
    int act[4];
    double ne = 0.0, na = 0.0;

    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            M[i*C+j] = (float)((i+1) * (j+1) + 0.5 * (i-j) + 0.25 * (i*j % 3));
    for (int j = 0; j < C; j++) x[j] = cosf(0.4f * (float)j);

    CHECK(wubu_spice_surrogate_build(M, R, C, 3, &s3) == 0);
    CHECK(wubu_spice_surrogate_build(M, R, C, 1, &s1) == 0);
    printf("EG-51 relerr rank3=%.5f rank1=%.5f\n", s3.relerr, s1.relerr);
    CHECK(s3.relerr < s1.relerr);

    CHECK(wubu_spice_route(conf, 4, s3.relerr, 0.05f, 0.7f, act) == 0);
    printf("EG-51 rank3 actions=[%d %d %d %d]\n", act[0], act[1], act[2], act[3]);
    CHECK(act[0] == WUBU_SPICE_LOOKAHEAD && act[1] == WUBU_SPICE_SURROGATE &&
          act[2] == WUBU_SPICE_SURROGATE && act[3] == WUBU_SPICE_LOOKAHEAD);

    CHECK(wubu_spice_route(conf, 4, s1.relerr, s3.relerr * 0.5f, 0.7f, act) == 0);
    printf("EG-51 accuracy opt-out actions=[%d %d %d %d]\n",
           act[0], act[1], act[2], act[3]);
    CHECK(act[1] == WUBU_SPICE_EXACT && act[2] == WUBU_SPICE_EXACT);

    CHECK(wubu_spice_surrogate_apply(&s3, 2, x, &approx) == 0);
    for (int j = 0; j < C; j++) ne += fabs((double)M[2*C+j] * x[j]);
    {
        double exact = 0.0;
        for (int k = 0; k < C; k++) exact += (double)M[2*C+k] * x[k];
        na = fabs(exact - approx);
    }
    printf("EG-51 surrogate row2 dot relerr=%.5f\n", na / ne);
    CHECK(na / ne < 0.05);

    CHECK(wubu_spice_route(NULL, 4, 0.1f, 0.2f, 0.7f, act) == -1);
    CHECK(wubu_spice_surrogate_build(M, 0, C, 1, &s1) == -1);
    CHECK(wubu_spice_surrogate_apply(&s3, -1, x, &approx) == -1);
    wubu_spice_surrogate_free(&s3);
    wubu_spice_surrogate_free(&s1);

    if (failures) {
        fprintf(stderr, "test_eg51_spice: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg51_spice: PASS\n");
    return 0;
}
