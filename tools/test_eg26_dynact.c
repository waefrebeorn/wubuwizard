/* test_eg26_dynact.c -- EG-26 gate: per-token activation quantization. */
#include <stdio.h>
#include <math.h>

#include "wubu_dynact.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    float loud[8] = { 8.0f, -4.0f, 2.0f, -1.0f, 0.0f, 0.5f, -0.25f, 1.0f };
    float quiet[8] = { 0.2f, -0.1f, 0.05f, 0.15f, -0.2f, 0.1f, 0.05f, -0.15f };
    unsigned char q[8];
    float deq[8], s1, s2, err1, err2, static_err, static_scale;
    int i;

    CHECK(wubu_dynact_quant_row(loud, 8, 4, q, &s1) == 0);
    CHECK(wubu_dynact_dequant_row(q, 8, 4, s1, deq) == 0);
    err1 = wubu_dynact_maxerr(loud, deq, 8);
    printf("EG-26 loud scale=%.4f err=%.5f\n", s1, err1);
    CHECK(err1 <= s1 / 2.0f + 1e-6f);

    CHECK(wubu_dynact_quant_row(quiet, 8, 4, q, &s2) == 0);
    CHECK(wubu_dynact_dequant_row(q, 8, 4, s2, deq) == 0);
    err2 = wubu_dynact_maxerr(quiet, deq, 8);
    printf("EG-26 quiet scale=%.4f err=%.5f\n", s2, err2);
    CHECK(err2 <= s2 / 2.0f + 1e-6f);
    CHECK(s2 < s1 / 4.0f);

    /* A shared scale taken from the loud token crushes the quiet token. */
    static_scale = s1;
    for (i = 0; i < 8; i++) {
        int v = (int)roundf(quiet[i] / static_scale);
        deq[i] = (float)v * static_scale;
    }
    static_err = wubu_dynact_maxerr(quiet, deq, 8);
    printf("EG-26 static-scale err=%.5f (dynamic %.5f)\n", static_err, err2);
    CHECK(static_err > err2 * 5.0f);

    CHECK(wubu_dynact_quant_row(NULL, 8, 4, q, &s1) == -1);
    CHECK(wubu_dynact_quant_row(loud, 0, 4, q, &s1) == -1);
    CHECK(wubu_dynact_quant_row(loud, 8, 1, q, &s1) == -1);
    CHECK(wubu_dynact_dequant_row(q, 8, 4, 0.0f, deq) == -1);
    CHECK(wubu_dynact_maxerr(loud, deq, 0) == -1.0f);

    if (failures) {
        fprintf(stderr, "test_eg26_dynact: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg26_dynact: PASS\n");
    return 0;
}
