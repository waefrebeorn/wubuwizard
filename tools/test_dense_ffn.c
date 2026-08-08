/* test_dense_ffn.c — dense SwiGLU FFN tests (C11, no deps). */
#include "wubu_dense_ffn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0, tests_pass = 0;
static void check(int cond, const char *msg) {
    tests_run++;
    if (cond) { tests_pass++; printf("  PASS: %s\n", msg); }
    else      { printf("  FAIL: %s\n", msg); }
}

/* Tiny F32 weights: gate = identity-ish (1 on diag), up = ones,
 * down = identity. y[i] should be ~ SiLU(x[i]) * sum(x) ... we
 * just verify the pipeline runs and returns finite values with
 * the right shape. */
int main(void)
{
    printf("=== dense FFN ===\n");

    /* NULL safety */
    check(wubu_dense_ffn_create(NULL, 0, NULL, 0, NULL, 0, 0, 0) == NULL,
          "null create rejected");
    check(wubu_dense_ffn_ready(NULL) == 0, "null not ready");

    /* d=4, ff=8. F32 (type 0) weights laid out row-major [ff][d]. */
    int d = 4, ff = 8;
    float gate_w[8 * 4], up_w[8 * 4], down_w[4 * 8];
    for (int i = 0; i < 8 * 4; i++) { gate_w[i] = 0.5f; up_w[i] = 2.0f; }
    for (int i = 0; i < 4 * 8; i++) down_w[i] = 0.25f;

    wubu_dense_ffn *f = wubu_dense_ffn_create(
        (const uint8_t *)gate_w, 0, (const uint8_t *)up_w, 0,
        (const uint8_t *)down_w, 0, d, ff);
    check(f != NULL, "create");
    check(wubu_dense_ffn_ready(f) == 1, "ready");

    float x[4] = { 1.0f, -1.0f, 2.0f, 0.5f };
    float y[4];
    wubu_dense_ffn_forward(f, x, y);
    check(isfinite(y[0]) && isfinite(y[1]) && isfinite(y[2]) && isfinite(y[3]),
          "output finite");

    /* symmetry check: forward(x) should equal -forward(-x) for
     * sign-symmetric weights (all rows equal -> sum over rows is
     * a fixed gain; SiLU(g) * g is even-ish... verify monotone
     * scaling instead: ||y|| grows with ||x||. */
    float x2[4] = { 2.0f, -2.0f, 4.0f, 1.0f };
    float y2[4];
    wubu_dense_ffn_forward(f, x2, y2);
    float n1 = 0, n2 = 0;
    for (int i = 0; i < 4; i++) { n1 += y[i] * y[i]; n2 += y2[i] * y2[i]; }
    check(n2 > n1, "larger input -> larger output");

    /* NULL safety on forward */
    wubu_dense_ffn_forward(NULL, x, y); /* no crash */
    wubu_dense_ffn_forward(f, NULL, y);
    wubu_dense_ffn_forward(f, x, NULL);

    wubu_dense_ffn_free(f);
    printf("=== Results: %d/%d passed ===\n", tests_pass, tests_run);
    return tests_pass == tests_run ? 0 : 1;
}
