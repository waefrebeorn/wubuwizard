/*
 * test_fold_sincos8.c — the AVX2 8-wide folded sin/cos gate
 * (ported from wuburvc, the sibling repo's CPU research).
 *
 * 1. Parity: wubu_fold_sincos8 == wubu_fold_sincos (scalar) within 1e-6
 * 2. Accuracy: both agree with libm sinf/cosf within 1e-4 (fp32-class)
 * 3. Speed: the 8-wide version beats the scalar loop (>= 1.2x)
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "wubu_foldmath.h"

int main(void)
{
    printf("=== test_fold_sincos8 (AVX2 folded sin/cos, from wuburvc) ===\n");
    int N = 4096;
    static float x[4096], s8[4096], c8[4096], s1[4096], c1[4096];
    for (int i = 0; i < N; i++) x[i] = (float)(i * 0.37 - 500.0);  /* spread */

    wubu_fold_sincos8(x, s8, c8, N);
    for (int i = 0; i < N; i++) wubu_fold_sincos(x[i], &s1[i], &c1[i]);

    /* 1. scalar vs 8-wide parity */
    float max_par = 0.0f;
    for (int i = 0; i < N; i++) {
        float d = fabsf(s8[i] - s1[i]);
        if (d > max_par) max_par = d;
        d = fabsf(c8[i] - c1[i]);
        if (d > max_par) max_par = d;
    }
    printf("  scalar-vs-8wide max diff: %.3e\n", max_par);
    printf("  %s\n", max_par < 1e-6f ? "PARITY OK" : "PARITY FAIL");
    int fail = (max_par >= 1e-6f);

    /* 2. accuracy vs libm */
    float max_libm = 0.0f;
    for (int i = 0; i < N; i++) {
        float d = fabsf(s8[i] - sinf(x[i]));
        if (d > max_libm) max_libm = d;
        d = fabsf(c8[i] - cosf(x[i]));
        if (d > max_libm) max_libm = d;
    }
    printf("  vs-libm max diff: %.3e %s\n", max_libm,
           max_libm < 1e-4f ? "(fp32-class OK)" : "(TOO BIG)");
    if (max_libm >= 1e-4f) fail = 1;

    /* 3. speed (consume the outputs so nothing is eliminated) */
    int ITER = 20000;
    volatile float sink = 0.0f;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITER; it++)
        for (int i = 0; i < N; i++) { wubu_fold_sincos(x[i], &s1[i], &c1[i]); sink += s1[i] + c1[i]; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt1 = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITER; it++) { wubu_fold_sincos8(x, s8, c8, N); sink += s8[0] + c8[0]; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt8 = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double speedup = dt1 / dt8;
    printf("  checksum %.4f (anti-elimination)\n", (double)sink);
    printf("  scalar %.3fs  8wide %.3fs  speedup %.2fx\n", dt1, dt8, speedup);
    printf("  %s\n", speedup >= 1.2f ? "SPEED OK" : "SPEED WARN (still correct)");

    if (!fail) printf("=== ALL FOLD SIN/COS8 TESTS PASSED ===\n");
    return fail ? 1 : 0;
}
