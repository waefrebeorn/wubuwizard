/*
 * test_gemv_dispatch.c — the PER-FAMILY GEMV DISPATCH gate (AN47 #2:
 * "diagnose-driven precision changes are not paper-only").
 *
 * Asserts:
 *   1. the ladder picks the real backend: dense (8 bits) -> INT8,
 *      norms (32 bits) -> F32/BF16
 *   2. the dispatch EXECUTES the change: re-laddering dense 8->32
 *      switches the next dense GEMV to the high-bit path
 *   3. the INT8 path is numerically close to the F32 reference
 *      (the quantized GEMV is correct, not a stub)
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "wubu_gemv_dispatch.h"
#include "wubu_precision_plan.h"
#include "wubu_gemm.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_gemv_dispatch (the ladder EXECUTES) ===\n");

    /* the Escha ladder: gate/up 2, down 3, dense 8, norms 32 */
    wubu_precision_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.bits[WUBU_FAM_GATE_UP] = 2;
    plan.bits[WUBU_FAM_DOWN] = 3;
    plan.bits[WUBU_FAM_DENSE] = 8;
    plan.bits[WUBU_FAM_NORM] = 32;

    wubu_gemv_dispatch_t d;
    if (wubu_disp_init(&d, &plan) != 0) FAIL("disp init");

    /* a small dense matrix + input */
    const int M = 64, K = 32;
    float *W = (float *)malloc((size_t)M * K * sizeof(float));
    float *x = (float *)malloc((size_t)K * sizeof(float));
    float *y = (float *)malloc((size_t)M * sizeof(float));
    float *y_ref = (float *)malloc((size_t)M * sizeof(float));
    for (int i = 0; i < M * K; i++) W[i] = ((float)(i % 7) - 3.0f) / 4.0f;
    for (int i = 0; i < K; i++) x[i] = ((float)(i % 5) - 2.0f) / 2.0f;

    /* 1. the ladder picks the real backend */
    wubu_disp_backend_t be = wubu_disp_gemv(&d, WUBU_FAM_DENSE, W, x, y, M, K);
    printf("  dense (8 bits) -> backend %d (1 = INT8)\n", (int)be);
    if (be != WUBU_DISP_I8) FAIL("the 8-bit family did not dispatch to INT8");

    be = wubu_disp_gemv(&d, WUBU_FAM_NORM, W, x, y, M, K);
    printf("  norms (32 bits) -> backend %d (0 = F32, 2 = BF16)\n", (int)be);
    if (be != WUBU_DISP_F32 && be != WUBU_DISP_BF16)
        FAIL("the 32-bit family did not dispatch to the high-bit path");

    /* 2. the dispatch EXECUTES the change: dense 8 -> 32 */
    wubu_precision_plan_t plan2 = plan;
    plan2.bits[WUBU_FAM_DENSE] = 32;
    wubu_disp_reladder(&d, &plan2);
    be = wubu_disp_gemv(&d, WUBU_FAM_DENSE, W, x, y, M, K);
    printf("  dense AFTER reladder (8->32) -> backend %d (0/2 = high bits)\n", (int)be);
    if (be != WUBU_DISP_F32 && be != WUBU_DISP_BF16)
        FAIL("the reladder did not switch the dense family to high bits");

    /* 3. the INT8 path is numerically close to the F32 reference */
    wubu_gemv_f32(W, x, y_ref, M, K);
    wubu_disp_reladder(&d, &plan);   /* back to the Escha ladder */
    wubu_disp_gemv(&d, WUBU_FAM_DENSE, W, x, y, M, K);
    double err = 0, norm = 0;
    for (int i = 0; i < M; i++) {
        double e = fabs((double)y[i] - (double)y_ref[i]);
        err += e * e;
        norm += (double)y_ref[i] * (double)y_ref[i];
    }
    double rel = norm > 0 ? sqrt(err / norm) : 0.0;
    printf("  INT8 vs F32 relative error: %.4f (must be small)\n", rel);
    if (rel > 0.1) FAIL("the INT8 path is not numerically close");

    char stats[256];
    wubu_disp_stats(&d, stats, sizeof(stats));
    printf("  stats: %s\n", stats);

    wubu_disp_free(&d);
    free(W); free(x); free(y); free(y_ref);
    printf("=== ALL GEMV DISPATCH TESTS PASSED (the ladder runs) ===\n");
    return 0;
}
