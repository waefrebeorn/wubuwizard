/* sd_gemm_probe2.c — JC sensitivity + vcvt cost for the F16 GEMM.
 * Measures wubu_sd_matmul_nt_f16 across JC values by calling with
 * different N shapes (N=64/128/256) and compares against the F32-x
 * kernel wubu_sd_matmul_nt (no vcvt) at the same shape. */
#include "wubu_sd_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void bench_f16(int M, int K, int N, const char *name) {
    uint16_t *x = (uint16_t *)aligned_alloc(64, (size_t)M * K * 2);
    float *W = (float *)aligned_alloc(64, (size_t)N * K * 4);
    float *y = (float *)aligned_alloc(64, (size_t)M * N * 4);
    for (int i = 0; i < M * K; i++) x[i] = (uint16_t)0x3C00; /* 1.0 */
    for (int i = 0; i < N * K; i++) W[i] = 0.001f;
    wubu_sd_matmul_nt_f16(x, W, M, K, N, y);
    double t0 = now_s();
    wubu_sd_matmul_nt_f16(x, W, M, K, N, y);
    double dt = now_s() - t0;
    volatile float s = 0; for (int i = 0; i < M * N; i += 97) s += y[i];
    printf("%-24s M=%-5d K=%-5d N=%-4d  %8.4fs  %7.2f GFLOP/s (ck=%.3f)\n",
           name, M, K, N, dt, 2.0 * (double)M * K * N / dt / 1e9, s);
    free(x); free(W); free(y);
}

static void bench_f32(int M, int K, int N, const char *name) {
    float *x = (float *)aligned_alloc(64, (size_t)M * K * 4);
    float *W = (float *)aligned_alloc(64, (size_t)N * K * 4);
    float *y = (float *)aligned_alloc(64, (size_t)M * N * 4);
    for (int i = 0; i < M * K; i++) x[i] = 1.0f;
    for (int i = 0; i < N * K; i++) W[i] = 0.001f;
    wubu_sd_matmul_nt(x, W, M, K, N, y);
    double t0 = now_s();
    wubu_sd_matmul_nt(x, W, M, K, N, y);
    double dt = now_s() - t0;
    volatile float s = 0; for (int i = 0; i < M * N; i += 97) s += y[i];
    printf("%-24s M=%-5d K=%-5d N=%-4d  %8.4fs  %7.2f GFLOP/s (ck=%.3f)\n",
           name, M, K, N, dt, 2.0 * (double)M * K * N / dt / 1e9, s);
    free(x); free(W); free(y);
}

int main(void) {
    /* JC for K=2304: 590KB/(4*2304*4)=16 cols... so N=64→JC=16? check */
    bench_f16(2048, 2304, 256, "f16 up.1 (N=256)");
    bench_f16(2048, 2304, 128, "f16 N=128");
    bench_f16(2048, 2304, 64,  "f16 N=64");
    bench_f16(2048, 2304, 512, "f16 N=512");
    bench_f32(2048, 2304, 256, "f32 up.1 (no vcvt)");
    bench_f32(2048, 2304, 64,  "f32 N=64");
    return 0;
}
