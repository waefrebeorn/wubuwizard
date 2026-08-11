/* sd_gemm_probe.c — isolate wubu_sd_matmul_nt_f16 throughput on the CM4.
 * Measures the GEMM alone (no im2col) for the upsample conv shapes:
 *   up.1.upsample: M=2048 K=2304 N=256   (per tile, T=4)
 *   up.2.upsample: M=1024 K=4608 N=512
 *   up.0.block:    M=2048 K=1152 N=128
 * Prints GFLOP/s. */
#include "wubu_sd_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void bench(int M, int K, int N, const char *name) {
    uint16_t *x = (uint16_t *)aligned_alloc(64, (size_t)M * K * 2);
    float *W = (float *)aligned_alloc(64, (size_t)N * K * 4);
    float *y = (float *)aligned_alloc(64, (size_t)M * N * 4);
    if (!x || !W || !y) { printf("alloc fail %s\n", name); return; }
    srand(42);
    for (int i = 0; i < M * K; i++) x[i] = (uint16_t)(rand() & 0xFFFF);
    for (int i = 0; i < N * K; i++) W[i] = ((float)(rand() % 2000) - 1000) / 100.0f;
    /* warmup */
    wubu_sd_matmul_nt_f16(x, W, M, K, N, y);
    double t0 = now_s();
    wubu_sd_matmul_nt_f16(x, W, M, K, N, y);
    double dt = now_s() - t0;
    double gflops = 2.0 * (double)M * K * N / dt / 1e9;
    /* checksum to defeat DCE */
    volatile float s = 0; for (int i = 0; i < M * N; i += 97) s += y[i];
    printf("%-22s M=%-5d K=%-5d N=%-4d  %8.3fs  %7.2f GFLOP/s (ck=%.3f)\n",
           name, M, K, N, dt, gflops, s);
    free(x); free(W); free(y);
}

int main(void) {
    bench(2048, 2304, 256, "up.1.upsample tile");
    bench(1024, 4608, 512, "up.2.upsample tile");
    bench(2048, 1152, 128, "up.0.block tile");
    bench(512, 2304, 256, "half-size");
    bench(256, 2304, 256, "quarter-size");
    return 0;
}
