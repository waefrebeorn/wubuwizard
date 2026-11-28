/* test_gemm_m_probe.c - minimal reproducer: does wubu_cuda_resident_f16_gemm compute Y[m][o]=sum_k X[m][k]*W[o][k]
 * for M>1? Uses W = identity-ish (w[o][k] = (o==k)?1:0) so EXPECTED Y[m][o] = X[m][o] exactly - any deviation is a
 * layout/batch bug, not rounding. Prints the matrices so the failure MODE is visible (transposed / row0 only / garbage).
 * usage: test_gemm_m_probe.exe <dll> <M> <K> <N>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
typedef int (*one_t)(long long, const void *, unsigned long long, const float *, int, int, float *, int);
int main(int argc, char **argv) {
    const char *dll = (argc > 1) ? argv[1] : "wubu_cuda.dll";
    int M = (argc > 2) ? atoi(argv[2]) : 2, K = (argc > 3) ? atoi(argv[3]) : 8, N = (argc > 4) ? atoi(argv[4]) : 8;
    HMODULE h = LoadLibraryA(dll);
    if (!h) { printf("load fail %lu\n", (unsigned long) GetLastError()); return 3; }
    one_t one = (one_t)(void *) GetProcAddress(h, "wubu_cuda_resident_f16_gemm");
    if (!one) { printf("no export\n"); return 4; }
    /* W: [N][K] f16, w[o][k] = (o==k) ? 1 : 0  (needs N<=K) */
    unsigned short *w = (unsigned short *) calloc((size_t) N * K, 2);
    for (int o = 0; o < N && o < K; ++o) w[(size_t) o * K + o] = 0x3C00; /* 1.0h */
    float *x = (float *) malloc((size_t) M * K * 4);
    for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) x[(size_t) m * K + k] = (float) (m * 100 + k);
    float *y = (float *) malloc((size_t) M * N * 4);
    for (long long t = 0; t < (long long) M * N; ++t) y[t] = -999.0f;
    int rc = one(777, w, (unsigned long long) N * K * 2, x, M, K, y, N);
    printf("M=%d K=%d N=%d  rc=%d   (expected Y[m][o] = X[m][o] = m*100+o)\n", M, K, N, rc);
    for (int m = 0; m < M; ++m) {
        printf("  row %d exp:", m);
        for (int o = 0; o < N; ++o) printf(" %6.1f", (o < K) ? (float) (m * 100 + o) : -1.0f);
        printf("\n  row %d got:", m);
        for (int o = 0; o < N; ++o) printf(" %6.1f", y[(size_t) m * N + o]);
        printf("\n");
    }
    return 0;
}
