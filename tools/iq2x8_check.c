/* iq2x8_check.c - parity and speed gate for the 8-column IQ2_XXS kernel.
 *
 * Loads wubu_cuda.dll dynamically so it links against nothing at build time.
 *
 * Two traps this tool exists to avoid, both of which it fell into once:
 *  - the IQ2 grid and k-sign tables live in __constant__ memory and are EMPTY until uploaded, so without an explicit
 *    upload both kernels compute with zero weights and the comparison reports 0 == 0, which proves nothing;
 *  - a zero reference magnitude must never be printed as a pass.
 *
 * usage: iq2x8_check K n_cols M grid.bin ksigns.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

typedef int (*check_fn)(int, int, int, double *, double *, double *, double *);
typedef int (*upl_fn)(const void *, const void *);

static long slurp(const char *path, void *dst, long want)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    const long n = (long) fread(dst, 1, (size_t) want, f);
    fclose(f);
    return n;
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        printf("usage: iq2x8_check K n_cols M grid.bin ksigns.bin\n");
        return 1;
    }
    const int K = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int M = atoi(argv[3]);

    HMODULE h = LoadLibraryA("wubu_cuda.dll");
    if (!h) { printf("LoadLibrary failed (%lu)\n", (unsigned long) GetLastError()); return 2; }

    check_fn fn = (check_fn) GetProcAddress(h, "wubu_cuda_iq2x8_check");
    upl_fn   up = (upl_fn)   GetProcAddress(h, "wubu_cuda_upload_iq2xxs");
    if (!fn) { printf("check symbol not found\n"); FreeLibrary(h); return 3; }
    if (!up) { printf("upload symbol not found\n"); FreeLibrary(h); return 4; }

    unsigned long long grid[256];
    unsigned char signs[128];
    if (slurp(argv[4], grid, sizeof grid) != (long) sizeof grid ||
        slurp(argv[5], signs, sizeof signs) != (long) sizeof signs) {
        printf("could not read grid/ksigns bins\n"); FreeLibrary(h); return 5;
    }
    const int urc = up(grid, signs);
    printf("upload rc=%d\n", urc);
    if (urc != 0) { FreeLibrary(h); return 6; }

    double md = -1.0, mr = -1.0, ms_m = -1.0, ms_8 = -1.0;
    const int rc = fn(K, N, M, &md, &mr, &ms_m, &ms_8);
    printf("rc=%d  K=%d n_cols=%d M=%d\n", rc, N ? K : K, N, M);

    const double mac = (double) M * (double) K * (double) N;
    if (ms_m > 0.0)
        printf("  mrow  %8.2f ms   %7.2f GMAC/s\n", ms_m, mac / (ms_m * 1e-3) / 1e9);
    if (ms_8 > 0.0)
        printf("  8col  %8.2f ms   %7.2f GMAC/s   speedup %.2fx\n", ms_8, mac / (ms_8 * 1e-3) / 1e9, ms_m / ms_8);

    printf("  mrow vs 8col: max|diff|=%.3e  max|ref|=%.3e  rel=%.3e\n", md, mr, mr > 0.0 ? md / mr : -1.0);
    if (mr <= 0.0)
        printf("  WARNING: reference magnitude is zero - this comparison proves nothing.\n");
    else if (md == 0.0)
        printf("  BIT-IDENTICAL\n");

    FreeLibrary(h);
    return rc;
}
