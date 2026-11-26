/* iq2x8_check.c - calls the DLL's parity gate for the 8-column IQ2_XXS kernel.
 * Loads wubu_cuda.dll dynamically so it links against nothing at build time. */
#include <stdio.h>
#include <windows.h>

typedef int (*check_fn)(int, int, int, double *, double *);
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
    if (argc < 6) { printf("usage: iq2x8_check K N M grid.bin ksigns.bin\n"); return 1; }
    const int K = (argc > 1) ? atoi(argv[1]) : 2304;
    const int N = (argc > 2) ? atoi(argv[2]) : 2304;
    const int M = (argc > 3) ? atoi(argv[3]) : 8;

    HMODULE h = LoadLibraryA("wubu_cuda.dll");
    if (!h) { printf("LoadLibrary failed (%lu)\n", (unsigned long) GetLastError()); return 2; }
    check_fn fn = (check_fn) GetProcAddress(h, "wubu_cuda_iq2x8_check");
    if (!fn) { printf("symbol not found\n"); FreeLibrary(h); return 3; }

    /* The IQ2 grid and signs live in __constant__ memory and are EMPTY until uploaded, so without this both kernels
     * compute with all-zero weights and the comparison reports 0 == 0, which proves nothing. */
    upl_fn up = (upl_fn) GetProcAddress(h, "wubu_cuda_upload_iq2xxs");
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

    double md = -1.0, mr = -1.0;
    const int rc = fn(K, N, M, &md, &mr);
    printf("rc=%d  K=%d n_cols=%d M=%d\n", rc, K, N, M);
    printf("  mrow vs 8col:  max|diff|=%.3e   max|ref|=%.3e   rel=%.3e\n", md, mr, mr > 0 ? md / mr : -1.0);
    if (mr <= 0.0) printf("  WARNING: reference magnitude is zero - this comparison proves nothing.\n");
    else if (md == 0.0) printf("  BIT-IDENTICAL\n");
    FreeLibrary(h);
    return rc;
}
