/* test_cuda_bridge.c -- gate: MinGW engine can load and call the nvcc-built DLL.
 *
 * The engine links with MinGW ld, which cannot consume nvcc/cudart MSVC-ABI
 * objects. The GPU kernels therefore live in wubu_cuda.dll and are reached with
 * LoadLibrary + GetProcAddress over extern "C" symbols. This gate proves the
 * bridge and checks one real kernel against a CPU reference.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

typedef int  (*fn_avail_t)(void);
typedef int  (*fn_name_t)(char *, int);
typedef int  (*fn_sgemv_host_t)(const float *, const float *, float *, int, int);

int main(int argc, char **argv)
{
    const char *dll = argc > 1 ? argv[1] : "wubu_cuda.dll";
    HMODULE h = LoadLibraryA(dll);
    if (!h) {
        fprintf(stderr, "test_cuda_bridge: could not load %s (err %lu)\n", dll, GetLastError());
        return 2;
    }

    fn_avail_t avail = (fn_avail_t)(void *)GetProcAddress(h, "wubu_cuda_available");
    fn_name_t  name  = (fn_name_t)(void *)GetProcAddress(h, "wubu_cuda_device_name");
    fn_sgemv_host_t sgemv = (fn_sgemv_host_t)(void *)GetProcAddress(h, "wubu_cuda_sgemv_host");
    CHECK(avail != NULL);
    CHECK(name != NULL);
    CHECK(sgemv != NULL);
    if (!avail || !name || !sgemv) { FreeLibrary(h); return 1; }

    char dev[256] = {0};
    CHECK(name(dev, (int)sizeof(dev)) == 0);
    printf("EG-112 bridge: device = %s\n", dev);
    if (!avail()) {
        printf("test_cuda_bridge: SKIP (no CUDA device visible)\n");
        FreeLibrary(h);
        return 0;
    }

    /* 64 x 96 GEMV against a CPU reference. */
    enum { ROWS = 64, COLS = 96 };
    static float w[ROWS * COLS], x[COLS], y[ROWS], ref[ROWS];
    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < COLS; ++c)
            w[r * COLS + c] = (float)((r * 31 + c * 17) % 23) * 0.03125f - 0.25f;
    for (int c = 0; c < COLS; ++c)
        x[c] = (float)(c % 11) * 0.125f - 0.5f;

    for (int r = 0; r < ROWS; ++r) {
        double acc = 0.0;
        for (int c = 0; c < COLS; ++c) acc += (double)w[r * COLS + c] * x[c];
        ref[r] = (float)acc;
    }

    const int rc = sgemv(w, x, y, ROWS, COLS);
    CHECK(rc == 0);
    float worst = 0.0f;
    for (int r = 0; r < ROWS; ++r) {
        const float d = fabsf(y[r] - ref[r]);
        if (d > worst) worst = d;
    }
    CHECK(worst < 1e-3f);
    printf("EG-112 bridge: gemv worst_abs=%.3g\n", worst);

    FreeLibrary(h);
    if (failures) { fprintf(stderr, "test_cuda_bridge: FAIL (%d)\n", failures); return 1; }
    printf("test_cuda_bridge: PASS\n");
    return 0;
}
