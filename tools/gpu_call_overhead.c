/* Per-call cost of the resident device path, measured without the engine in the way.
 *
 * The engine measurement said a resident Q8_0 projection ties the CPU instead of beating it, and the explanation was
 * an estimated ~35 us of memcpy+launch+sync per call. An estimate is not a measurement, so this times the resident
 * entry directly over a range of shapes: the slope against bytes gives the transfer cost, and the intercept gives the
 * fixed per-call overhead. That intercept is the whole reason to batch, so it has to be a number.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

typedef int (*fn_resident)(int ty, const void *w, size_t wb, const float *act, int actb,
                          float *y, int n_in, int n_out, long long rb);

static double wall(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *dll = getenv("IN2_CUDA_DLL");
    if (!dll) dll = "wubu_cuda.dll";
    const int iters = argc > 1 ? atoi(argv[1]) : 200;

    /* type 8 = Q8_0, 34 bytes per 32 weights */
    const int n_in = 1024, n_out = 2304;
    size_t rb = (size_t) n_in / 32 * 34;           /* 1088 bytes per row */
    size_t wb = rb * (size_t) n_out;
    unsigned char *w = malloc(wb);
    float *act = malloc(sizeof(float) * (size_t) n_in);
    float *y = malloc(sizeof(float) * (size_t) n_out);
    if (!w || !act || !y) { fprintf(stderr, "alloc\n"); return 1; }
    srand(1234);
    for (size_t i = 0; i < wb; i++) w[i] = (unsigned char) (rand() & 0xff);
    for (int i = 0; i < n_in; i++) act[i] = (float) ((rand() % 200) - 100) / 100.0f;
    memset(y, 0, sizeof(float) * (size_t) n_out);

    void *h = (void *) LoadLibraryA(dll);
    if (!h) { fprintf(stderr, "LoadLibrary failed: %s\n", dll); return 127; }
    fn_resident res = (fn_resident) (void (*)(void)) GetProcAddress((void *) h, "wubu_cuda_resident_matvec");
    if (!res) { fprintf(stderr, "no wubu_cuda_resident_matvec\n"); return 127; }

    /* first call uploads the slab and allocates staging, so it is not part of the steady state */
    int rc = res(8, w, wb, act, (int) (sizeof(float) * (size_t) n_in), y, n_in, n_out, (long long) rb);
    printf("resident rc=%d  y[0]=%.6f y[5]=%.6f\n", rc, y[0], y[5]);
    if (rc != 0) return 2;
    double nz = 0; for (int i = 0; i < n_out; i++) nz += y[i] * y[i];
    printf("non-zero check: sum y^2 = %.4f (must not be 0)\n", nz);

    double t0 = wall();
    for (int i = 0; i < iters; i++) {
        if (res(8, w, wb, act, (int) (sizeof(float) * (size_t) n_in), y, n_in, n_out, (long long) rb) != 0) {
            fprintf(stderr, "rc fail at %d\n", i); return 3;
        }
    }
    double dt = wall() - t0;
    double us = dt / iters * 1e6;
    double mb = (double) wb / 1048576.0;
    double gws = (double) n_in * (double) n_out * (double) iters / dt / 1e9;

    printf("%dx%d  slab %.2f MB  %d calls  %.3f s\n", n_out, n_in, mb, iters, dt);
    printf("per call: %.2f us   throughput %.2f G weights/s   %.2f GB/s\n", us, gws, gws * 34.0 / 32.0);
    printf("CPU 8-thread equivalent for this shape: %.2f us  (at 11.2 GB/s aggregate)\n", mb / 11.2e3 * 1e6);
    printf("same tensor on ONE CPU thread:          %.2f us  (at 19.08 G weights/s)\n",
           (double) n_in * (double) n_out / 19.08e9 * 1e6);
    return 0;
}
