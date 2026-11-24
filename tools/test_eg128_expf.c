/* test_eg128_expf.c -- EG-128: device expf vs the C library's, bit for bit.
 *
 * The fused attention path still differs from the CPU by ~3.5e-07 after matching every
 * reduction order. The softmax exponentials are the last operation whose implementation is
 * chosen by the toolchain rather than by this codebase, so measure it instead of assuming. */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <windows.h>

typedef int (*fn_t)(const float *, float *, int);

int main(int argc, char **argv)
{
    HMODULE h = LoadLibraryA(argc > 1 ? argv[1] : "wubu_cuda.dll");
    if (!h) { printf("EG-128: SKIP (no dll)\n"); return 0; }
    fn_t probe = (fn_t)(void *)GetProcAddress(h, "wubu_cuda_expf_probe");
    if (!probe) { printf("EG-128: SKIP (no export)\n"); FreeLibrary(h); return 0; }

    /* score - max is in (-inf, 0]; sample the range the softmax actually feeds expf */
    enum { N = 4096 };
    static float x[N], y[N];
    unsigned int s = 0x2468aceu;
    for (int i = 0; i < N; ++i) {
        s = s * 1103515245u + 12345u;
        const float u = ((float)((s >> 16) & 0x7FFF) / 16384.0f) - 1.0f;
        x[i] = -40.0f * (u * u) * (float)((s >> 8) & 1 ? 0.05f : 1.0f);
    }
    if (probe(x, y, N) != 0) { printf("EG-128 FAIL probe\n"); return 1; }

    int diff = 0, maxulp = 0;
    float worst_x = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float c = expf(x[i]);
        if (memcmp(&c, &y[i], 4) != 0) {
            ++diff;
            int ci, yi;
            memcpy(&ci, &c, 4);
            memcpy(&yi, &y[i], 4);
            const int d = abs(ci - yi);
            if (d > maxulp) { maxulp = d; worst_x = x[i]; }
        }
    }
    printf("EG-128 expf dev-vs-libc: %d/%d differ, max %d ulp (worst at x=%.6f)\n", diff, N, maxulp, worst_x);
    printf("EG-128 note: this is a measurement, not a pass/fail gate\n");
    return 0;
}
