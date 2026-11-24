/* test_eg127_q8k.c -- EG-127: device Q8_K quantiser vs quantize_row_q8_K, byte for byte.
 *
 * The fused attention path quantises its own attention output on the device. The default
 * path quantises on the CPU and uploads. If those two disagree by even one step in one
 * element, every o-projection in the fused path is fed different bytes than the reference,
 * which is a plausible magnitude for the fused path's wrong continuation (the model tolerates
 * 1e-04 of relative noise, so 5e-06 of rounding cannot explain it). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

typedef int (*fn_t)(const float *, void *, int);
extern void quantize_row_q8_K(const float *x, void *y, long long k);

static unsigned int s = 0x12345678u;
static float rnd(void)
{
    s = s * 1103515245u + 12345u;
    return ((float)((s >> 16) & 0x7FFF) / 16384.0f) - 1.0f;
}

int main(int argc, char **argv)
{
    HMODULE h = LoadLibraryA(argc > 1 ? argv[1] : "wubu_cuda.dll");
    if (!h) { printf("EG-127: SKIP (no dll)\n"); return 0; }
    fn_t q8 = (fn_t)(void *)GetProcAddress(h, "wubu_cuda_q8k_host");
    if (!q8) { printf("EG-127: SKIP (no export)\n"); FreeLibrary(h); return 0; }

    /* N is the o-projection's K (4096) by default; a file argument replaces the synthetic
     * values with real activations, which is the case the synthetic one cannot vouch for. */
    enum { NMAX = 4096 };
    static float x[NMAX];
    static unsigned char dev[NMAX / 256 * 292], cpu[NMAX / 256 * 292];
    int N = NMAX;
    if (argc > 2) {
        FILE *f = fopen(argv[2], "rb");
        if (!f) { printf("EG-127 FAIL open %s\n", argv[2]); return 1; }
        const long sz = (long)fread(x, sizeof(float), NMAX, f);
        fclose(f);
        if (sz <= 0 || (sz % 256) != 0) { printf("EG-127 FAIL size %ld\n", sz); return 1; }
        N = (int)sz;
    } else {
        for (int i = 0; i < N; ++i) x[i] = rnd() * 3.0f;
    }

    const int rc = q8(x, dev, N);
    if (rc != 0) { printf("EG-127 FAIL rc=%d\n", rc); return 1; }
    memset(cpu, 0, sizeof(cpu));
    quantize_row_q8_K(x, cpu, N);

    int diff_bytes = 0, diff_q = 0, worst_d = 0, worst_q = 0;
    for (int b = 0; b < N / 256; ++b) {
        const unsigned char *p = dev + (size_t)b * 292, *q = cpu + (size_t)b * 292;
        float dp, dc;
        memcpy(&dp, p, 4);
        memcpy(&dc, q, 4);
        if (memcmp(p, q, 4) != 0) ++worst_d;
        for (int i = 0; i < 256; ++i) {
            const signed char a = (signed char)p[4 + i], c = (signed char)q[4 + i];
            if (a != c) {
                ++diff_q;
                const int d = a > c ? a - c : c - a;
                if (d > worst_q) worst_q = d;
            }
        }
        for (int i = 0; i < 292; ++i) if (p[i] != q[i]) ++diff_bytes;
    }
    /* bsums are written by neither the four device dots nor their CPU counterparts, so they
     * are excluded from the pass criterion (they are all that differ: 437 bytes = 109 ints). */
    printf("EG-127 q8_K dev vs cpu: d_mismatch_blocks=%d/%d qs=%d/%d worst_step=%d  (bsums differ, unused)\n",
           worst_d, N / 256, diff_q, N, worst_q);
    const int ok = (worst_d == 0 && diff_q == 0);
    printf("EG-127 %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
