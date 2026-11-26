/* iq_kernel_wall.c - how fast is the IQ2_XXS dot kernel really, per thread?

This is the decisive measurement for the whole prefill wall. The engine sustains 43.3 GFLOPS at prefill, which is
the machine's fp32 ceiling, and upstream reports ~17 G weights/s per core for this exact format (14.94 ns per
256-weight block). If this kernel reaches that, 61 hours for a 1M prefill is the floor and the remaining lever is
fewer FLOPs (pooling/sparsity). If it is far below, there is a kernel win worth 4-5x on everything compute-bound.

Build:
    gcc -O2 -std=c11 -march=znver4 -static -I include -o iq_wall.exe tools/iq_kernel_wall.c src/wubu_iq_c11.c -lm
Run:
    ./iq_wall.exe            # type 16 = IQ2_XXS, 4096 rows x 2304
    ./iq_wall.exe 17         # other types: 17 IQ2_XS, 19 IQ1_S, 22 IQ2_S, 42 Q8_0-ish
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "wubu_iq_c11.h"
#include "wubu_iq2q8.h"

/* Minimal IEEE half encoder so the bench does not have to link the engine: only small positive scales are needed. */
static uint16_t half_of(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    return (uint16_t) (sign | ((uint32_t) e << 10) | (mant >> 13));
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + 1e-9 * (double) t.tv_nsec;
}

int main(int argc, char **argv)
{
    const int ty = argc > 1 ? atoi(argv[1]) : 16;
    const int n = 2304;              /* matches this model's d_model and d_ff_expert rows */
    const int rows = 4096;
    const int reps = argc > 2 ? atoi(argv[2]) : 5;

    if (!wubu_iq_rows1_supported(ty)) { printf("type %d not supported by rows1 here\n", ty); return 1; }
    const int be = wubu_iq_block_elems(ty);
    const char *kind = wubu_iq_act_kind(ty);
    printf("type %d: %s, block_elems=%d, act=%s\n", ty,
           ty == 16 ? "IQ2_XXS" : ty == 17 ? "IQ2_XS" : ty == 19 ? "IQ1_S" :
           ty == 22 ? "IQ2_S" : ty == 42 ? "Q8_0" : "?", be, kind ? kind : "?");

    /* Row bytes: n / block_elems blocks per row. Sizes per the static_asserts in ggml-common.h. */
    int bytes_per_block;
    switch (ty) {
        case 16: bytes_per_block = 66;  break;   /* iq2_xxs */
        case 17: bytes_per_block = 74;  break;   /* iq2_xs  */
        case 19: bytes_per_block = 50;  break;   /* iq1_s   */
        case 22: bytes_per_block = 82;  break;   /* iq2_s   */
        case 42: bytes_per_block = 34;  break;   /* q8_0    */
        default: bytes_per_block = 66;  break;
    }
    const size_t row_bytes = (size_t)(n / be) * (size_t) bytes_per_block;
    const size_t wbytes = row_bytes * (size_t) rows;

    unsigned char *w = (unsigned char *) malloc(wbytes);
    /* Activations: a block_q8_K array is one block per 256 elements for the K-quants, one per 32 for Q8_0. */
    const size_t ablocks = (size_t)(n / 256);
    in2_q8k_blk *act = (in2_q8k_blk *) malloc(ablocks * sizeof(in2_q8k_blk));
    float *out = (float *) malloc((size_t) rows * sizeof(float));
    if (!w || !act || !out) { printf("alloc failed (%zu bytes)\n", wbytes); return 1; }

    /* Fill the weights with a repeating pattern. This is deliberately not all zeros: a zero block can make the sign
       and grid table lookups hit the same cache line every time, which would flatter the measurement. */
    unsigned long long s = 12345;
    for (size_t i = 0; i < wbytes; ++i) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; w[i] = (unsigned char) (s >> 33); }
    /* Set each block's fp16 scale to a small positive value so the float path is exercised, not short-circuited. */
    for (int r = 0; r < rows; ++r) {
        unsigned char *row = w + (size_t) r * row_bytes;
        for (int b = 0; b < n / be; ++b) {
            uint16_t h = half_of(0.01f);
            memcpy(row + (size_t) b * bytes_per_block, &h, sizeof h);
        }
    }
    for (size_t b = 0; b < ablocks; ++b) {
        act[b].d = 0.02f;
        for (int i = 0; i < 256; ++i) act[b].qs[i] = (int8_t) ((i * 7 + 3) % 31 - 15);
        for (int i = 0; i < 16; ++i) act[b].bsums[i] = (int16_t) ((i % 5) - 2);
    }

    const void *acts[1] = { act };
    float *outs[1] = { out };
    double best = 1e9;
    for (int rep = 0; rep < reps; ++rep) {
        const double t0 = now();
        wubu_iq_rows1(ty, w, row_bytes, n, acts, 1, outs, 0, rows);
        const double dt = now() - t0;
        if (dt < best) best = dt;
        printf("  rep %d: %.4f s\n", rep, dt);
    }

    const double weights = (double) rows * (double) n;
    const double gws = weights / best / 1e9;
    printf("\n%.2f M weights in %.4f s = %.2f G weights/s (single thread), %.2f GFLOPS\n",
           weights / 1e6, best, gws, 2.0 * gws);
    printf("weights read: %.2f MB, so %.2f GB/s\n", wbytes / 1048576.0, wbytes / best / 1e9);
    printf("ns per 256-weight block: %.2f\n", 1e9 * best / (weights / 256.0));
    printf("\nreference: upstream AVX2 IQ2_XXS = 17.1 G weights/s per core (14.94 ns/block)\n");
    printf("           this measurement is %.1fx that reference\n", gws / 17.1);

    free(w); free(act); free(out);
    return 0;
}
