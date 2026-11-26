/* iq_mlp_curve.c - bytes/s versus thread count at three working-set sizes.
 *
 * Two briefs disagree about this engine. One says decode is concurrency-bound: ~8 of the 24 miss slots Zen 4 exposes
 * per core are in flight, so it achieves 15 % of the achievable stream and row tiling would fix it. The other (from
 * Strata, on a 6-thread desktop Zen) says a quantised decode kernel is already at 100 % of RAM bandwidth at one token
 * per expert, so there is NO kernel headroom and tiling would buy nothing. Both cannot be true here, and which one is
 * decides whether the next month goes into tiling kernels or into something else.
 *
 * This settles it the only way it can be settled: run the engine's real IQ kernel over three working sets and watch
 * where the curve flattens.
 *
 *   512 KB  fits in L2  (1 MB per core on Zen 4)   -> latency/issue bound; a flat curve means the kernel itself
 *   8 MB    fits in L3  (16 MB shared)              -> shared-cache bound
 *   2048 MB streams from DRAM                       -> the real decode regime
 *
 * If DRAM throughput keeps climbing to 12 threads and approaches the machine's stream rate, we are memory-bound and
 * the tiling thesis is dead. If it flattens while still far below that rate, the kernel is the wall and R=4 is the fix.
 *
 * Build:
 *   gcc -O3 -std=c11 -march=znver4 -fopenmp -static -I include -o iq_mlp_curve.exe \
 *       tools/iq_mlp_curve.c src/wubu_iq_c11.c -lm
 * Run (thread count comes from OMP_NUM_THREADS):
 *   OMP_NUM_THREADS=8 ./iq_mlp_curve.exe 16 2048
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "wubu_iq_c11.h"
#include "wubu_iq2q8.h"
#if defined(_OPENMP)
#include <omp.h>
#endif

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

static int bytes_per_block_of(int ty)
{
    switch (ty) {
        case 16: return 66;   /* iq2_xxs */
        case 17: return 74;   /* iq2_xs  */
        case 19: return 50;   /* iq1_s   */
        case 22: return 82;   /* iq2_s   */
        case 42: return 34;   /* q8_0    */
        default: return 66;
    }
}

int main(int argc, char **argv)
{
    const int ty   = argc > 1 ? atoi(argv[1]) : 16;
    const int mb   = argc > 2 ? atoi(argv[2]) : 512;
    const int reps = argc > 3 ? atoi(argv[3]) : 3;
    const int n    = 2304;

    if (!wubu_iq_supported(ty) && !wubu_iq_rows1_supported(ty)) { printf("type %d unsupported\n", ty); return 1; }
    const int be = wubu_iq_block_elems(ty);
    const int bpb = bytes_per_block_of(ty);
    const size_t row_bytes = (size_t) (n / be) * (size_t) bpb;

    size_t want = (size_t) mb * 1024u * 1024u;
    int rows = (int) (want / row_bytes);
    if (rows < 1) rows = 1;
    const size_t wbytes = row_bytes * (size_t) rows;

    unsigned char *w = (unsigned char *) malloc(wbytes);
    const size_t ablocks = (size_t) (n / 256);
    in2_q8k_blk *act = (in2_q8k_blk *) malloc(ablocks * sizeof(in2_q8k_blk));
    float *out = (float *) malloc((size_t) rows * sizeof(float));
    if (!w || !act || !out) { printf("alloc failed: %zu bytes\n", wbytes); return 1; }

    /* Same fill discipline as iq_kernel_wall: random payload so table lookups do not all hit one line, but a valid
     * small fp16 scale per block so the float path is really executed. */
    unsigned long long s = 99991;
    for (size_t i = 0; i < wbytes; ++i) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; w[i] = (unsigned char) (s >> 33); }
    for (int r = 0; r < rows; ++r) {
        unsigned char *row = w + (size_t) r * row_bytes;
        for (int b = 0; b < n / be; ++b) {
            uint16_t h = half_of(0.01f);
            memcpy(row + (size_t) b * bpb, &h, sizeof h);
        }
    }
    for (size_t b = 0; b < ablocks; ++b) {
        act[b].d = 0.02f;
        for (int i = 0; i < 256; ++i) act[b].qs[i] = (int8_t) ((i * 7 + 3) % 31 - 15);
        for (int i = 0; i < 16; ++i) act[b].bsums[i] = (int16_t) ((i % 5) - 2);
    }

    const void *acts[1] = { act };
    float *outs[1] = { out };

    /* one warm pass so page faults and mapping are not charged to the measurement */
    wubu_iq_rows(ty, w, row_bytes, n, acts, 1, outs, 0, rows);

    double best = 1e9, worst = 0;
    for (int rep = 0; rep < reps; ++rep) {
        const double t0 = now();
        wubu_iq_rows(ty, w, row_bytes, n, acts, 1, outs, 0, rows);
        const double dt = now() - t0;
        if (dt < best)  best = dt;
        if (dt > worst) worst = dt;
    }

    const double weights = (double) rows * (double) n;
    const double gws = weights / best / 1e9;
    const double gbs = (double) wbytes / best / 1e9;
    int nt = 1;
#if defined(_OPENMP)
    nt = omp_get_max_threads();
#endif
    printf("type=%-3d %5d MB  %7d rows  threads=%2d  %7.2f G w/s  %6.2f GB/s  (spread %.0f%%)\n",
           ty, (int) (wbytes >> 20), rows, nt, gws, gbs, 100.0 * (worst - best) / best);

    free(w); free(act); free(out);
    return 0;
}
