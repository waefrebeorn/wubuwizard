/* gpu_resident_bench.c -- what does an IQ2_XXS slab that is ALREADY in VRAM stream at?
 *
 * The per-call host entry copies the weights every time, so it measures PCIe. A tandem engine keeps the dense tier
 * resident, so this is the number that decides whether the card is worth having. Sweep the slab size: at 1 MB the
 * launch dominates, at hundreds of MB the VRAM bandwidth should show.
 *
 * Build: gcc -O2 -static -o gpu_resident_bench.exe tools/gpu_resident_bench.c <dir>/wubu_cuda.lib -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } q8k_blk;

extern int wubu_cuda_iq2xxs_bench_host(const void *h_w, const void *h_q8, float *h_y,
                                       int K, int n_cols, int64_t col_stride_bytes,
                                       int reps, double *out_us);
extern int wubu_cuda_upload_iq2xxs(const void *grid_256_u64, const void *ksigns_128_u8);
extern const void *wubu_iq2xxs_grid_bytes(void);
extern const void *wubu_iq2xxs_ksigns_bytes(void);

static uint16_t half_of(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = (x >> 13) & 0x3ffu;
    if (e <= 0) return (uint16_t) sign;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t) e << 10) | m);
}

int main(int argc, char **argv)
{
    const int K = 2304;
    const int nblocks = K / 256;
    const int bpb = 66;
    const size_t row_bytes = (size_t) nblocks * (size_t) bpb;   /* 594 */

    static const double mb[] = { 1.3, 16.0, 64.0, 256.0, 1024.0, 4096.0 };
    const int nsz = (int)(sizeof mb / sizeof mb[0]);

    if (wubu_cuda_upload_iq2xxs(wubu_iq2xxs_grid_bytes(), wubu_iq2xxs_ksigns_bytes()) != 0) {
        printf("table upload failed\n");
        return 2;
    }
    q8k_blk *act = (q8k_blk *) malloc((size_t) nblocks * sizeof(q8k_blk));
    if (!act) return 1;
    for (int b = 0; b < nblocks; ++b) {
        act[b].d = 0.02f;
        for (int i = 0; i < 256; ++i) act[b].qs[i] = (int8_t) ((int)((i * 37 + b * 11) % 61) - 30);
        for (int i = 0; i < 16; ++i) {
            int sum = 0;
            for (int j = 0; j < 16; ++j) sum += act[b].qs[i * 16 + j];
            act[b].bsums[i] = (int16_t) sum;
        }
    }

    printf("%10s %10s %12s %14s %12s\n", "slab MB", "rows", "kernel us", "G weights/s", "GB/s read");
    for (int z = 0; z < nsz; ++z) {
        const double want = mb[z] * 1048576.0;
        int rows = (int)(want / (double) row_bytes);
        if (rows < 64) rows = 64;
        const size_t wbytes = row_bytes * (size_t) rows;
        unsigned char *w = (unsigned char *) malloc(wbytes);
        float *y = (float *) malloc((size_t) rows * sizeof(float));
        if (!w || !y) { printf("%10.1f  alloc failed (%.0f MB)\n", mb[z], wbytes / 1048576.0); free(w); free(y); continue; }
        unsigned long long s = 999 + z;
        for (size_t i = 0; i < wbytes; i += 8) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            memcpy(w + i, &s, (wbytes - i) >= 8 ? 8 : (wbytes - i));
        }
        for (int r = 0; r < rows; ++r) {
            unsigned char *row = w + (size_t) r * row_bytes;
            for (int b = 0; b < nblocks; ++b) {
                uint16_t h = half_of(0.01f);
                memcpy(row + (size_t) b * bpb, &h, sizeof h);
            }
        }
        double us = 0.0;
        const int rc = wubu_cuda_iq2xxs_bench_host(w, act, y, K, rows, (int64_t) row_bytes, 5, &us);
        if (rc != 0) { printf("%10.1f  rc=%d\n", mb[z], rc); free(w); free(y); continue; }
        const double wts = (double) rows * (double) K;
        printf("%10.1f %10d %12.1f %14.2f %12.2f\n", wbytes / 1048576.0, rows, us,
               wts / (us * 1e-6) / 1e9, (double) wbytes / (us * 1e-6) / 1e9);
        free(w); free(y);
    }
    free(act);
    printf("\nCPU kernel reference: 19.08 G weights/s single-threaded, 2.24x from 1->8 threads\n");
    printf("RTX 4050 Laptop: 6 GB GDDR6, ~192 GB/s theoretical\n");
    return 0;
}
