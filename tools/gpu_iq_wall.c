/* gpu_iq_wall.c - what does the GPU actually do with an IQ2_XXS matvec, end to end?
 *
 * The CPU kernel measures 19.08 G weights/s single-threaded (tools/iq_kernel_wall.c). This calls the same shape through
 * wubu_cuda_quant_matvec_host, which takes HOST weights, so the number includes the host->device copy - i.e. it is the
 * realistic cost of the "fetch a slab and compute it on the GPU" path that Strata/Avalon use for expert misses, not a
 * pure kernel number. Both are needed; this is the one that decides whether fetching beats computing.
 *
 * Build (MSVC's cl.exe is needed by nvcc but not by this file):
 *   nvcc ... -shared -o wubu_cuda.dll src/wubu_cuda_bridge.cu src/wubu_cuda_quant.cu -lcudart   (produces wubu_cuda.lib)
 *   gcc -O2 -std=c11 -o gpu_iq_wall.exe tools/gpu_iq_wall.c <dir>/wubu_cuda.lib
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

/* matches block_q8_K / c_q8_k: one float scale, 256 int8, 16 int16 sums */
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } q8k_blk;

extern int wubu_cuda_quant_matvec_host(const void *h_w, int type, const void *h_xq8,
                                       float *h_y, int K, int n_cols, int64_t col_stride_bytes);

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + 1e-9 * (double) t.tv_nsec;
}

int main(int argc, char **argv)
{
    const int K = 2304;                 /* one row of this model */
    const int cols = argc > 1 ? atoi(argv[1]) : 2304;   /* rows of the matrix = output width */
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    const int nblocks = K / 256;        /* 9 */
    const int gtype = getenv("GPU_TYPE") ? atoi(getenv("GPU_TYPE")) : 16;
    const int blk_bytes = getenv("GPU_BLK") ? atoi(getenv("GPU_BLK")) : 66;  /* iq2_xxs 66, iq2_s 82 */
    const size_t wbytes = (size_t) cols * nblocks * blk_bytes;

    unsigned char *w = (unsigned char *) malloc(wbytes);
    q8k_blk *a = (q8k_blk *) malloc((size_t) nblocks * sizeof(q8k_blk));
    float *y = (float *) malloc((size_t) cols * sizeof(float));
    if (!w || !a || !y) { printf("alloc failed (%zu bytes)\n", wbytes); return 1; }

    unsigned long long s = 999;
    for (size_t i = 0; i < wbytes; ++i) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; w[i] = (unsigned char) (s >> 33); }
    for (int b = 0; b < nblocks; ++b) {
        a[b].d = 0.02f;
        for (int i = 0; i < 256; ++i) a[b].qs[i] = (int8_t) ((i * 5 + 1) % 29 - 14);
        for (int i = 0; i < 16; ++i) a[b].bsums[i] = (int16_t) ((i % 3) - 1);
    }

    double best = 1e9;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now();
        const int rc = wubu_cuda_quant_matvec_host(w, gtype, a, y, K, cols, (int64_t) nblocks * blk_bytes);
        const double dt = now() - t0;
        if (dt < best) best = dt;
        printf("  type %d: rep %d: rc=%d %.4f s  (y[0]=%g y[last]=%g)\n", r, rc, dt, y[0], y[cols - 1]);
    }

    const double weights = (double) cols * (double) K;
    printf("\n%.2f M weights, %.2f MB of quantised slab, in %.4f s\n",
           weights / 1e6, wbytes / 1048576.0, best);
    printf("  %.2f G weights/s end-to-end (includes host->device copy)\n", weights / best / 1e9);
    printf("  effective host->device bandwidth if copy-bound: %.2f GB/s\n", wbytes / best / 1e9);
    printf("\nCPU reference: 19.08 G weights/s single-threaded, 2.24x from 1->8 threads\n");
    printf("PCIe 4.0 x8 expectation: ~10-13 GB/s, so this slab should copy in ~%.0f us\n",
           wbytes / 11e9 * 1e6);
    free(w); free(a); free(y);
    return 0;
}
