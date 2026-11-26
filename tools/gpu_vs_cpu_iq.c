/* gpu_vs_cpu_iq.c -- does the new device IQ2_XXS row kernel compute the right thing?
 *
 * The reference here is the EXACT dequantised path from src/dequant_iq2_xxs.c: decode each weight block to floats and
 * dot against the activation reconstructed from the q8_K blocks the device kernel consumes. Agreement proves the
 * device kernel has the right block layout, grid index, sign bits and 4-bit subgroup scale - the four things that can
 * silently produce a plausible-looking wrong answer.
 *
 * Build:
 *   nvcc ... -shared -o wubu_cuda.dll src/wubu_cuda_bridge.cu src/wubu_cuda_quant.cu -lcudart
 *   gcc -O2 -static -I include -o gpu_vs_cpu_iq.exe tools/gpu_vs_cpu_iq.c src/dequant_iq2_xxs.c <dir>/wubu_cuda.lib -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } q8k_blk;

extern void dequantize_iq2_xxs_block(const uint8_t *block, float *output);
extern int wubu_cuda_quant_matvec_host(const void *h_w, int type, const void *h_xq8,
                                       float *h_y, int K, int n_cols, int64_t col_stride_bytes);
extern int wubu_cuda_upload_iq2xxs(const void *grid_256_u64, const void *ksigns_128_u8);
extern const void *wubu_iq2xxs_grid_bytes(void);
extern const void *wubu_iq2xxs_ksigns_bytes(void);

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + 1e-9 * (double) t.tv_nsec;
}

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
    const int ty = 16;
    const int K = 2304;
    const int rows = argc > 1 ? atoi(argv[1]) : 2304;
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    const int bpb = 66;
    const int nblocks = K / 256;
    const size_t row_bytes = (size_t) nblocks * (size_t) bpb;
    const size_t wbytes = row_bytes * (size_t) rows;

    unsigned char *w = (unsigned char *) malloc(wbytes);
    q8k_blk *act = (q8k_blk *) malloc((size_t) nblocks * sizeof(q8k_blk));
    float *ref = (float *) malloc((size_t) rows * sizeof(float));
    float *gpu = (float *) malloc((size_t) rows * sizeof(float));
    float *tmp = (float *) malloc(256 * sizeof(float));
    if (!w || !act || !ref || !gpu || !tmp) { printf("alloc failed (%zu bytes)\n", wbytes); return 1; }

    unsigned long long s = 12345;
    for (size_t i = 0; i < wbytes; ++i) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; w[i] = (unsigned char) (s >> 33); }
    for (int r = 0; r < rows; ++r) {
        unsigned char *row = w + (size_t) r * row_bytes;
        for (int b = 0; b < nblocks; ++b) {
            uint16_t h = half_of(0.01f);
            memcpy(row + (size_t) b * bpb, &h, sizeof h);
        }
    }
    for (int b = 0; b < nblocks; ++b) {
        act[b].d = 0.02f;
        for (int i = 0; i < 256; ++i) act[b].qs[i] = (int8_t) ((int)((i * 37 + b * 11) % 61) - 30);
        for (int i = 0; i < 16; ++i) {
            int sum = 0;
            for (int j = 0; j < 16; ++j) sum += act[b].qs[i * 16 + j];
            act[b].bsums[i] = (int16_t) sum;
        }
    }

    const int urc = wubu_cuda_upload_iq2xxs(wubu_iq2xxs_grid_bytes(), wubu_iq2xxs_ksigns_bytes());
    printf("table upload rc=%d   %d rows x %d weights, %.2f MB\n", urc, rows, K, wbytes / 1048576.0);
    if (urc != 0) return 2;

    const double tr0 = now();
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        const unsigned char *row = w + (size_t) r * row_bytes;
        for (int b = 0; b < nblocks; ++b) {
            dequantize_iq2_xxs_block(row + (size_t) b * bpb, tmp);
            const double dq = (double) act[b].d;
            for (int i = 0; i < 256; ++i) acc += (double) tmp[i] * dq * (double) act[b].qs[i];
        }
        ref[r] = (float) acc;
    }
    const double t_ref = now() - tr0;

    double t_gpu = 1e9;
    int rc_last = 0;
    for (int r = 0; r < reps; ++r) {
        memset(gpu, 0, (size_t) rows * sizeof(float));
        const double tg0 = now();
        rc_last = wubu_cuda_quant_matvec_host(w, ty, act, gpu, K, rows, (int64_t) row_bytes);
        const double dt = now() - tg0;
        if (dt < t_gpu) t_gpu = dt;
        int nonzero = 0;
        for (int i = 0; i < rows; ++i) if (gpu[i] != 0.0f) { nonzero = 1; break; }
        printf("  rep %d: rc=%d %.4f s   output %s\n", r, rc_last, dt, nonzero ? "NON-ZERO" : "ALL ZEROS");
    }

    double maxabs = 0.0, maxrel = 0.0, mag = 0.0;
    int over = 0;
    for (int i = 0; i < rows; ++i) {
        const double d = fabs((double) ref[i] - (double) gpu[i]);
        const double m = fabs((double) ref[i]);
        if (m > mag) mag = m;
        if (d > maxabs) maxabs = d;
        if (m > 1e-6) { const double rel = d / m; if (rel > maxrel) maxrel = rel; if (rel > 1e-3) over++; }
    }
    printf("\nreference magnitude max |y| = %.6f\n", mag);
    printf("max |ref-gpu| = %.3e    max relative = %.3e    rows over 1e-3 relative = %d / %d\n",
           maxabs, maxrel, over, rows);

    const double wts = (double) rows * (double) K;
    printf("\nreference (dequant+dot) : %.3f s\n", t_ref);
    printf("GPU host call           : %.4f s  -> %.2f G weights/s (includes host->device copy)\n", t_gpu, wts / t_gpu / 1e9);
    printf("CPU kernel reference    : 19.08 G weights/s single-threaded (measured elsewhere)\n");
    printf("slab %.2f MB; PCIe 4.0 x8 at ~11 GB/s would copy it in %.0f us\n",
           wbytes / 1048576.0, wbytes / 11e9 * 1e6);
    free(w); free(act); free(ref); free(gpu); free(tmp);
    return over ? 1 : 0;
}
