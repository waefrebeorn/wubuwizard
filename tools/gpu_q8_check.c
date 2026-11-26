/* gpu_q8_check.c -- does the device Q8_0 (type 8) matvec compute the right number?
 *
 * Q8_0 block = fp16 d + 32 int8 = 34 bytes; the activation is the same block type (32 elements). The reference is
 * the exact dequantised dot in double, so agreement gates the block layout, the scale and the accumulation.
 * Prints plain values and a numeric error, and never a timing until the numbers are right.
 *
 * Build: gcc -O2 -static -I include -o gpu_q8_check.exe tools/gpu_q8_check.c <dir>/wubu_cuda.lib -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef struct { uint16_t d; int8_t qs[32]; } q8_0_blk;

extern int wubu_cuda_quant_matvec_host(const void *h_w, int type, const void *h_xq8,
                                       float *h_y, int K, int n_cols, int64_t col_stride_bytes);

static float h2f(uint16_t h)
{
    const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) f = s << 31;
        else { int32_t ex = -1; uint32_t mm = m; while (!(mm & 0x400u)) { mm <<= 1; ex--; }
               f = (s << 31) | ((uint32_t)(127 - 15 - ex) << 23) | ((mm & 0x3ffu) << 13); }
    } else if (e == 31) f = (s << 31) | 0x7f800000u | (m << 13);
    else f = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float out;
    memcpy(&out, &f, 4);
    return out;
}

static uint16_t f2h(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000u;
    int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = (x >> 13) & 0x3ffu;
    if (e <= 0) return (uint16_t) s;
    if (e >= 31) return (uint16_t)(s | 0x7c00u);
    return (uint16_t)(s | ((uint32_t) e << 10) | m);
}

int main(int argc, char **argv)
{
    const int ty = 8;
    const int K = 2304;
    const int rows = argc > 1 ? atoi(argv[1]) : 2304;
    const int bpb = 34;
    const int nblocks = K / 32;

    const size_t row_bytes = (size_t) nblocks * (size_t) bpb;
    const size_t wbytes = row_bytes * (size_t) rows;
    unsigned char *w = (unsigned char *) malloc(wbytes);
    q8_0_blk *act = (q8_0_blk *) malloc((size_t) nblocks * sizeof(q8_0_blk));
    float *y = (float *) malloc((size_t) rows * sizeof(float));
    float *ref = (float *) malloc((size_t) rows * sizeof(float));
    if (!w || !act || !y || !ref) { printf("alloc failed\n"); return 1; }

    unsigned long long s = 4242;
    for (size_t i = 0; i < wbytes; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        w[i] = (unsigned char) (s >> 33);
    }
    /* real scales, so the fp16 path is exercised and a sign bug cannot hide */
    for (int r = 0; r < rows; ++r) {
        unsigned char *row = w + (size_t) r * row_bytes;
        for (int b = 0; b < nblocks; ++b) {
            uint16_t h = f2h(0.01f + 0.001f * (float)((r + b) % 7));
            memcpy(row + (size_t) b * bpb, &h, 2);
        }
    }
    for (int b = 0; b < nblocks; ++b) {
        act[b].d = f2h(0.02f);
        for (int i = 0; i < 32; ++i) act[b].qs[i] = (int8_t) ((int)((i * 13 + b * 7) % 51) - 25);
    }

    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        const unsigned char *row = w + (size_t) r * row_bytes;
        for (int b = 0; b < nblocks; ++b) {
            q8_0_blk blk;
            memcpy(&blk, row + (size_t) b * bpb, sizeof blk);
            const double wd = (double) h2f(blk.d);
            const double ad = (double) h2f(act[b].d);
            double dot = 0.0;
            for (int i = 0; i < 32; ++i) dot += (double) blk.qs[i] * (double) act[b].qs[i];
            acc += wd * ad * dot;
        }
        ref[r] = (float) acc;
    }

    memset(y, 0, (size_t) rows * sizeof(float));
    const int rc = wubu_cuda_quant_matvec_host(w, ty, act, y, K, rows, (int64_t) row_bytes);
    int nz = 0;
    for (int i = 0; i < rows; ++i) if (y[i] != 0.0f) nz++;
    printf("rc=%d  non-zero rows %d / %d\n", rc, nz, rows);
    printf("y[0]=%g  ref[0]=%g    y[last]=%g  ref[last]=%g\n", y[0], ref[0], y[rows - 1], ref[rows - 1]);

    double mx = 0.0, mag = 0.0;
    int over = 0;
    for (int i = 0; i < rows; ++i) {
        const double d = fabs((double) ref[i] - (double) y[i]);
        const double m = fabs((double) ref[i]);
        if (m > mag) mag = m;
        if (d > mx) mx = d;
        if (m > 1e-6 && d / m > 1e-3) over++;
    }
    printf("max |ref-gpu| = %.3e on magnitude %.4f ; rows over 1e-3 rel = %d / %d\n", mx, mag, over, rows);
    free(w); free(act); free(y); free(ref);
    return (rc == 0 && nz > 0 && over == 0) ? 0 : 1;
}
