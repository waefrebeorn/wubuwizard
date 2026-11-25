/* Gate for the fused quant dots: the kernel must reproduce the dequantise-then-dot result.
 *
 * This is an instrument-validation test, not a benchmark: it proves the fused path computes the same number
 * as the reference path on synthetic blocks, so a later mismatch on the model means the wiring, not the maths.
 * Timing is reported because it is free to report, not because it is the claim.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "wubu_dequant_fp4.h"

#define N_ELEMS 2304     /* a real down row: 36 blocks of 64 */
#define NBLK    (N_ELEMS / 64)

static void fill(unsigned char *blk, size_t n, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < n; ++i) blk[i] = (unsigned char)(rand() & 0xFF);
    /* keep every fp16 scale finite and positive: exponent field 1..30 */
    for (long b = 0; b < NBLK; ++b) {
        unsigned e = 1u + (unsigned)(rand() % 30);
        unsigned man = (unsigned)(rand() & 0x3FF);
        unsigned half = (e << 10) | man;
        blk[b * 18 + 0] = (unsigned char)(half & 0xFF);
        blk[b * 18 + 1] = (unsigned char)((half >> 8) & 0xFF);
    }
}

int main(void)
{
    unsigned char *blk = malloc(NBLK * 18);
    float *x = malloc(N_ELEMS * sizeof(float));
    float *row = malloc(N_ELEMS * sizeof(float));
    if (!blk || !x || !row) return 2;

    double worst = 0.0;
    for (unsigned t = 0; t < 64; ++t) {
        fill(blk, NBLK * 18, 1234u + t);
        srand(999u + t);
        float maxx = 0.0f;
        for (int i = 0; i < N_ELEMS; ++i) {
            x[i] = (float)((rand() / (double)RAND_MAX) * 2.0 - 1.0);
            if (fabsf(x[i]) > maxx) maxx = fabsf(x[i]);
        }
        /* reference: materialise the row, then dot */
        dequantize_row_q2_0(blk, row, N_ELEMS);
        double ref = 0.0, mag = 0.0;
        for (int i = 0; i < N_ELEMS; ++i) { ref += (double)row[i] * x[i]; mag += fabs((double)row[i] * x[i]); }
        /* fused */
        float got = 0.0f;
        dot_row_q2_0(blk, x, N_ELEMS, &got);
        const double denom = mag > 1e-30 ? mag : 1e-30;
        const double rel = fabs((double)got - ref) / denom;
        if (rel > worst) worst = rel;
        if (t == 0) printf("trial0 ref=%.9g fused=%.9g rel=%.3e |x|max=%.4f\n", ref, (double)got, rel, maxx);
    }
    printf("worst relative error over 64 trials: %.3e  -> %s\n", worst, worst < 1e-5 ? "PASS" : "FAIL");

    const int iters = 120000;
    clock_t t0 = clock();
    float acc = 0.0f;
    for (int k = 0; k < iters; ++k) dot_row_q2_0(blk, x, N_ELEMS, &acc);
    double t_fused = (double)(clock() - t0) / CLOCKS_PER_SEC;
    t0 = clock();
    for (int k = 0; k < iters; ++k) {
        dequantize_row_q2_0(blk, row, N_ELEMS);
        double s = 0.0;
        for (int i = 0; i < N_ELEMS; ++i) s += (double)row[i] * x[i];
        acc += (float)s;
    }
    double t_ref = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("row dot x%d: fused %.3fs  dequant+dot %.3fs  speedup %.2fx (acc %.4f)\n",
           iters, t_fused, t_ref, t_ref / (t_fused > 1e-9 ? t_fused : 1e-9), acc);
    free(blk); free(x); free(row);
    return worst < 1e-5 ? 0 : 1;
}
