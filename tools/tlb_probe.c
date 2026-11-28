/* TLB-pressure probe: does the Q8_0 row kernel slow down as the WORKING SET grows?
 *
 * Hypothesis under test (from the engine's own measurements: 40,600 pages/token, concurrency 0.33x per added thread, hot-page-
 * cache buying only 5%): the limit is TLB reach, not bandwidth, not disk, not locks. If that is right, ns/element must be flat
 * while the working set fits a few thousand pages and then degrade sharply once it exceeds TLB capacity - the knee is the
 * predicton, and its absence would falsify the hypothesis.
 *
 * Fixed total work per point (same number of elements processed), varying only how many DISTINCT pages those elements span.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubu_iq2q8.h"
#include "wubu_iq_c11.h"

void wubu_iq_rows(int ggml_type, const void *w, size_t row_bytes, int n,
                  const void *const *act, int nt, float *const *out, int r0, int r1);
#include <windows.h>
double in2_hires_now(void){ static LARGE_INTEGER f; LARGE_INTEGER c; if(!f.QuadPart) QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart/(double)f.QuadPart; }
double in2_ty_sec[256]; long in2_ty_calls[256]; int in2_ty_par = 0; int in2_ty_depth = 0;

#define NIN   2048
#define NROW  16
#define BLK   34                       /* Q8_0 block: d(2) + qs[32] */
#define RB    ((size_t) NIN / 32 * BLK) /* 2176 bytes per row */

int main(void)
{
    const size_t row_bytes = RB;
    const size_t rows_per_slab = 256;                 /* 557 KB per slab */
    const size_t slab_bytes   = rows_per_slab * row_bytes;
    const size_t TOTAL_ELEMS  = 64u * 1024 * 1024;    /* constant work: 64M elements per point */
    const size_t slabs[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };
    const int ns = (int)(sizeof(slabs) / sizeof(slabs[0]));

    printf("Q8_0 kernel: working-set sweep, fixed work = %zu M elements\n", TOTAL_ELEMS / 1024 / 1024);
    printf("  slab = %zu rows = %.2f MB\n\n", rows_per_slab, (double) slab_bytes / 1048576.0);

    unsigned char *w = (unsigned char *) malloc(slab_bytes * 256);
    float *x = (float *) malloc(sizeof(float) * NIN);
    in2_q8_0_blk *aq = (in2_q8_0_blk *) malloc(sizeof(in2_q8_0_blk) * (NIN / 32));
    float *out = (float *) malloc(sizeof(float) * NROW);
    if (!w || !x || !aq || !out) { printf("oom\n"); return 2; }

    for (size_t i = 0; i < slab_bytes * 256; ++i) w[i] = (unsigned char) (i * 1103515245u >> 13);
    for (int i = 0; i < NIN; ++i) x[i] = ((i * 37) % 199 - 99) / 99.0f;

    /* quantise with the Q8_0 path the kernel actually consumes */
    for (int b = 0; b < NIN / 32; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) { float a = fabsf(x[b * 32 + i]); if (a > amax) amax = a; }
        const float d = amax / 127.0f;
        aq[b].d = in2_f32_to_f16(d ? d : 1e-8f);
        for (int i = 0; i < 32; ++i) {
            float q = x[b * 32 + i] / d;
            int v = (int) lrintf(q);
            if (v > 127) v = 127; if (v < -128) v = -128;
            aq[b].qs[i] = (int8_t) v;
        }
    }

    for (int s = 0; s < ns; ++s) {
        const size_t nslab = slabs[s];
        if (nslab * slab_bytes > slab_bytes * 256) continue;
        const size_t rows_total = nslab * rows_per_slab;
        const size_t passes = TOTAL_ELEMS / (rows_total * NIN) + 1;

        /* warm the mapping once so page-table setup is not counted as a per-point cost */
        const void *actp[1] = { aq };
        for (size_t r = 0; r < rows_total; r += 1) {
            float *op[1] = { out };
            wubu_iq_rows(8, w + r * row_bytes, row_bytes, NIN, actp, 1, op, 0, 1);
        }

        double t0 = in2_hires_now();
        size_t done = 0;
        for (size_t p = 0; p < passes; ++p)
            for (size_t r = 0; r < rows_total; ++r) {
                float *op[1] = { out };
                wubu_iq_rows(8, w + r * row_bytes, row_bytes, NIN, actp, 1, op, 0, 1);
                done += NIN;
            }
        double dt = in2_hires_now() - t0;
        printf("  set=%5zu slabs = %7.2f MB   %6.2f ns/elem   %6.2f GB/s effective\n",
               nslab, (double) (nslab * slab_bytes) / 1048576.0,
               dt * 1e9 / (double) done, (double) done * 4.0 / dt / 1e9);
    }


    /* FIRST-TOUCH vs WARM: the engine touches ~41k FRESH pages per token, so the cost that matters is the first pass over
     * bytes, not the second. A large anonymous buffer makes that comparable without any file involvement. */
    {
        const size_t BIG = 512u * 1024 * 1024;
        unsigned char *b = (unsigned char *) malloc(BIG);
        if (b) {
            for (size_t i = 0; i < BIG; i += 4096) b[i] = 1;      /* ensure committed, pages not yet in working set */
            const size_t rows = BIG / row_bytes;
            const void *actp2[1] = { aq };
            for (int pass = 0; pass < 3; ++pass) {
                double t0 = in2_hires_now(); size_t done = 0;
                for (size_t r = 0; r < rows; ++r) {
                    float *op[1] = { out };
                    wubu_iq_rows(8, b + r * row_bytes, row_bytes, NIN, actp2, 1, op, 0, 1);
                    done += NIN;
                }
                double dt = in2_hires_now() - t0;
                printf("  512 MB pass %d: %6.2f ns/elem  %6.2f GB/s\n", pass + 1,
                       dt * 1e9 / (double) done, (double) done * 4.0 / dt / 1e9);
            }
            free(b);
        }
    }
    free(w); free(x); free(aq); free(out);
    return 0;
}
