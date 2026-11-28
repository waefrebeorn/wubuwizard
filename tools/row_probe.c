/* Row-granularity probe: is the 12.2x gap in-engine a PER-ROW cost or a per-BYTE cost?
 *
 * The engine's expert region measured 479.2 us for 384 rows of a 2304-element IQ2_XXS matrix (row_bytes = 594). The isolated
 * Q8_0 probe measured 0.05 ns/elem with 2176-byte rows. Those differ in TWO ways, so this probe reproduces the engine's exact
 * geometry (ty=16, n=2304, rb=594) and then also runs a fat-row case (ty=16, n=4096, rb=1056) at the same element count. If
 * ns/elem is governed by rows, the fat-row case is much faster per element; if by bytes, they match.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubu_iq2q8.h"
#include "wubu_iq_c11.h"
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include <assert.h>
#include "ggml-common.h"
#include <windows.h>

void wubu_iq_rows(int ggml_type, const void *w, size_t row_bytes, int n,
                  const void *const *act, int nt, float *const *out, int r0, int r1);
double in2_hires_now(void){ static LARGE_INTEGER f; LARGE_INTEGER c; if(!f.QuadPart) QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart/(double)f.QuadPart; }
double in2_ty_sec[256]; long in2_ty_calls[256]; int in2_ty_par = 0; int in2_ty_depth = 0;

typedef struct { uint16_t d; uint16_t qs[QK_K / 8]; } blk16;
static_assert(sizeof(blk16) == 66, "IQ2_XXS block must be 66 bytes");

static uint32_t rs = 12345u;
static uint32_t nx(void){ rs = rs*1664525u + 1013904223u; return rs >> 8; }

static void run(const char *label, int n_in, int nblk_per_row, int rows, int reps)
{
    const size_t rb = (size_t) nblk_per_row * sizeof(blk16);
    unsigned char *w = (unsigned char *) malloc(rb * (size_t) rows);
    float *x = (float *) malloc(sizeof(float) * n_in);
    in2_q8k_blk *aq = (in2_q8k_blk *) malloc(sizeof(in2_q8k_blk) * (n_in / 256));
    float *out = (float *) malloc(sizeof(float) * rows);
    if (!w || !x || !aq || !out) { printf("oom\n"); return; }
    for (size_t i = 0; i < rb * (size_t) rows; ++i) w[i] = (unsigned char)(nx() & 0xFF);
    for (int i = 0; i < n_in; ++i) x[i] = ((int)(nx() % 2001) - 1000) / 1000.0f;
    in2_q8k_quantise(x, n_in / 256, aq);

    const void *actp[1] = { aq };
    for (int r = 0; r < rows; ++r) { float *op[1] = { out + r };            /* warm */
        wubu_iq_rows(16, w + (size_t) r * rb, rb, n_in, actp, 1, op, 0, 1); }

    double t0 = in2_hires_now(); size_t done = 0;
    for (int p = 0; p < reps; ++p)
        for (int r = 0; r < rows; ++r) {
            float *op[1] = { out + r };
            wubu_iq_rows(16, w + (size_t) r * rb, rb, n_in, actp, 1, op, 0, 1);
            done += (size_t) n_in;
        }
    double dt = in2_hires_now() - t0;
    printf("  %-22s rows=%4d rb=%4zu bytes=%8.2f MB  %6.2f ns/elem  %5.2f GB/s  %6.3f us/row\n",
           label, rows, rb, (double)(rb * (size_t)rows) / 1048576.0,
           dt * 1e9 / (double) done, (double) done * 4.0 / dt / 1e9,
           dt * 1e6 / (double)(reps * rows));
    free(w); free(x); free(aq); free(out);
}

int main(void)
{
    printf("IQ2_XXS (ty=16) row-granularity probe\n");
    run("engine geometry", 2304, 9, 384, 20);     /* exactly the region that reported 479.2 us */
    run("engine geom x4 rows", 2304, 9, 1536, 5);
    run("fat rows", 4096, 16, 216, 20);           /* same elements, 2.7x fewer rows */
    run("skinny rows", 256, 1, 3456, 3);          /* same elements, 16x more rows */
    /* DRAM-resident: 200k rows x 594 B = 113 MB, walked ONCE. This is the only case whose bytes cannot sit in L2/L3, so it is
     * the only one comparable to what the engine does per token. */
    run("DRAM 113 MB, one pass", 2304, 9, 200000, 1);

    /* UNFAULTED anonymous memory: VirtualAlloc reserves but nothing touches it, so every 4 KB page the kernel walks is a
     * first-touch fault. This isolates fault cost from pattern, residency and row size - the only variable left that
     * differs between this probe and the engine, which reads a file-backed mapping. */
    {
        const int rows = 200000;
        const size_t rb2 = 9 * sizeof(blk16);
        const size_t bytes = rb2 * (size_t) rows;
        unsigned char *v = (unsigned char *) VirtualAlloc(NULL, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (v) {
            float *x2 = (float *) malloc(sizeof(float) * 2304);
            in2_q8k_blk *aq2 = (in2_q8k_blk *) malloc(sizeof(in2_q8k_blk) * 9);
            float *o2 = (float *) malloc(sizeof(float) * 64);
            for (int i = 0; i < 2304; ++i) x2[i] = ((i * 37) % 199 - 99) / 99.0f;
            in2_q8k_quantise(x2, 9, aq2);
            const void *ap[1] = { aq2 };
            double t0 = in2_hires_now(); size_t done = 0;
            for (int r = 0; r < rows; ++r) {
                float *op[1] = { o2 };
                wubu_iq_rows(16, v + (size_t) r * rb2, rb2, 2304, ap, 1, op, 0, 1);
                done += 2304;
            }
            double dt = in2_hires_now() - t0;
            printf("  UNFAULTED 113 MB       rows=%4d rb=%4zu bytes=%8.2f MB  %6.2f ns/elem  %5.2f GB/s  %6.3f us/row\n",
                   rows, rb2, (double) bytes / 1048576.0,
                   dt * 1e9 / (double) done, (double) done * 4.0 / dt / 1e9, dt * 1e6 / (double) rows);
            free(x2); free(aq2); free(o2); VirtualFree(v, 0, MEM_RELEASE);
        }
    }
    return 0;
}
