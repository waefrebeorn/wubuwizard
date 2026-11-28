/* Independent numerical reference for the quantised row kernel (Q8_0 / ty=8).
 *
 * WHY: every test in this repo compares the engine against itself. KV vs full-recompute, the prefix property and the
 * four-prompt sweep all pass if BOTH paths are wrong in the same way, and the only upstream agreement we have is one
 * prompt at 12 tokens. This test compares the kernel against the specification it is supposed to implement:
 *   y[o] = sum_blocks  f16(w.d) * f16(a.d) * sum_i w.qs[i] * a.qs[i]
 * accumulated in double. The kernel accumulates in float/int, so the assertion is a relative-error bound, not equality.
 *
 * Build:  gcc -O2 -Iinclude tools/iq_ref_check.c src/wubu_iq_c11.c src/wubu_act_quant.c -o iq_ref_check
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

/* wubu_iq_c11.c references the engine's clock and per-type counters. Providing stubs keeps this test independent of the
 * engine: it links the KERNEL only, so a failure here cannot be caused by the forward pass. */
double in2_hires_now(void) { return 0.0; }
double in2_ty_sec[256];
long   in2_ty_calls[256];
int    in2_ty_par = 0;
int    in2_ty_depth = 0;

#define NB  64          /* blocks per row  => n_in = 2048 */
#define NIN (NB * 32)
#define NROW 16         /* rows to check */

static uint32_t rs = 12345u;
static uint32_t nx(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static float frnd(void) { return ((int)(nx() % 2001) - 1000) / 1000.0f; }

int main(void)
{
const size_t rb = (size_t) NB * sizeof(in2_q8_0_blk);   /* 34 bytes per block, same layout for Q8_0 weights */
    unsigned char *w = (unsigned char *) malloc(rb * NROW);
    float *x = (float *) malloc(sizeof(float) * NIN);
    in2_q8_0_blk *aq = (in2_q8_0_blk *) malloc(sizeof(in2_q8_0_blk) * NB);
    float *ref = (float *) malloc(sizeof(float) * NROW);
    float *got = (float *) malloc(sizeof(float) * NROW);
    if (!w || !x || !aq || !ref || !got) { fprintf(stderr, "oom\n"); return 2; }

    for (int i = 0; i < NIN; ++i) x[i] = frnd();

    /* synthetic Q8_0 weights: fp16 scale + 32 int8 per block, generated directly (no encoder needed) */
    for (int o = 0; o < NROW; ++o)
        for (int b = 0; b < NB; ++b) {
            in2_q8_0_blk *blk = (in2_q8_0_blk *) (w + (size_t) o * rb) + b;
            blk->d = in2_f32_to_f16(0.001f + 0.0001f * (float)(nx() % 20));
            for (int i = 0; i < 32; ++i) blk->qs[i] = (int8_t) ((int)(nx() % 255) - 127);
        }

    in2_q8_0_quantise(x, NB, aq);

    /* reference: the specification, in double */
    for (int o = 0; o < NROW; ++o) {
        double acc = 0.0;
        const in2_q8_0_blk *row = (const in2_q8_0_blk *) (w + (size_t) o * rb);
        for (int b = 0; b < NB; ++b) {
            const double wd = (double) in2_f16_to_f32(row[b].d);
            const double ad = (double) in2_f16_to_f32(aq[b].d);
            int isum = 0;
            for (int i = 0; i < 32; ++i) isum += (int) row[b].qs[i] * (int) aq[b].qs[i];
            acc += wd * ad * (double) isum;
        }
        ref[o] = (float) acc;
    }

    /* kernel */
    const void *actp[1] = { aq };
    for (int o = 0; o < NROW; ++o) {
        float *op[1] = { got + o };
        wubu_iq_rows(8, (const void *) ((const unsigned char *) w + (size_t) o * rb), rb, NIN, actp, 1, op, 0, 1);
    }

    /* relative error, reported as min/median/max - never a mean alone */
    double err[NROW];
    for (int o = 0; o < NROW; ++o) {
        const double d = fabs((double) got[o] - (double) ref[o]);
        const double s = fmax(fabs((double) ref[o]), 1e-12);
        err[o] = d / s;
    }
    for (int i = 1; i < NROW; ++i) {         /* insertion sort, NROW is tiny */
        double v = err[i]; int j = i - 1;
        while (j >= 0 && err[j] > v) { err[j + 1] = err[j]; --j; }
        err[j + 1] = v;
    }
    printf("Q8_0 reference check: %d rows x %d elements (%d blocks)\n", NROW, NIN, NB);
    printf("  diag: supported=%d rows1_supported=%d sizeof(blk)=%zu expect 34 row_bytes=%zu\n",
           wubu_iq_supported(8), wubu_iq_rows1_supported(8), sizeof(in2_q8_0_blk), rb);
    {   /* single-row entry: if this writes a value while rows() writes zeros, the r0/r1 form is what I got wrong */
        float one = 0.0f; float *op1[1] = { &one };
        wubu_iq_rows1(8, (const void *) w, rb, NIN, actp, 1, op1, 0, 1);
        float lot = 0.0f; float *op2[1] = { &lot };
        wubu_iq_rows(8, (const void *) w, rb, NIN, actp, 1, op2, 0, 1);
        printf("  diag: rows1[0]=%.6f  rows[0]=%.6f  ref[0]=%.6f\n", one, lot, (double) ref[0]);
    }
    printf("  rel err min %.3e  median %.3e  max %.3e\n", err[0], err[NROW / 2], err[NROW - 1]);
    printf("  sample  ref[0]=%.6f got[0]=%.6f | ref[1]=%.6f got[1]=%.6f\n", ref[0], got[0], ref[1], got[1]);
    const int pass = (err[NROW - 1] < 1e-5);
    printf("  %s (tolerance 1e-5)\n", pass ? "PASS" : "FAIL");
    /* --- quantiser round-trip: measures in2_q8_0_quantise ALONE, no kernel involved.
     * The kernel test above consumes the same quantiser on both sides, so a wrong quantiser cancels out of it entirely.
     * Q8_0 is symmetric int8 with one fp16 scale per 32 elements, so the expected relative L2 error is ~4e-3. */
    {
        double num = 0.0, den = 0.0;
        for (int b = 0; b < NB; ++b) {
            const double d = (double) in2_f16_to_f32(aq[b].d);
            for (int i = 0; i < 32; ++i) {
                const double q = d * (double) aq[b].qs[i];
                const double e = q - (double) x[b * 32 + i];
                num += e * e; den += (double) x[b * 32 + i] * (double) x[b * 32 + i];
            }
        }
        const double rel = sqrt(num / den);
        printf("  quantiser round-trip: rel L2 err %.3e (Q8_0 32-elem blocks, expect ~4e-3)\n", rel);
    }

    free(w); free(x); free(aq); free(ref); free(got);
    return pass ? 0 : 1;
}
