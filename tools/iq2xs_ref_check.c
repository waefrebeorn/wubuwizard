/* Independent numerical reference for the IQ2_XS row kernel (ggml_type 17) - 6 EXPERT tensors, 2.3125 bpw.
 *
 * Dequantisation copied VERBATIM from upstream llama.cpp dequantize_row_iq2_xs; iq2xs_grid, ksigns_iq2xs and kmask_iq2xs
 * come from ggml-common.h. Layout asserted at 74 bytes = d(2) + qs[32] uint16 (64) + scales[8].
 *
 * NOTE: src/gguf_reader.c's own dequantize_iq2_xs_row carries the comment "still assumes the old 108B layout -- needs
 * rewrite", so this test is also a check on whether the KERNEL agrees with upstream's real layout while the repo's scalar
 * fallback does not. If the kernel passes here, the fallback is the defective one.
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

void wubu_iq_rows(int ggml_type, const void *w, size_t row_bytes, int n,
                  const void *const *act, int nt, float *const *out, int r0, int r1);
double in2_hires_now(void) { return 0.0; }
double in2_ty_sec[256]; long in2_ty_calls[256]; int in2_ty_par = 0; int in2_ty_depth = 0;

#define QKK   256
#define NBLK  16
#define NIN   (QKK * NBLK)
#define NROW  8
typedef struct { uint16_t d; uint16_t qs[QKK / 8]; uint8_t scales[QKK / 32]; } blk17;
static_assert(sizeof(blk17) == 74, "IQ2_XS block must be 74 bytes: d(2) + qs[32]u16(64) + scales(8)");

static uint32_t rs = 555111u;
static uint32_t nx(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

/* verbatim structure of upstream dequantize_row_iq2_xs for one superblock */
static void deq_iq2xs(const blk17 *x, double *y)
{
    const double d = (double) in2_f16_to_f32(x->d);
    for (int ib32 = 0; ib32 < QKK / 32; ++ib32) {
        const double db0 = d * (0.5 + (double) (x->scales[ib32] & 0xf)) * 0.25;
        const double db1 = d * (0.5 + (double) (x->scales[ib32] >> 4)) * 0.25;
        const double db[2] = { db0, db1 };
        for (int l = 0; l < 4; ++l) {
            const uint8_t *grid = (const uint8_t *) (iq2xs_grid + (x->qs[4 * ib32 + l] & 511));
            const uint8_t  signs = ksigns_iq2xs[x->qs[4 * ib32 + l] >> 9];
            for (int j = 0; j < 8; ++j)
                y[j] = db[l / 2] * (double) grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0 : 1.0);
            y += 8;
        }
    }
}

int main(void)
{
    const size_t rb = sizeof(blk17);
    blk17 *w = (blk17 *) malloc(rb * NROW * NBLK);
    float *x = (float *) malloc(sizeof(float) * NIN);
    in2_q8k_blk *aq = (in2_q8k_blk *) malloc(sizeof(in2_q8k_blk) * NBLK);
    float *got = (float *) malloc(sizeof(float) * NROW);
    double *y = (double *) malloc(sizeof(double) * QKK);
    if (!w || !x || !aq || !got || !y) { fprintf(stderr, "oom\n"); return 2; }

    for (int i = 0; i < NIN; ++i) x[i] = ((int)(nx() % 2001) - 1000) / 1000.0f;
    for (int o = 0; o < NROW; ++o)
        for (int b = 0; b < NBLK; ++b) {
            blk17 *blk = w + ((size_t) o * NBLK + b);
            blk->d = in2_f32_to_f16(0.003f + 0.0005f * (float)(nx() % 8));
            for (int i = 0; i < QKK / 8; ++i) blk->qs[i] = (uint16_t)(nx() & 0xFFFFu);
            for (int i = 0; i < QKK / 32; ++i) blk->scales[i] = (uint8_t)(nx() & 0xFFu);
        }
    in2_q8k_quantise(x, NBLK, aq);

    double ref[NROW], noise[NROW];
    for (int o = 0; o < NROW; ++o) {
        double acc = 0.0; float accf = 0.0f;
        const blk17 *row = w + (size_t) o * NBLK;
        for (int b = 0; b < NBLK; ++b) {
            deq_iq2xs(&row[b], y);
            const double ad = (double) aq[b].d;
            for (int i = 0; i < QKK; ++i) {
                const double t = y[i] * (ad * (double) aq[b].qs[i]);
                acc  += t;
                accf  = accf + (float) t;
            }
        }
        ref[o] = acc;
        noise[o] = fabs((double) accf - acc) / fmax(fabs(acc), 1e-12);
    }
    double nmax = noise[0];
    for (int o = 1; o < NROW; ++o) if (noise[o] > nmax) nmax = noise[o];

    for (int o = 0; o < NROW; ++o) {
        const void *actp[1] = { aq }; float *op[1] = { got + o };
        wubu_iq_rows(17, (const void *) ((const unsigned char *) w + (size_t) o * rb * NBLK), rb, NIN, actp, 1, op, 0, 1);
    }

    double err[NROW];
    for (int o = 0; o < NROW; ++o)
        err[o] = fabs((double) got[o] - ref[o]) / fmax(fabs(ref[o]), 1e-12);
    for (int i = 1; i < NROW; ++i) { double v = err[i]; int j = i - 1;
        while (j >= 0 && err[j] > v) { err[j + 1] = err[j]; --j; } err[j + 1] = v; }

    printf("IQ2_XS reference check (ggml_type 17): %d rows x %d elements (%d superblocks), blocksize=%zu\n",
           NROW, NIN, NBLK, rb);
    printf("  rel err min %.3e  median %.3e  max %.3e\n", err[0], err[NROW / 2], err[NROW - 1]);
    printf("  sample ref[0]=%.6f got[0]=%.6f | ref[1]=%.6f got[1]=%.6f\n", ref[0], got[0], ref[1], got[1]);
    printf("  float32 sequential-noise floor max %.3e\n", nmax);
    const int pass = (err[NROW - 1] <= 8.0 * nmax);
    printf("  %s (kernel err %.3e vs 8x float32 noise floor %.3e)\n",
           pass ? "PASS" : "FAIL", err[NROW - 1], 8.0 * nmax);
    free(w); free(x); free(aq); free(got); free(y);
    return pass ? 0 : 1;
}
