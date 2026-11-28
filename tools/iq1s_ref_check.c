/* Independent numerical reference for the IQ1_S row kernel (ggml_type 19) - 6 EXPERT tensors, 1.5625 bpw.
 *
 * Dequantisation copied VERBATIM from upstream llama.cpp dequantize_row_iq1_s; grid and IQ1S_DELTA come from
 * ggml-common.h. Block is d(2) + qs[32] + qh[8]*2 = 50 bytes, asserted rather than assumed (a wrong block size produced a
 * NaN result and a wasted investigation once already in this repo).
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
typedef struct { uint16_t d; uint8_t qs[QKK / 8]; uint16_t qh[QKK / 32]; } blk19;
static_assert(sizeof(blk19) == 50, "IQ1_S block must be 50 bytes: d(2) + qs(32) + qh(16)");

static uint32_t rs = 20260101u;
static uint32_t nx(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

/* verbatim structure of upstream dequantize_row_iq1_s for one superblock */
static void deq_iq1s(const blk19 *x, double *y)
{
    const double d = (double) in2_f16_to_f32(x->d);
    const uint8_t  *qs = x->qs;
    const uint16_t *qh = x->qh;
    for (int ib = 0; ib < QKK / 32; ++ib) {
        const double dl = d * (double) (2 * ((qh[ib] >> 12) & 7) + 1);
        const double delta = (qh[ib] & 0x8000) ? -(double) IQ1S_DELTA : (double) IQ1S_DELTA;
        for (int l = 0; l < 4; ++l) {
            const int8_t *grid = (const int8_t *) (iq1s_grid + (qs[l] | (((qh[ib] >> (3 * l)) & 7) << 8)));
            for (int j = 0; j < 8; ++j) y[j] = dl * ((double) grid[j] + delta);
            y += 8;
        }
        qs += 4;
    }
}

int main(void)
{
    const size_t rb = sizeof(blk19);
    blk19 *w = (blk19 *) malloc(rb * NROW * NBLK);
    float *x = (float *) malloc(sizeof(float) * NIN);
    in2_q8k_blk *aq = (in2_q8k_blk *) malloc(sizeof(in2_q8k_blk) * NBLK);
    float *got = (float *) malloc(sizeof(float) * NROW);
    double *y = (double *) malloc(sizeof(double) * QKK);
    if (!w || !x || !aq || !got || !y) { fprintf(stderr, "oom\n"); return 2; }

    for (int i = 0; i < NIN; ++i) x[i] = ((int)(nx() % 2001) - 1000) / 1000.0f;
    for (int o = 0; o < NROW; ++o)
        for (int b = 0; b < NBLK; ++b) {
            blk19 *blk = w + ((size_t) o * NBLK + b);
            blk->d = in2_f32_to_f16(0.004f + 0.001f * (float)(nx() % 8));
            for (int i = 0; i < QKK / 8; ++i) blk->qs[i] = (uint8_t)(nx() & 0xFFu);
            for (int i = 0; i < QKK / 32; ++i) blk->qh[i] = (uint16_t)(nx() & 0xFFFFu);
        }
    in2_q8k_quantise(x, NBLK, aq);

    double ref[NROW], noise[NROW];
    for (int o = 0; o < NROW; ++o) {
        double acc = 0.0; float accf = 0.0f;
        const blk19 *row = w + (size_t) o * NBLK;
        for (int b = 0; b < NBLK; ++b) {
            deq_iq1s(&row[b], y);
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
        wubu_iq_rows(19, (const void *) ((const unsigned char *) w + (size_t) o * rb * NBLK), rb, NIN, actp, 1, op, 0, 1);
    }

    double err[NROW];
    for (int o = 0; o < NROW; ++o)
        err[o] = fabs((double) got[o] - ref[o]) / fmax(fabs(ref[o]), 1e-12);
    for (int i = 1; i < NROW; ++i) { double v = err[i]; int j = i - 1;
        while (j >= 0 && err[j] > v) { err[j + 1] = err[j]; --j; } err[j + 1] = v; }

    printf("IQ1_S reference check (ggml_type 19): %d rows x %d elements (%d superblocks), blocksize=%zu\n",
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
