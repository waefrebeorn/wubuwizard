/* Independent numerical reference for the IQ3_XXS row kernel (ggml_type 18) - 3 ATTENTION tensors, 3.0625 bpw.
 * The LAST unverified quantised format in this model.
 *
 * Copied VERBATIM from upstream dequantize_row_iq3_xxs. Hazards specific to this format, all of which a from-memory version
 * gets wrong: qs holds 96 bytes of which the first QK_K/4 (=64) are grid indices and the last 32 are interleaved
 * scales-and-signs; the 4-bit scale lives in the TOP 4 BITS of a uint32 read from those 32 bytes; and 7-bit sign patterns are
 * extracted at 7*l shifts, not 8*l. Two grid lookups per 8 elements (grid1 from qs[2l], grid2 from qs[2l+1]), with grid2
 * filling elements 4..7 using sign mask bits j+4.
 *
 * Layout asserted at 98 bytes: d(2) + qs(96). ACTIVATION: block_q8_K.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include "wubu_iq2q8.h"
#include "wubu_iq_c11.h"
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

void wubu_iq_rows(int ggml_type, const void *w, size_t row_bytes, int n,
                  const void *const *act, int nt, float *const *out, int r0, int r1);
double in2_hires_now(void) { return 0.0; }
double in2_ty_sec[256]; long in2_ty_calls[256]; int in2_ty_par = 0; int in2_ty_depth = 0;

#define QKK   256
#define NBLK  16
#define NIN   (QKK * NBLK)
#define NROW  8
typedef struct { uint16_t d; uint8_t qs[3 * QKK / 8]; } blk18;
static_assert(sizeof(blk18) == 98, "IQ3_XXS block must be 98 bytes: d(2) + qs(96)");

static uint32_t rs = 313131u;
static uint32_t nx(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

/* verbatim structure of upstream dequantize_row_iq3_xxs for one superblock */
static void deq_iq3xxs(const blk18 *x, double *y)
{
    const double d = (double) in2_f16_to_f32(x->d);
    const uint8_t *qs = x->qs;
    const uint8_t *scales_and_signs = x->qs + QKK / 4;
    uint32_t aux32;
    for (int ib32 = 0; ib32 < QKK / 32; ++ib32) {
        memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
        const double db = d * (0.5 + (double) (aux32 >> 28)) * 0.5;
        for (int l = 0; l < 4; ++l) {
            const uint8_t  signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const uint8_t *grid1 = (const uint8_t *) (iq3xxs_grid + qs[2 * l + 0]);
            const uint8_t *grid2 = (const uint8_t *) (iq3xxs_grid + qs[2 * l + 1]);
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db * (double) grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.0 : 1.0);
                y[j + 4] = db * (double) grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.0 : 1.0);
            }
            y += 8;
        }
        qs += 8;
    }
}

int main(void)
{
    const size_t rb = sizeof(blk18);
    blk18 *w = (blk18 *) malloc(rb * NROW * NBLK);
    float *x = (float *) malloc(sizeof(float) * NIN);
    in2_q8k_blk *aq = (in2_q8k_blk *) malloc(sizeof(in2_q8k_blk) * NBLK);
    float *got = (float *) malloc(sizeof(float) * NROW);
    double *y = (double *) malloc(sizeof(double) * QKK);
    if (!w || !x || !aq || !got || !y) { fprintf(stderr, "oom\n"); return 2; }

    for (int i = 0; i < NIN; ++i) x[i] = ((int)(nx() % 2001) - 1000) / 1000.0f;
    for (int o = 0; o < NROW; ++o)
        for (int b = 0; b < NBLK; ++b) {
            blk18 *blk = w + ((size_t) o * NBLK + b);
            blk->d = in2_f32_to_f16(0.0022f + 0.0004f * (float)(nx() % 8));
            for (int i = 0; i < 3 * QKK / 8; ++i) blk->qs[i] = (uint8_t)(nx() & 0xFFu);
        }
    in2_q8k_quantise(x, NBLK, aq);

    double ref[NROW], noise[NROW];
    for (int o = 0; o < NROW; ++o) {
        double acc = 0.0; float accf = 0.0f;
        const blk18 *row = w + (size_t) o * NBLK;
        for (int b = 0; b < NBLK; ++b) {
            deq_iq3xxs(&row[b], y);
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
        wubu_iq_rows(18, (const void *) ((const unsigned char *) w + (size_t) o * rb * NBLK), rb, NIN, actp, 1, op, 0, 1);
    }

    double err[NROW];
    for (int o = 0; o < NROW; ++o)
        err[o] = fabs((double) got[o] - ref[o]) / fmax(fabs(ref[o]), 1e-12);
    for (int i = 1; i < NROW; ++i) { double v = err[i]; int j = i - 1;
        while (j >= 0 && err[j] > v) { err[j + 1] = err[j]; --j; } err[j + 1] = v; }

    printf("IQ3_XXS reference check (ggml_type 18): %d rows x %d elements (%d superblocks), blocksize=%zu\n",
           NROW, NIN, NBLK, rb);
    printf("  rel err min %.3e  median %.3e  max %.3e\n", err[0], err[NROW / 2], err[NROW - 1]);
    printf("  sample ref[0]=%.6f got[0]=%.6f\n", ref[0], got[0]);
    printf("  float32 sequential-noise floor max %.3e\n", nmax);
    const int pass = (err[NROW - 1] <= 8.0 * nmax);
    printf("  %s (kernel err %.3e vs 8x float32 noise floor %.3e)\n",
           pass ? "PASS" : "FAIL", err[NROW - 1], 8.0 * nmax);
    free(w); free(x); free(aq); free(got); free(y);
    return pass ? 0 : 1;
}
