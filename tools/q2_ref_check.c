/* Independent numerical reference for the Q2_0 row kernel (ggml_type 42) - the most common EXPERT weight type in the
 * IQ2_XXS model (17 expert tensors), i.e. on the 39% path.
 *
 * Convention taken from the kernel itself, NOT from wubu_turboquant.h (which is a different 32-element / bias-2 format):
 *   64 elements per block, qs[16], one fp16 d, weight = d * (q - 1) for q in 0..3.
 *   Lane order was determined empirically with tools/q2_lane_probe.c: index = 4*byte + shift (sequential, injective).
 *   The 64-element weight block pairs with TWO 32-element Q8_0 activation blocks: qs[0..7] with act[0], qs[8..15] with act[1].
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
double in2_hires_now(void) { return 0.0; }
double in2_ty_sec[256]; long in2_ty_calls[256]; int in2_ty_par = 0; int in2_ty_depth = 0;

#define QK2   64
#define NB2   32                     /* weight blocks */
#define NIN2  (QK2 * NB2)            /* 2048 elements per row */
#define NROW2 16
typedef struct { uint16_t d; uint8_t qs[16]; } blk42;

static uint32_t rs = 987654321u;
static uint32_t nx(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

int main(void)
{
    const size_t rb = sizeof(blk42);
    blk42 *w = (blk42 *) malloc(rb * NROW2 * NB2);
    float *x = (float *) malloc(sizeof(float) * NIN2);
    in2_q8_0_blk *aq = (in2_q8_0_blk *) malloc(sizeof(in2_q8_0_blk) * (NIN2 / 32));
    float *got = (float *) malloc(sizeof(float) * NROW2);
    if (!w || !x || !aq || !got) { fprintf(stderr, "oom\n"); return 2; }

    for (int i = 0; i < NIN2; ++i) x[i] = ((int)(nx() % 2001) - 1000) / 1000.0f;
    for (int o = 0; o < NROW2; ++o)
        for (int b = 0; b < NB2; ++b) {
            blk42 *blk = w + ((size_t) o * NB2 + b);
            blk->d = in2_f32_to_f16(0.002f + 0.0005f * (float)(nx() % 8));
            for (int i = 0; i < 16; ++i) blk->qs[i] = (uint8_t)(nx() & 0xFFu);
        }
    in2_q8_0_quantise(x, NIN2 / 32, aq);

    /* reference: the specification, in double, using the empirically confirmed lane order */
    double ref[NROW2];
    for (int o = 0; o < NROW2; ++o) {
        double acc = 0.0;
        const blk42 *row = w + (size_t) o * NB2;
        for (int b = 0; b < NB2; ++b) {
            const double wd = (double) in2_f16_to_f32(row[b].d);
            for (int half = 0; half < 2; ++half) {
                const in2_q8_0_blk *ab = &aq[b * 2 + half];
                const double ad = (double) in2_f16_to_f32(ab->d);
                int isum = 0;
                for (int j = 0; j < 32; ++j) {
                    /* element index within the block: 32*half + j  ->  byte = idx/4, shift = idx%4 */
                    const int idx = 32 * half + j;
                    const int q = (row[b].qs[idx / 4] >> (2 * (idx % 4))) & 0x3;
                    isum += (q - 1) * (int) ab->qs[j];
                }
                acc += wd * ad * (double) isum;
            }
        }
        ref[o] = acc;
    }

    for (int o = 0; o < NROW2; ++o) {
        const void *actp[1] = { aq }; float *op[1] = { got + o };
        wubu_iq_rows(42, (const void *)((const unsigned char *) w + (size_t) o * rb * NB2), rb, NIN2, actp, 1, op, 0, 1);
    }

    double err[NROW2];
    for (int o = 0; o < NROW2; ++o)
        err[o] = fabs((double) got[o] - ref[o]) / fmax(fabs(ref[o]), 1e-12);
    for (int i = 1; i < NROW2; ++i) { double v = err[i]; int j = i - 1;
        while (j >= 0 && err[j] > v) { err[j + 1] = err[j]; --j; } err[j + 1] = v; }

    printf("Q2_0 reference check (ggml_type 42, the HOT expert type): %d rows x %d elements (%d blocks of %d)\n",
           NROW2, NIN2, NB2, QK2);
    printf("  rel err min %.3e  median %.3e  max %.3e\n", err[0], err[NROW2 / 2], err[NROW2 - 1]);
    printf("  sample ref[0]=%.6f got[0]=%.6f | ref[1]=%.6f got[1]=%.6f\n", ref[0], got[0], ref[1], got[1]);
    const int pass = (err[NROW2 - 1] < 1e-5);
    printf("  %s (tolerance 1e-5)\n", pass ? "PASS" : "FAIL");
    free(w); free(x); free(aq); free(got);
    return pass ? 0 : 1;
}
