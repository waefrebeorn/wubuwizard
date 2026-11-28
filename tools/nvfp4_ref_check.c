/* Cross-check for the NVFP4 row kernel (ggml_type 40) - 7 EXPERT tensors.
 *
 * NVFP4 is THIS PROJECT'S OWN format, so there is no upstream oracle. The reference here is the project's independent
 * dequantiser dequantize_row_nvfp4() in src/wubu_dequant_fp4.c - a different file, a different author's code path, and a
 * completely different implementation strategy (scalar table lookup) from the kernel's AVX2 nibble shuffle. Agreement
 * between them is a genuine cross-check between two implementations, not self-consistency.
 *
 * Layout (36 bytes / 64 elements):
 *   scales[4]  one UE4M3 scale per 16 elements
 *   qs[32]     64 x 4-bit codes; within a 16-element group, low nibbles are elements 0..7 and high nibbles 8..15
 * ACTIVATION: block_q8_0 (32 elements) - ty=40 is in the q8_0_act group, not the q8_K group.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include "wubu_iq2q8.h"
#include "wubu_iq_c11.h"
#include "wubu_nvfp4.h"

void wubu_iq_rows(int ggml_type, const void *w, size_t row_bytes, int n,
                  const void *const *act, int nt, float *const *out, int r0, int r1);
void dequantize_row_nvfp4(const unsigned char *data, float *output, long n_elems);
double in2_hires_now(void) { return 0.0; }
double in2_ty_sec[256]; long in2_ty_calls[256]; int in2_ty_par = 0; int in2_ty_depth = 0;

#define QKN   64
#define NBLK  32                 /* 64-element blocks per row */
#define NIN   (QKN * NBLK)       /* 2048 */
#define NROW  8
typedef struct { uint8_t scales[4]; uint8_t qs[32]; } blk40;
static_assert(sizeof(blk40) == 36, "NVFP4 block must be 36 bytes: scales(4) + qs(32)");

static uint32_t rs = 192837u;
static uint32_t nx(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

int main(void)
{
    const size_t rb = sizeof(blk40) * NBLK;
    unsigned char *w = (unsigned char *) malloc(rb * NROW);
    float *x = (float *) malloc(sizeof(float) * NIN);
    in2_q8_0_blk *aq = (in2_q8_0_blk *) malloc(sizeof(in2_q8_0_blk) * (NIN / 32));
    float *dq = (float *) malloc(sizeof(float) * NIN);
    float *got = (float *) malloc(sizeof(float) * NROW);
    if (!w || !x || !aq || !dq || !got) { fprintf(stderr, "oom\n"); return 2; }

    for (int i = 0; i < NIN; ++i) x[i] = ((int)(nx() % 2001) - 1000) / 1000.0f;
    for (int o = 0; o < NROW; ++o)
        for (int b = 0; b < NBLK; ++b) {
            blk40 *blk = (blk40 *) (w + (size_t) o * rb) + b;
            for (int s = 0; s < 4; ++s) blk->scales[s] = (uint8_t)(nx() & 0x7Fu);  /* positive UE4M3 range */
            for (int i = 0; i < 32; ++i) blk->qs[i] = (uint8_t)(nx() & 0xFFu);
        }
    in2_q8_0_quantise(x, NIN / 32, aq);

    double ref[NROW], noise[NROW];
    for (int o = 0; o < NROW; ++o) {
        dequantize_row_nvfp4(w + (size_t) o * rb, dq, NIN);   /* project's own independent dequantiser */
        double acc = 0.0; float accf = 0.0f;
        for (int b = 0; b < NIN / 32; ++b) {
            const double ad = (double) in2_f16_to_f32(aq[b].d);
            for (int i = 0; i < 32; ++i) {
                const double t = (double) dq[b * 32 + i] * (ad * (double) aq[b].qs[i]);
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
        wubu_iq_rows(40, (const void *) (w + (size_t) o * rb), rb, NIN, actp, 1, op, 0, 1);
    }

    double err[NROW];
    for (int o = 0; o < NROW; ++o)
        err[o] = fabs((double) got[o] - ref[o]) / fmax(fabs(ref[o]), 1e-12);
    for (int i = 1; i < NROW; ++i) { double v = err[i]; int j = i - 1;
        while (j >= 0 && err[j] > v) { err[j + 1] = err[j]; --j; } err[j + 1] = v; }

    printf("NVFP4 cross-check (ggml_type 40): %d rows x %d elements (%d blocks of %d), row_bytes=%zu\n",
           NROW, NIN, NBLK, QKN, rb);
    printf("  rel err min %.3e  median %.3e  max %.3e\n", err[0], err[NROW / 2], err[NROW - 1]);
    printf("  sample ref[0]=%.6f got[0]=%.6f | ref[1]=%.6f got[1]=%.6f\n", ref[0], got[0], ref[1], got[1]);
    printf("  float32 sequential-noise floor max %.3e\n", nmax);
    const int pass = (err[NROW - 1] <= 8.0 * nmax);
    printf("  %s (kernel err %.3e vs 8x float32 noise floor %.3e)\n",
           pass ? "PASS" : "FAIL", err[NROW - 1], 8.0 * nmax);
    free(w); free(x); free(aq); free(dq); free(got);
    return pass ? 0 : 1;
}
