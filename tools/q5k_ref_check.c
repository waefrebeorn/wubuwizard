/* Independent numerical reference for the Q5_K row kernel (ggml_type 13) - 16 ATTENTION tensors, 5.5 bpw.
 *
 * Copied VERBATIM from upstream dequantize_row_q5_K plus the shared get_scale_min_k4 helper. Q5_K extends Q4_K with a 5th
 * bit plane in qh, addressed by bit masks u1/u2 that shift left by 2 for each 64-element group (u1 = 1,4,16,64 and
 * u2 = 2,8,32,128). That shifting-mask detail is exactly what a from-memory implementation gets wrong.
 *
 * Layout asserted at 176 bytes: d(2) + dmin(2) + scales(12) + qh(32) + qs(128). ACTIVATION: block_q8_K.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include "wubu_iq2q8.h"
#include "wubu_iq_c11.h"

void wubu_iq_rows(int ggml_type, const void *w, size_t row_bytes, int n,
                  const void *const *act, int nt, float *const *out, int r0, int r1);
double in2_hires_now(void) { return 0.0; }
double in2_ty_sec[256]; long in2_ty_calls[256]; int in2_ty_par = 0; int in2_ty_depth = 0;

#define QKK   256
#define NBLK  16
#define NIN   (QKK * NBLK)
#define NROW  8
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } blk13;
static_assert(sizeof(blk13) == 176, "Q5_K block must be 176 bytes: d(2)+dmin(2)+scales(12)+qh(32)+qs(128)");

static uint32_t rs = 777333u;
static uint32_t nx(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

/* verbatim structure of upstream dequantize_row_q5_K for one superblock */
static void deq_q5k(const blk13 *x, double *y)
{
    const uint8_t *ql = x->qs;
    const uint8_t *qh = x->qh;
    const double d  = (double) in2_f16_to_f32(x->d);
    const double mn = (double) in2_f16_to_f32(x->dmin);
    int is = 0;
    uint8_t sc, m;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QKK; j += 64) {
        get_scale_min_k4(is + 0, x->scales, &sc, &m);
        const double d1 = d * (double) sc; const double m1 = mn * (double) m;
        get_scale_min_k4(is + 1, x->scales, &sc, &m);
        const double d2 = d * (double) sc; const double m2 = mn * (double) m;
        for (int l = 0; l < 32; ++l)
            *y++ = d1 * ((double) (ql[l] & 0xF) + ((qh[l] & u1) ? 16.0 : 0.0)) - m1;
        for (int l = 0; l < 32; ++l)
            *y++ = d2 * ((double) (ql[l] >> 4) + ((qh[l] & u2) ? 16.0 : 0.0)) - m2;
        ql += 32; is += 2;
        u1 <<= 2; u2 <<= 2;
    }
}

int main(void)
{
    const size_t rb = sizeof(blk13);
    blk13 *w = (blk13 *) malloc(rb * NROW * NBLK);
    float *x = (float *) malloc(sizeof(float) * NIN);
    in2_q8k_blk *aq = (in2_q8k_blk *) malloc(sizeof(in2_q8k_blk) * NBLK);
    float *got = (float *) malloc(sizeof(float) * NROW);
    double *y = (double *) malloc(sizeof(double) * QKK);
    if (!w || !x || !aq || !got || !y) { fprintf(stderr, "oom\n"); return 2; }

    for (int i = 0; i < NIN; ++i) x[i] = ((int)(nx() % 2001) - 1000) / 1000.0f;
    for (int o = 0; o < NROW; ++o)
        for (int b = 0; b < NBLK; ++b) {
            blk13 *blk = w + ((size_t) o * NBLK + b);
            blk->d    = in2_f32_to_f16(0.0018f + 0.0003f * (float)(nx() % 8));
            blk->dmin = in2_f32_to_f16(0.0009f + 0.0002f * (float)(nx() % 8));
            for (int i = 0; i < 12; ++i) blk->scales[i] = (uint8_t)(nx() & 0xFFu);
            for (int i = 0; i < 32; ++i) blk->qh[i] = (uint8_t)(nx() & 0xFFu);
            for (int i = 0; i < 128; ++i) blk->qs[i] = (uint8_t)(nx() & 0xFFu);
        }
    in2_q8k_quantise(x, NBLK, aq);

    double ref[NROW], noise[NROW];
    for (int o = 0; o < NROW; ++o) {
        double acc = 0.0; float accf = 0.0f;
        const blk13 *row = w + (size_t) o * NBLK;
        for (int b = 0; b < NBLK; ++b) {
            deq_q5k(&row[b], y);
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
        wubu_iq_rows(13, (const void *) ((const unsigned char *) w + (size_t) o * rb * NBLK), rb, NIN, actp, 1, op, 0, 1);
    }

    double err[NROW];
    for (int o = 0; o < NROW; ++o)
        err[o] = fabs((double) got[o] - ref[o]) / fmax(fabs(ref[o]), 1e-12);
    for (int i = 1; i < NROW; ++i) { double v = err[i]; int j = i - 1;
        while (j >= 0 && err[j] > v) { err[j + 1] = err[j]; --j; } err[j + 1] = v; }

    printf("Q5_K reference check (ggml_type 13): %d rows x %d elements (%d superblocks), blocksize=%zu\n",
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
