/* Decisive IQ2_XXS check: REAL weight bytes from the model file, not synthetic blocks.
 *
 * WHY THIS TEST EXISTS: the synthetic harness in tools/iq2xxs_ref_check.c mismatches the kernel by O(1), but the gate passes
 * 15/15 against the llama.cpp oracle on this same model, whose experts include 7 IQ2_XXS tensors. Those two facts cannot
 * both be "kernel is wrong", so the mismatch is most likely my synthetic block layout. Real bytes remove that possibility:
 * they cannot be misaligned by my struct choices.
 *
 * Geometry comes from the tensor itself: blk.5.ffn_gate_up_exps.weight is dims [2304, 2304, 128], so one row is 2304
 * elements = 9 superblocks of 256, and row_bytes = 9 * sizeof(block_iq2_xxs) = 9 * 66 = 594.
 *
 * Dequantisation is upstream's dequantize_row_iq2_xxs, copied verbatim. The activation is arbitrary but IDENTICAL on both
 * sides, so it cannot mask a difference.
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
#define NROW  16
#define NAVAL 256
#define NBLK  (NAVAL / QKK)     /* 9 superblocks per row */
typedef struct { uint16_t d; uint8_t qs[QKK / 4]; } blk16;

static uint32_t rs = 31337u;
static uint32_t nx(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

static void deq_iq2xxs(const blk16 *x, double *y)
{
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *) aux32;
    const double d = (double) in2_f16_to_f32(x->d);
    for (int ib32 = 0; ib32 < QKK / 32; ++ib32) {
        memcpy(aux32, x->qs + 8 * ib32, 2 * sizeof(uint32_t));   /* upstream indexes uint16_t*, so 4*ib32 there == 8*ib32 BYTES here */
        const double db = d * (double) (0.5f + (float) (aux32[1] >> 28)) * 0.25;
        for (int l = 0; l < 4; ++l) {
            const uint8_t *grid = (const uint8_t *) (iq2xxs_grid + aux8[l]);
            const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
            for (int j = 0; j < 8; ++j)
                y[j] = db * (double) grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0 : 1.0);
            y += 8;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <abs_offset>\n", argv[0]); return 2; }
    const long long off = atoll(argv[2]);
    const size_t rb = (size_t) NBLK * sizeof(blk16);
    unsigned char *raw = (unsigned char *) malloc(rb * NROW);
    float *x = (float *) malloc(sizeof(float) * NAVAL);
    in2_q8k_blk *aq = (in2_q8k_blk *) malloc(sizeof(in2_q8k_blk) * NBLK);
    float *got = (float *) malloc(sizeof(float) * NROW);
    double *y = (double *) malloc(sizeof(double) * NAVAL);
    if (!raw || !x || !aq || !got || !y) { fprintf(stderr, "oom\n"); return 2; }

    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { fprintf(stderr, "cannot open model\n"); return 2; }
    if (fseek(fp, (long) off, SEEK_SET) != 0) { fprintf(stderr, "seek failed\n"); return 2; }
    if (fread(raw, 1, rb * NROW, fp) != rb * NROW) { fprintf(stderr, "read failed (short)\n"); return 2; }
    fclose(fp);

    for (int i = 0; i < NAVAL; ++i) x[i] = ((int)(nx() % 2001) - 1000) / 1000.0f;
    in2_q8k_quantise(x, NBLK, aq);

    double ref[NROW], noise[NROW];
    for (int o = 0; o < NROW; ++o) {
        const blk16 *row = (const blk16 *) (raw + (size_t) o * rb);
        double acc = 0.0; float accf = 0.0f;
        for (int b = 0; b < NBLK; ++b) {
            deq_iq2xxs(&row[b], y + (size_t) b * QKK);
            const double ad = (double) aq[b].d;
            for (int i = 0; i < QKK; ++i) {
                const double t = y[(size_t) b * QKK + i] * (ad * (double) aq[b].qs[i]);
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
        wubu_iq_rows(16, (const void *) (raw + (size_t) o * rb), rb, NAVAL, actp, 1, op, 0, 1);
    }

    double err[NROW];
    for (int o = 0; o < NROW; ++o)
        err[o] = fabs((double) got[o] - ref[o]) / fmax(fabs(ref[o]), 1e-12);
    for (int i = 1; i < NROW; ++i) { double v = err[i]; int j = i - 1;
        while (j >= 0 && err[j] > v) { err[j + 1] = err[j]; --j; } err[j + 1] = v; }

    printf("IQ2_XXS REAL-DATA check: %d real rows x %d elements (row_bytes=%zu, offset=%lld)\n",
           NROW, NAVAL, rb, off);
    printf("  rel err min %.3e  median %.3e  max %.3e\n", err[0], err[NROW / 2], err[NROW - 1]);
    printf("  sample ref[0]=%.6f got[0]=%.6f | ref[1]=%.6f got[1]=%.6f\n", ref[0], got[0], ref[1], got[1]);
    printf("  float32 sequential-noise floor max %.3e\n", nmax);
    const int pass = (err[NROW - 1] <= 8.0 * nmax);
    printf("  %s (kernel err %.3e vs 8x float32 noise floor %.3e)\n",
           pass ? "PASS" : "FAIL", err[NROW - 1], 8.0 * nmax);
    free(raw); free(x); free(aq); free(got); free(y);
    return pass ? 0 : 1;
}
