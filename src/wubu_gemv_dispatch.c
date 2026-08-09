/*
 * wubu_gemv_dispatch.c -- the PER-FAMILY GEMV DISPATCH (see header).
 * The precision ladder EXECUTES, it is not paper.
 */
#include "wubu_gemv_dispatch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_gemm.h"
#include "wubu_bf16_gemv.h"

int wubu_disp_init(wubu_gemv_dispatch_t *d, const wubu_precision_plan_t *plan)
{
    if (!d || !plan) return -1;
    memset(d, 0, sizeof(*d));
    for (int f = 0; f < WUBU_FAM_COUNT; f++) {
        d->bits[f] = plan->bits[f];
        d->q8[f] = NULL; d->qscale[f] = NULL;
        d->q_rows[f] = d->q_cols[f] = 0;
    }
    d->last_backend = WUBU_DISP_F32;
    return 0;
}

int wubu_disp_reladder(wubu_gemv_dispatch_t *d,
                       const wubu_precision_plan_t *plan)
{
    if (!d || !plan) return -1;
    for (int f = 0; f < WUBU_FAM_COUNT; f++) {
        if (d->bits[f] != plan->bits[f]) {
            /* the ladder changed: the next GEMV re-quantizes */
            free(d->q8[f]); d->q8[f] = NULL;
            free(d->qscale[f]); d->qscale[f] = NULL;
            d->q_rows[f] = d->q_cols[f] = 0;
            d->bits[f] = plan->bits[f];
        }
    }
    return 0;
}

/* the quantized GEMV (INT8, per-row scale — wubu_gemm's i8 path) */
static wubu_disp_backend_t gemv_i8(wubu_gemv_dispatch_t *d,
                                   wubu_family_t family,
                                   const float *W, const float *x,
                                   float *y, int n_out, int n_in)
{
    /* re-quantize when the shape/bits changed */
    if (!d->q8[family] || d->q_rows[family] != n_out || d->q_cols[family] != n_in) {
        free(d->q8[family]); free(d->qscale[family]);
        d->q8[family] = (int8_t *)malloc((size_t)n_out * n_in);
        d->qscale[family] = (float *)malloc((size_t)n_out * sizeof(float));
        if (!d->q8[family] || !d->qscale[family]) return WUBU_DISP_F32;
        wubu_gemv_quantize_i8(W, d->q8[family], d->qscale[family], n_out, n_in);
        d->q_rows[family] = n_out; d->q_cols[family] = n_in;
    }
    wubu_gemv_i8(d->q8[family], d->qscale[family], x, y, n_out, n_in);
    d->n_i8++;
    return WUBU_DISP_I8;
}

wubu_disp_backend_t wubu_disp_gemv(wubu_gemv_dispatch_t *d,
                                   wubu_family_t family,
                                   const float *W, const float *x,
                                   float *y, int n_out, int n_in)
{
    if (!d || !W || !x || !y) return WUBU_DISP_F32;
    int bits = d->bits[family];
    wubu_disp_backend_t be;
    if (bits >= 16) {
        /* >=16 bits: the F32 reference (or the BF16 kernel when the
         * hardware has it — the agnostic ladder's top rung) */
        int used_bf16 = 0;
        wubu_bf16_gemv(W, x, y, n_out, n_in, &used_bf16);
        if (used_bf16) { d->n_bf16++; be = WUBU_DISP_BF16; }
        else { wubu_gemv_f32(W, x, y, n_out, n_in); d->n_f32++; be = WUBU_DISP_F32; }
    } else {
        /* <16 bits: the INT8 path — the Escha filtering (gate/up at 2-3
         * bits is filtered downstream; here the ladder EXECUTES as the
         * nearest representable kernel) */
        be = gemv_i8(d, family, W, x, y, n_out, n_in);
    }
    d->last_backend = be;
    return be;
}

void wubu_disp_free(wubu_gemv_dispatch_t *d)
{
    if (!d) return;
    for (int f = 0; f < WUBU_FAM_COUNT; f++) {
        free(d->q8[f]);
        free(d->qscale[f]);
        d->q8[f] = NULL; d->qscale[f] = NULL;
    }
    memset(d, 0, sizeof(*d));
}

void wubu_disp_stats(const wubu_gemv_dispatch_t *d, char *buf, size_t cap)
{
    if (!d || !buf || cap == 0) return;
    snprintf(buf, cap,
             "f32=%llu i8=%llu bf16=%llu last=%d ladder[gate/up,down,dense,norm]=%d/%d/%d/%d",
             (unsigned long long)d->n_f32,
             (unsigned long long)d->n_i8,
             (unsigned long long)d->n_bf16,
             (int)d->last_backend,
             d->bits[WUBU_FAM_GATE_UP], d->bits[WUBU_FAM_DOWN],
             d->bits[WUBU_FAM_DENSE], d->bits[WUBU_FAM_NORM]);
}
