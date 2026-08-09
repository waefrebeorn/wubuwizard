/*
 * wubu_gemv_dispatch.h -- PER-FAMILY GEMV DISPATCH ON THE PRECISION
 * LADDER (AN47 #2: "bind structure/compute/precision morphs to actual
 * GEMV backends so diagnose-driven precision changes are not
 * paper-only"). C11.
 *
 * The precision plan says the per-family bit ladder (gate/up = 2,
 * down = 3, dense = 8, norms = 32). The dispatch layer makes that
 * LADDER EXECUTE: a matrix family + the plan's bits -> the real GEMV
 * backend (F32, INT8, BF16). When the diagnose changes the plan, the
 * NEXT call on that family actually runs the new backend.
 *
 * Pure C11, opaque-free, reuses wubu_gemm + wubu_bf16_gemv.
 */
#ifndef WUBU_GEMV_DISPATCH_H
#define WUBU_GEMV_DISPATCH_H

#include "wubu_precision_plan.h"

/* the executed backend (what actually ran) */
typedef enum {
    WUBU_DISP_F32 = 0,      /* the f32 reference */
    WUBU_DISP_I8,           /* int8 quantized (per-row scale) */
    WUBU_DISP_BF16,         /* AVX512-BF16 when available */
    WUBU_DISP_COUNT
} wubu_disp_backend_t;

/* the dispatch state: the per-family precision bits (the ladder) */
typedef struct {
    int bits[WUBU_FAM_COUNT];   /* the EXECUTING ladder (bits/value) */
    /* the quantized weight cache (one i8 buffer per family — the
     * ladder change re-quantizes on the next call) */
    int8_t *q8[WUBU_FAM_COUNT];
    float  *qscale[WUBU_FAM_COUNT];
    int    q_rows[WUBU_FAM_COUNT], q_cols[WUBU_FAM_COUNT];
    /* telemetry: what actually ran */
    wubu_disp_backend_t last_backend;
    uint64_t n_f32, n_i8, n_bf16;
} wubu_gemv_dispatch_t;

/* D1: init with a precision plan's ladder. */
int wubu_disp_init(wubu_gemv_dispatch_t *d, const wubu_precision_plan_t *plan);

/* D2: RE-LADDER — the diagnose changed the plan: the next GEMV on
 * each family re-quantizes under the new bits. Returns 0. */
int wubu_disp_reladder(wubu_gemv_dispatch_t *d,
                       const wubu_precision_plan_t *plan);

/* D3: the executed GEMV: family + the FP32 weight + the input ->
 * the real backend. The ladder decides: >=16 bits -> F32, 8 bits ->
 * INT8 (quantized once), <8 bits -> the low-bit path (INT8 with the
 * family's scale, the Escha filtering). Returns the backend that ran. */
wubu_disp_backend_t wubu_disp_gemv(wubu_gemv_dispatch_t *d,
                                   wubu_family_t family,
                                   const float *W, const float *x,
                                   float *y, int n_out, int n_in);

/* D4: free the quantized caches. */
void wubu_disp_free(wubu_gemv_dispatch_t *d);

/* D5: the dispatch stats (what actually ran). */
void wubu_disp_stats(const wubu_gemv_dispatch_t *d, char *buf, size_t cap);

#endif
