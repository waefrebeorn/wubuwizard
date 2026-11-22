/*
 * wubu_vecdot_dispatch.c -- EG-01 runtime per-type vec_dot dispatch.
 *
 * The quant kernels in quantized_dot_generic.c select AVX2/SSE/generic with
 * compile-time #if inside each wrapper. This module makes the *type* choice
 * a runtime table lookup, so a runner can probe a GGUF type and ask whether
 * a kernel exists before touching weights. Unknown types are refused.
 *
 * The row path dispatches once and reuses the function pointer across all
 * activation rows: one weight row is shared by every token in the batch.
 */

#include "wubu_vecdot_dispatch.h"
#include "gguf_reader.h"

/* Kernels with external linkage in src/quantized_dot_generic.c. */
extern void q4_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx,
                         const void *vy, size_t by, int nrc);
extern void q5_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx,
                         const void *vy, size_t by, int nrc);
extern void q6_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx,
                         const void *vy, size_t by, int nrc);
extern void q8_0_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx,
                         const void *vy, size_t by, int nrc);
extern void iq2_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx,
                            const void *vy, size_t by, int nrc);
extern void iq3_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx,
                            const void *vy, size_t by, int nrc);
extern void iq4_xs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx,
                           const void *vy, size_t by, int nrc);

static const wubu_vecdot_entry_t kTable[] = {
    { GGML_TYPE_Q4_K,   "Q4_K",   q4_K_vec_dot,   1 },
    { GGML_TYPE_Q5_K,   "Q5_K",   q5_K_vec_dot,   1 },
    { GGML_TYPE_Q6_K,   "Q6_K",   q6_K_vec_dot,   1 },
    { GGML_TYPE_Q8_0,   "Q8_0",   q8_0_vec_dot,   1 },
    { GGML_TYPE_IQ2_XXS,"IQ2_XXS",iq2_xxs_vec_dot,1 },
    { GGML_TYPE_IQ3_XXS,"IQ3_XXS",iq3_xxs_vec_dot,1 },
    { GGML_TYPE_IQ4_XS, "IQ4_XS", iq4_xs_vec_dot, 1 },
};

const wubu_vecdot_entry_t *wubu_vecdot_lookup(int ggml_type) {
    for (size_t i = 0; i < sizeof(kTable) / sizeof(kTable[0]); i++)
        if (kTable[i].ggml_type == ggml_type) return &kTable[i];
    return NULL;
}

int wubu_vecdot_supported(int ggml_type) {
    return wubu_vecdot_lookup(ggml_type) != NULL;
}

int wubu_vecdot_table_count(void) {
    return (int)(sizeof(kTable) / sizeof(kTable[0]));
}

int wubu_vecdot_one(int ggml_type, int n, const void *w, const void *x,
                    float *out) {
    if (!out || n <= 0 || !w || !x) return -1;
    const wubu_vecdot_entry_t *e = wubu_vecdot_lookup(ggml_type);
    if (!e || !e->fn) return -1;
    out[0] = 0.0f; /* some kernels assign, Q8_0 accumulates */
    e->fn(n, out, 0, w, 0, x, 0, 1);
    return 0;
}

int wubu_vecdot_rows(int ggml_type, int n, const void *w,
                     const void *const *x_rows, int nr, float *out,
                     int *lookups_out) {
    if (!out || !x_rows || nr <= 0 || n <= 0 || !w) return -1;
    /* One table lookup for the whole batch is the point of this path. */
    const wubu_vecdot_entry_t *e = wubu_vecdot_lookup(ggml_type);
    if (lookups_out) *lookups_out = 1;
    if (!e || !e->fn) return -1;
    for (int r = 0; r < nr; r++) {
        if (!x_rows[r]) return -1;
        out[r] = 0.0f;
        e->fn(n, &out[r], 0, w, 0, x_rows[r], 0, 1);
    }
    return 0;
}
