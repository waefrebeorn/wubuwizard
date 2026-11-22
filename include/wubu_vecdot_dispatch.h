#ifndef WUBU_VECDOT_DISPATCH_H
#define WUBU_VECDOT_DISPATCH_H

/*
 * wubu_vecdot_dispatch.h -- EG-01: runtime per-type vec_dot dispatch and
 * multi-row GEMV. The underlying kernels live in quantized_dot_generic.c;
 * this module owns only the dispatch seam and the row loop.
 *
 * Unknown quantization types fail closed: no table entry means no kernel.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*wubu_vecdot_fn)(int n, float *s, size_t bs,
                               const void *vx, size_t bx,
                               const void *vy, size_t by, int nrc);

typedef struct {
    int          ggml_type;      /* GGML_TYPE_* value */
    const char  *name;           /* kernel family label */
    wubu_vecdot_fn fn;           /* owning kernel */
    int          nrows_per_call; /* rows the kernel handles natively */
} wubu_vecdot_entry_t;

/* Table access. Returns NULL for unknown types. */
const wubu_vecdot_entry_t *wubu_vecdot_lookup(int ggml_type);
int  wubu_vecdot_supported(int ggml_type);
int  wubu_vecdot_table_count(void);

/* Single row: out[0] = dot(weight row, activation row). */
int  wubu_vecdot_one(int ggml_type, int n, const void *w, const void *x,
                     float *out);

/* Multi-row GEMV: out[r] = dot(weight row, x_rows[r]) for r in [0,nr).
 * The kernel is dispatched ONCE for the whole batch; lookups_out (optional)
 * receives the number of table lookups, so a gate can prove the batch path
 * does not re-dispatch per row. */
int  wubu_vecdot_rows(int ggml_type, int n, const void *w,
                      const void *const *x_rows, int nr, float *out,
                      int *lookups_out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_VECDOT_DISPATCH_H */
