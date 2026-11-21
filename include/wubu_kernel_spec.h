#ifndef WUBU_KERNEL_SPEC_H
#define WUBU_KERNEL_SPEC_H

/* wubu_kernel_spec.h — LP-10 compile-time kernel family specialization.
 *
 * A model family is NOT a branch inside a kernel body. Instead each family is
 * a compile-time data row keyed by a family name: the row's kernel-body
 * descriptor (attention variant, rotary pairs, window tokens, MoE routing,
 * stop-token list) selects which specialized kernel body runs. Kernel bodies
 * contain no model-family branching; the family key is only a selector.
 *
 * The identity hash is a stable FNV-1a 64-bit value computed over the
 * kernel-body descriptor alone. It is a pure function of those descriptor
 * bytes: it never reads the family key, the row index, or the number of
 * registered rows. Therefore registering a new family row leaves every
 * existing kernel-body identity byte-identical, which is the LP-10 gate.
 *
 * WaefreBeorn Umbrella License v3.0
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_KERNEL_SPEC_MAX_FAMILY  32  /* family key length incl. NUL */
#define WUBU_KERNEL_SPEC_MAX_STOP    32  /* stop tokens per body */
#define WUBU_KERNEL_SPEC_MAX_ROWS    64  /* registry capacity */

/* Attention variants: one specialized kernel body per value. */
typedef enum wubu_attn_variant {
    WUBU_ATTN_GQA    = 0,
    WUBU_ATTN_MHA    = 1,
    WUBU_ATTN_MLA    = 2,
    WUBU_ATTN_LINEAR = 3,
    WUBU_ATTN_SSM    = 4
} wubu_attn_variant_t;

/* MoE routing modes. WUBU_MOE_NONE means a dense FFN. */
typedef enum wubu_moe_route {
    WUBU_MOE_NONE = 0,
    WUBU_MOE_TOP1 = 1,
    WUBU_MOE_TOP2 = 2,
    WUBU_MOE_TOP8 = 8
} wubu_moe_route_t;

/* Kernel-body descriptor: the complete compile-time specialization.
 * Integers are serialized in fixed little-endian order for hashing so the
 * identity is independent of host byte order. */
typedef struct wubu_kernel_body {
    wubu_attn_variant_t attn;
    int                rotary_pairs;   /* 0 = no rotary embedding */
    int                window_tokens;  /* 0 = full attention */
    wubu_moe_route_t   moe;
    int                stop_tokens[WUBU_KERNEL_SPEC_MAX_STOP];
    size_t             n_stop_tokens;  /* <= WUBU_KERNEL_SPEC_MAX_STOP */
} wubu_kernel_body_t;

/* A family row: family key -> specialized kernel body. Value type; callers
 * build these as compile-time static const data and register them. */
typedef struct wubu_kernel_row {
    char               family[WUBU_KERNEL_SPEC_MAX_FAMILY];
    wubu_kernel_body_t body;
} wubu_kernel_row_t;

/* Opaque registry. Layout is private to wubu_kernel_spec.c. */
typedef struct wubu_kernel_spec wubu_kernel_spec_t;

wubu_kernel_spec_t *wubu_kernel_spec_create(void);
void wubu_kernel_spec_free(wubu_kernel_spec_t *spec);

/* Register a family row (copied by value). Returns 1 on success, 0 if the
 * registry is full, the key is empty/too long, the row count is invalid, or
 * the family key is already registered. */
int wubu_kernel_spec_register(wubu_kernel_spec_t *spec,
                              const wubu_kernel_row_t *row);

/* Select the row registered under `family`. Returns a borrowed pointer to
 * the internal copy (valid until free), or NULL when the family is unknown. */
const wubu_kernel_row_t *wubu_kernel_spec_select(const wubu_kernel_spec_t *spec,
                                                  const char *family);

/* Number of registered rows. */
size_t wubu_kernel_spec_count(const wubu_kernel_spec_t *spec);

/* Stable FNV-1a 64-bit identity over a kernel-body descriptor. Pure function
 * of `body` bytes; two equal descriptors always yield the same value and a
 * family row added elsewhere never changes it. */
uint64_t wubu_kernel_body_identity(const wubu_kernel_body_t *body);

/* Identity of the body registered under `family`, or 0 if unknown. */
uint64_t wubu_kernel_spec_identity(const wubu_kernel_spec_t *spec,
                                   const char *family);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KERNEL_SPEC_H */
