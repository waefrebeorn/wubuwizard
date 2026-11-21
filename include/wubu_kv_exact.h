/*
 * wubu_kv_exact.h -- exact-ID KV page allocation with a free-ID list.
 *
 * A page owns `capacity` numbered slots [0, capacity). The set of slots that
 * are currently free is kept as an explicit free-ID list. A request names the
 * EXACT ids it wants; the allocator either grants all of them or none.
 *
 * Contract:
 *   - can_alloc_exact() is a pure query: it validates the request and checks
 *     that every requested slot is available, without touching any state.
 *   - alloc_exact() is atomic: it builds a complete replacement free list in
 *     scratch memory and only swaps it in after every check succeeds. On any
 *     failure the allocator is left BYTE-IDENTICAL to its pre-call state.
 *   - status is returned as a value; errno is never consulted.
 *
 * Opaque struct, strict C11, no third-party dependencies.
 */
#ifndef WUBU_KV_EXACT_H
#define WUBU_KV_EXACT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_kv_exact wubu_kv_exact_t;

/* Status-as-value. OK == 0; every failure is a distinct positive code. */
typedef enum {
    WUBU_KV_EXACT_OK           = 0,
    WUBU_KV_EXACT_ERR_ARG      = 1,  /* NULL pointer / n < 0 */
    WUBU_KV_EXACT_ERR_BAD_ID   = 2,  /* negative id or id >= capacity */
    WUBU_KV_EXACT_ERR_DUP_ID   = 3,  /* a requested id appears twice */
    WUBU_KV_EXACT_ERR_QUOTA    = 4,  /* n exceeds the per-call quota */
    WUBU_KV_EXACT_ERR_NO_CAPACITY = 5, /* n exceeds the number of free slots */
    WUBU_KV_EXACT_ERR_NOT_FREE = 6,  /* alloc: a named slot is in use */
    WUBU_KV_EXACT_ERR_OOM      = 7,  /* scratch allocation failed */
    WUBU_KV_EXACT_ERR_NOT_IN_USE = 8 /* release: a named slot is already free */
} wubu_kv_exact_status_t;

/* Create a page with all `capacity` slots free and a per-call limit `quota`.
 * capacity and quota must be > 0. Returns NULL on invalid args or OOM. */
wubu_kv_exact_t *wubu_kv_exact_create(int capacity, int quota);
void wubu_kv_exact_free(wubu_kv_exact_t *a);

/* Pure validation + availability check. No mutation, no allocation. */
wubu_kv_exact_status_t wubu_kv_exact_can_alloc_exact(const wubu_kv_exact_t *a,
                                                     const int *ids, int n);

/* Atomic exact allocation. On failure the object is byte-identical. */
wubu_kv_exact_status_t wubu_kv_exact_alloc_exact(wubu_kv_exact_t *a,
                                                 const int *ids, int n);

/* Atomic release of previously allocated ids (same failure guarantees). */
wubu_kv_exact_status_t wubu_kv_exact_release(wubu_kv_exact_t *a,
                                             const int *ids, int n);

/* Introspection. */
int wubu_kv_exact_capacity(const wubu_kv_exact_t *a);
int wubu_kv_exact_quota(const wubu_kv_exact_t *a);
int wubu_kv_exact_free_count(const wubu_kv_exact_t *a);
int wubu_kv_exact_is_free(const wubu_kv_exact_t *a, int id);

/* Deterministic state serialization for byte-identical assertions.
 * snapshot_size() returns the number of bytes snapshot() will write.
 * snapshot() writes exactly that many bytes into `out`. */
size_t wubu_kv_exact_snapshot_size(const wubu_kv_exact_t *a);
void wubu_kv_exact_snapshot(const wubu_kv_exact_t *a, unsigned char *out);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KV_EXACT_H */
