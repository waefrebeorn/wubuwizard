#ifndef WUBU_KV_FETCH_H
#define WUBU_KV_FETCH_H

/*
 * wubu_kv_fetch.h -- LP-12: tolerance-parameterised KV fetch for the 9P layer.
 *
 * The fast side keeps a sketch of every KV token plus a few pivotal tokens in
 * full precision. Before touching the host, an error estimate (derived only
 * from local data: sketch-vs-pivot residual weighted by the query) decides:
 *
 *   estimate <= eps   -> WUBU_KV_FETCH_NO      (sketch + pivots, no host call)
 *   estimate >= delta -> WUBU_KV_FETCH_FORCED  (full page must be host-fetched)
 *   otherwise         -> WUBU_KV_FETCH_UNCERTAIN (sketch used, flagged)
 *
 * Opaque module seam: only the typedef, callback type and functions are
 * public; all page state and scratch buffers stay in the .c file.
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_kv_fetch wubu_kv_fetch_t;

typedef enum {
    WUBU_KV_FETCH_NO = 0,
    WUBU_KV_FETCH_UNCERTAIN = 1,
    WUBU_KV_FETCH_FORCED = 2
} wubu_kv_fetch_decision_t;

/* Host page supplier. Fills out_k/out_v with n*d floats each. Return 0 on
 * success, nonzero on failure (the fetch is then aborted, not half-applied). */
typedef int (*wubu_kv_fetch_host_fn)(void *userdata, float *out_k, float *out_v,
                                     int n, int d);

wubu_kv_fetch_t *wubu_kv_fetch_create(int n, int d, float eps, float delta);
void wubu_kv_fetch_free(wubu_kv_fetch_t *f);

/* Install the local side: sketch keys/values [n*d], plus n_pivots pivotal
 * token indices and their exact keys/values [n_pivots*d]. Copies all inputs. */
int wubu_kv_fetch_set_local(wubu_kv_fetch_t *f,
                            const float *sketch_k, const float *sketch_v,
                            const int *pivot_idx,
                            const float *pivot_k, const float *pivot_v,
                            int n_pivots);

/* Local-only error estimate for this query. No host access. */
float wubu_kv_fetch_estimate(const wubu_kv_fetch_t *f, const float *query);

/* Decide and compute attention output [d]. Returns 0 on success. */
int wubu_kv_fetch_attention(wubu_kv_fetch_t *f, const float *query,
                            wubu_kv_fetch_host_fn host, void *userdata,
                            float *out);

wubu_kv_fetch_decision_t wubu_kv_fetch_decision(const wubu_kv_fetch_t *f);
int wubu_kv_fetch_host_calls(const wubu_kv_fetch_t *f);
float wubu_kv_fetch_last_estimate(const wubu_kv_fetch_t *f);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KV_FETCH_H */
