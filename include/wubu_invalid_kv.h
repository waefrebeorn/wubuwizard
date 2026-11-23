#ifndef WUBU_INVALID_KV_H
#define WUBU_INVALID_KV_H

/*
 * wubu_invalid_kv.h -- EG-40: reuse KV invalidated by a context edit as
 * speculative drafts, verify after prefill, preempt on low acceptance
 * (Callo lineage). Pure C11.
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * draft/fresh are n token ids. Verifies the longest common prefix. If the
 * acceptance rate meets `threshold`, returns the accepted count and the fresh
 * prefix is safe to commit; otherwise preempts (accepted = 0) and the caller
 * must use the freshly prefilled KV only.
 */
int wubu_invalid_kv_verify(const int *draft, const int *fresh, int n,
                           float threshold, int *accepted,
                           float *accept_rate, int *preempt);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_INVALID_KV_H */
