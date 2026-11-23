/* wubu_two_level_verify.c -- EG-92 two-level quantized pre-verify gate. C11. */
#include "wubu_two_level_verify.h"

#include <stdlib.h>

struct wubu_tlv {
    int chain_len;
    int committed;
    int chain[WUBU_TLV_MAX_N];
};

wubu_tlv_t *wubu_tlv_create(void)
{
    return (wubu_tlv_t *)calloc(1, sizeof(wubu_tlv_t));
}

void wubu_tlv_free(wubu_tlv_t *p) { free(p); }

int wubu_tlv_q_verify(wubu_tlv_t *p, const int *draft_ids,
                      const int *q_match, int n, int *chain_ids)
{
    int k = 0;
    if (!p || !draft_ids || !q_match || !chain_ids ||
        n < 1 || n > WUBU_TLV_MAX_N) return -1;
    while (k < n && q_match[k] == 1) ++k;
    for (int i = 0; i < k; ++i) {
        p->chain[i] = draft_ids[i];
        chain_ids[i] = draft_ids[i];
    }
    p->chain_len = k;
    p->committed = 0;
    return k;
}

int wubu_tlv_chain_len(const wubu_tlv_t *p) { return p ? p->chain_len : -1; }

int wubu_tlv_committed(const wubu_tlv_t *p) { return p ? p->committed : -1; }

int wubu_tlv_exact_commit(wubu_tlv_t *p, const int *exact_chain_match,
                          const int *sub_parent, const int *sub_ids,
                          const int *exact_match, int sub_n, int budget,
                          int *out_ids)
{
    int accepted[WUBU_TLV_MAX_N], use_n, w = 0;
    if (!p || !exact_chain_match || !out_ids || p->chain_len < 1 ||
        sub_n < 0 || budget < p->chain_len || budget > WUBU_TLV_MAX_N) return -1;
    if (sub_n > 0 && (!sub_parent || !sub_ids || !exact_match)) return -1;
    for (int i = 0; i < p->chain_len; ++i) {
        if (exact_chain_match[i] != 0 && exact_chain_match[i] != 1) return -1;
        if (exact_chain_match[i] == 0) {
            p->committed = 0;   /* quantized chain not ratified by exact pass */
            return 0;
        }
    }
    use_n = sub_n < budget - p->chain_len ? sub_n : budget - p->chain_len;
    for (int i = 0; i < use_n; ++i) {
        if ((sub_parent[i] != -1 && (sub_parent[i] < 0 || sub_parent[i] >= i)) ||
            (exact_match[i] != 0 && exact_match[i] != 1)) return -1;
    }
    for (int i = 0; i < p->chain_len; ++i) out_ids[w++] = p->chain[i];
    for (int i = 0; i < use_n; ++i) {
        int par = sub_parent[i] < 0 ? p->chain_len - 1
                                    : p->chain_len + sub_parent[i];
        int ok = exact_match[i] &&
                 (par == p->chain_len - 1 || accepted[par - p->chain_len]);
        accepted[i] = ok;
        if (ok) out_ids[w++] = sub_ids[i];
    }
    p->committed = w;
    return w;
}
