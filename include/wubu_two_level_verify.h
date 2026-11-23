#ifndef WUBU_TWO_LEVEL_VERIFY_H
#define WUBU_TWO_LEVEL_VERIFY_H

/*
 * wubu_two_level_verify.h -- EG-92: SpecPilot two-level verify (TLV).
 * A quantized target view greedily accepts a draft chain without committing;
 * the exact pass verifies that chain and the hung subtree, then commits once.
 * Chain tokens cannot be committed by the quantized pass.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_TLV_MAX_N 32

typedef struct wubu_tlv wubu_tlv_t;

wubu_tlv_t *wubu_tlv_create(void);
void wubu_tlv_free(wubu_tlv_t *p);

/* Quantized pre-verify: greedy prefix of q_match (1 = accepted) becomes the
 * uncommitted chain. Returns chain length, -1 on bad input. */
int wubu_tlv_q_verify(wubu_tlv_t *p, const int *draft_ids,
                      const int *q_match, int n, int *chain_ids);

int wubu_tlv_chain_len(const wubu_tlv_t *p);
int wubu_tlv_committed(const wubu_tlv_t *p);

/* Exact pass: verify the chain, hang the truncated subtree under it, commit
 * chain + accepted subtree tokens in one pass. exact_chain_match has
 * chain_len flags; sub_parent[i] is -1 or an earlier subtree index; budget N
 * truncates the subtree to N-chain_len nodes. out_ids holds committed ids.
 * Returns committed token count (0 when the exact pass rejects the chain),
 * or -1 on bad input. */
int wubu_tlv_exact_commit(wubu_tlv_t *p, const int *exact_chain_match,
                          const int *sub_parent, const int *sub_ids,
                          const int *exact_match, int sub_n, int budget,
                          int *out_ids);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_TWO_LEVEL_VERIFY_H */
