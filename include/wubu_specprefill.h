#ifndef WUBU_SPECPREFILL_H
#define WUBU_SPECPREFILL_H

/*
 * wubu_specprefill.h -- EG-58: draft-attention prompt compression before the
 * base model (SpecPrefill lineage), reimplemented in C11. One gate: prefill
 * token-count reduction (TTFT proxy) versus answer-logit fidelity.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Draft-only salience: dot of a cheap projection of each token embedding with
 * the draft query. Writes n scores, returns 0 or -1. */
int wubu_specprefill_score(const float *emb, int n, int dim, int draft_dim,
                           const float *q_draft, float *score);

/* Keep the top keep_n tokens by draft score. */
int wubu_specprefill_select(const float *score, int n, int keep_n, int *keep);

/* Base-model answer logits from the retained prompt tokens. */
int wubu_specprefill_answer(const float *emb, const int *keep, int n_keep,
                            int dim, const float *q, float *logits);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SPECPREFILL_H */
