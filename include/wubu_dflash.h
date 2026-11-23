#ifndef WUBU_DFLASH_H
#define WUBU_DFLASH_H

/*
 * wubu_dflash.h -- EG-56: block-diffusion drafter with fused target hidden
 * states injected into every draft layer's KV (DFlash lineage), reimplemented
 * in C11. One gate: accepted tokens per target pass versus an uninjected draft.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* For each of n tokens in a diffusion block, fuse the target hidden state into
 * the draft hidden at every layer and score acceptance against the target.
 * Writes accepted tokens and per-token acceptance probabilities.
 * Returns accepted count, or -1 on bad arguments. */
int wubu_dflash_block(const float *h_target, const float *h_draft,
                      int n_tokens, int dim, int layers, float inject,
                      const float *uniforms, float *probs);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_DFLASH_H */
