/* wubu_shrink.h -- the model-shrink operators (AM01: the symmetric amoeba).
 *
 * The amoeba can extend a pseudopod (grow) but cannot retract it. These
 * are the retraction operators, mirroring wubu_grow exactly:
 *
 *   wubu_shrink_block(m, pos)  -- depth shrink (ShortGPT: remove the
 *       block with the lowest BI score, rewire the residual path).
 *   wubu_shrink_ffn(m, pos)    -- width shrink (low-norm prune with
 *       gate-zero re-insertion for function-preservation).
 *   wubu_train_shrink(tr, pos) -- the train-state pair (realloc the
 *       grad/momentum arrays to match, freeing the removed slot).
 *
 * Shrink is FITNESS-GATED (the doctrine): the caller validates held-out
 * loss after the shrink; if loss spikes beyond tolerance, restore from
 * the archived pre-shrink state (the 5+1 recovery). Grow is
 * function-preserving; shrink is loss-tolerated.
 *
 * C11, self-contained (wraps the wubu_model_t layout + wubu_grow.h).
 */
#ifndef WUBU_SHRINK_H
#define WUBU_SHRINK_H

#include "wubu.h"        /* wubu_model_t, wubu_block_t */
#include "wubu_grow.h"   /* the block layout + train-state pattern */

#ifdef __cplusplus
extern "C" {
#endif

/* SHRINK DEPTH: remove the block at `pos` from the model. The removed
 * block's buffers are freed; later blocks shift down; n_layers--.
 * The residual path is rewired by the caller's validate step (the
 * shrink is fitness-gated, not function-preserving). Returns 0. */
int wubu_shrink_block(wubu_model_t *m, int pos);

/* SHRINK WIDTH: prune the FFN columns with the lowest norms at `pos`
 * and re-insert a gate-zero projection (function-preserving: the
 * forward-before == forward-after to fp tolerance). Returns 0. */
int wubu_shrink_ffn(wubu_model_t *m, int pos, float prune_frac);

/* TRAIN-STATE PAIR: free the removed block's grad/momentum slots and
 * shift the arrays down (the mirror of wubu_train_grow). Returns 0. */
int wubu_train_shrink(wubu_train_t *tr, int pos, int n_layers);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SHRINK_H */
