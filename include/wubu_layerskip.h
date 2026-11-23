#ifndef WUBU_LAYERSKIP_H
#define WUBU_LAYERSKIP_H

/*
 * wubu_layerskip.h -- EG-60: increasing layer dropout plus an exit-layer query
 * cache reused by the remainder pass (LayerSkip lineage), reimplemented in C11.
 * One gate: KV bytes saved versus output parity against the full stack.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Run the stack with per-layer dropout probabilities. exit_layer >= 0 caches
 * the hidden state there and the remainder pass resumes from that cache.
 * Writes output and the number of bytes of KV that a full pass would have
 * materialized but this path skips. Returns 0 or -1. */
int wubu_layerskip_forward(const float *x, int dim, int layers,
                           const float *dropout, int exit_layer,
                           float *out, size_t *kv_saved);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_LAYERSKIP_H */
