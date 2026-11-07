/* wubu_shrink_kv.h — the KV-space amoeba shrink operator
 *
 * The growth side (wubu_grow_kv) grows KV blocks toward UNDER-coherent
 * files. This is the mirror: PRUNE DEAD KV REGIONS. A region is pruned
 * when it is both low-utilization (little attention mass lands in it) and
 * low-coherence (the model does not understand it) — memory the model
 * allocated but never uses. Reclamation frees namespace blocks for the
 * grow operator and the freelist.
 *
 * Analysis is driven by post-forward coherence scores + attention-derived
 * utilization from the trainer (wubu_kv_coherence_diag), the same signal
 * the grow operator consumes — pruning is the cleanup half of the amoeba
 * grow/shrink loop (research/060, INDEX AN04, build-order module #6).
 *
 * C11, opaque structs, no third-party deps.
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, module #6 (AN22).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_SHRINK_KV_H
#define WUBU_SHRINK_KV_H

#include <stddef.h>
#include <stdint.h>
#include "wubu_kv_embedding.h"
#include "wubu_kvfs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Configuration for the KV shrink operator */
typedef struct {
    double utilization_threshold;  /* below this attention mass, the region is idle */
    double coherence_threshold;    /* below this coherence, the model fails it */
    uint32_t block_size;           /* floats per KV block */
} wubu_shrink_kv_cfg_t;

/* Default config (util 0.2, coherence 0.4). */
wubu_shrink_kv_cfg_t wubu_shrink_kv_default_cfg(void);

/* Opaque handle. */
typedef struct wubu_shrink_kv wubu_shrink_kv_t;

/* Create the KV shrink operator over a KV namespace + embedding bridge.
 * Returns NULL on failure. */
wubu_shrink_kv_t *wubu_shrink_kv_create(wubu_kv_embedding_t *kv,
                                         const wubu_shrink_kv_cfg_t *cfg);

/* ANALYZE: classify each encoded file in /kv/in/ as pruneable or not.
 * utilizations / coherences / entropies are per-file arrays of length
 * n_files (from post-forward attention + coherence reward). A file is
 * pruneable iff utilization < threshold AND coherence < threshold (a dead
 * region: never attended to AND not understood). The ranked list is stored
 * internally for SHRINK to consume.
 * Returns the number of pruneable files. */
int wubu_shrink_kv_analyze(wubu_shrink_kv_t *s,
                           const char **paths,
                           const float *utilizations,
                           const float *coherences,
                           const float *entropies,
                           int n_files);

/* Get the current pruneable paths (after analyze).
 * Fills out_paths with up to cap paths; caller does not own the strings.
 * Returns the count. */
int wubu_shrink_kv_prunable(const wubu_shrink_kv_t *s,
                            const char **out_paths, int cap);

/* SHRINK: unreclaim up to max_prune pruneable KV regions.
 * Each prune unmounts the region from the namespace (the backing blocks
 * return to the freelist for future grow/re-encode).
 * Returns the number of regions pruned. */
int wubu_shrink_kv_shrink(wubu_shrink_kv_t *s, int max_prune);

/* Cumulative number of regions pruned since creation. */
unsigned long wubu_shrink_kv_total_pruned(const wubu_shrink_kv_t *s);

/* Free the shrink operator. Does NOT free kv or fs. */
void wubu_shrink_kv_free(wubu_shrink_kv_t *s);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_SHRINK_KV_H */
