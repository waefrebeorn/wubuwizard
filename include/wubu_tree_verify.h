#ifndef WUBU_TREE_VERIFY_H
#define WUBU_TREE_VERIFY_H

/*
 * wubu_tree_verify.h -- EG-21: tree attention + longest-prefix speculative
 * verification (Medusa / EAGLE tree lineage), reimplemented in C11.
 *
 * Nodes form a parent-indexed draft tree rooted at node 0. Each node carries
 * the acceptance probability p_target/p_draft at its position. Verification
 * walks each root-to-leaf chain, accepts while deterministic rejection draws
 * stay under that probability, and selects the longest accepted chain. A
 * rejection at depth k emits one bonus token sampled from the target.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int parent;      /* -1 only for the root */
    int token;
    float accept_p;  /* target/draft ratio, clamped to [0,1] */
    int bonus_token; /* emitted when this node is rejected */
} wubu_tree_node_t;

/* Build an ancestor mask: mask[i*stride+j] = 1 if j is i or an ancestor of i.
 * stride must be >= n_nodes. Returns 0 or -1. */
int wubu_tree_mask(const wubu_tree_node_t *nodes, int n_nodes,
                   unsigned char *mask, int stride);

/* Verify with uniforms from uniforms[0..n_uniforms-1] (0 <= u < 1).
 * Writes accepted tokens plus an optional bonus into out (capacity cap).
 * Returns accepted token count (excluding root), or -1 on bad input. */
int wubu_tree_verify(const wubu_tree_node_t *nodes, int n_nodes,
                     const float *uniforms, int n_uniforms,
                     int *out, int cap);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_TREE_VERIFY_H */
