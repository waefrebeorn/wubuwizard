#ifndef WUBU_TREE_ACCEPT_H
#define WUBU_TREE_ACCEPT_H

/*
 * wubu_tree_accept.h -- EG-91: pointer-jumping speculative tree acceptance.
 * SpecPilot d7 (_d7_tree_accept_kernel), reimplemented as a C11 index gate.
 *
 * ceil(log2 N) doubling rounds compute depth, all-edges-match and ancestor
 * reach in parallel; the deepest fully-matching node wins by the max key
 * depth*(N+1) + (N-index). Path reconstruction walks only the accepted path,
 * and the KV permutation touches only those slots.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Doubling rounds needed for a tree of n nodes. -1 when n < 1. */
int wubu_tree_rounds(int n);

/* parent[0] = -1; parent[i] in [0,i). match[0] = 1, match[i] in {0,1}.
 * Writes root..deepest-accepted-node into out_path (capacity n).
 * Returns path length >= 1, or -1 on invalid input. */
int wubu_tree_accept_jump(const int *parent, const int *match,
                          int n, int *out_path);

/* Compact KV slots: accepted path first, then remaining slots ascending.
 * out_slots has capacity n; returns n, or -1 on invalid input. */
int wubu_tree_kv_permute(const int *path, int path_len, int n, int *out_slots);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_TREE_ACCEPT_H */
