#ifndef WUBU_DYNAMIC_TREE_H
#define WUBU_DYNAMIC_TREE_H

/*
 * wubu_dynamic_tree.h -- EG-90: EAGLE-2 confidence-shaped dynamic tree.
 * Draft log-probabilities choose the tree shape: cumulative scores rank
 * candidates globally, top_k limits siblings per level, and selected nodes
 * carry their ancestors so the verifier sees a valid tree with depths.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_DYN_TREE_MAX 128

typedef struct wubu_dyn_tree wubu_dyn_tree_t;

/* total_tokens in [1,128], top_k in [1,64]. NULL on bad input. */
wubu_dyn_tree_t *wubu_dyn_tree_create(int total_tokens, int top_k);
void wubu_dyn_tree_free(wubu_dyn_tree_t *p);

/* parent[0] = -1, parent[i] in [0,i); logp[i] <= 0 (root logp 0).
 * Writes selected node ids in descending cumulative-score order, their parent
 * positions in that array (-1 for root), and tree depths.
 * Returns number selected, or -1 on bad input. */
int wubu_dyn_tree_build(wubu_dyn_tree_t *p,
                        const int *parent, const double *logp, int n,
                        int *selected, int *parent_pos, int *depth);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_DYNAMIC_TREE_H */
