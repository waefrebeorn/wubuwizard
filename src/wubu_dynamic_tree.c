/* wubu_dyn_tree.c -- EG-90 EAGLE-2 dynamic tree builder. Strict C11. */
#include "wubu_dynamic_tree.h"

#include <stdlib.h>
#include <string.h>

struct wubu_dyn_tree {
    int total, top_k;
};

wubu_dyn_tree_t *wubu_dyn_tree_create(int total_tokens, int top_k)
{
    wubu_dyn_tree_t *p;
    if (total_tokens < 1 || total_tokens > WUBU_DYN_TREE_MAX ||
        top_k < 1 || top_k > 64) return NULL;
    p = (wubu_dyn_tree_t *)calloc(1, sizeof(*p));
    if (p) { p->total = total_tokens; p->top_k = top_k; }
    return p;
}

void wubu_dyn_tree_free(wubu_dyn_tree_t *p) { free(p); }

int wubu_dyn_tree_build(wubu_dyn_tree_t *p,
                        const int *parent, const double *logp, int n,
                        int *selected, int *parent_pos, int *depth)
{
    double score[WUBU_DYN_TREE_MAX];
    int order[WUBU_DYN_TREE_MAX];
    unsigned char kept[WUBU_DYN_TREE_MAX] = {0}, eligible[WUBU_DYN_TREE_MAX];
    int limit, k = 0;
    if (!p || !parent || !logp || !selected || !parent_pos || !depth ||
        n < 1 || n > WUBU_DYN_TREE_MAX || parent[0] != -1 || logp[0] != 0.0)
        return -1;
    memset(eligible, 1, sizeof(eligible));
    for (int i = 1; i < n; ++i) {
        if (parent[i] < 0 || parent[i] >= i || logp[i] > 0.0) return -1;
    }
    score[0] = 0.0;
    for (int i = 1; i < n; ++i) score[i] = score[parent[i]] + logp[i];
    /* Only the top_k highest-logp siblings of each parent are eligible. */
    for (int i = 1; i < n; ++i) {
        int better = 0;
        for (int j = 1; j < n; ++j)
            if (j != i && parent[j] == parent[i] && logp[j] > logp[i]) ++better;
        if (better >= p->top_k) eligible[i] = 0;
    }
    for (int i = 0; i < n; ++i) order[i] = i;
    for (int i = 1; i < n; ++i) {           /* score desc, index asc */
        int key = order[i], j = i - 1;
        while (j >= 0 && (score[order[j]] < score[key] ||
              (score[order[j]] == score[key] && order[j] > key))) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = key;
    }
    limit = p->total < n ? p->total : n;
    for (int r = 0; r < n && k < limit; ++r) {
        int node = order[r];
        if (!eligible[node] || kept[node]) continue;
        /* Ancestor closure: add the path to the root first. */
        int chain[WUBU_DYN_TREE_MAX], c = 0;
        for (int at = node; at >= 0; at = parent[at]) chain[c++] = at;
        for (int a = c - 1; a >= 0 && k < limit; --a) {
            int anc = chain[a];
            if (!kept[anc]) {
                kept[anc] = 1;
                selected[k] = anc;
                parent_pos[k] = -1;
                depth[k] = 0;
                ++k;
            }
        }
    }
    /* Re-sort the selected set by cumulative score for the verifier order. */
    for (int i = 1; i < k; ++i) {
        int key = selected[i], j = i - 1;
        while (j >= 0 && (score[selected[j]] < score[key] ||
              (score[selected[j]] == score[key] && selected[j] > key))) {
            selected[j + 1] = selected[j];
            parent_pos[j + 1] = parent_pos[j];
            depth[j + 1] = depth[j];
            --j;
        }
        selected[j + 1] = key;
    }
    for (int i = 0; i < k; ++i) {
        int node = selected[i], par = parent[node];
        parent_pos[i] = -1;
        depth[i] = 0;
        for (int j = 0; j < k; ++j) {
            if (selected[j] == par) { parent_pos[i] = j; depth[i] = depth[j] + 1; break; }
        }
    }
    return k;
}
