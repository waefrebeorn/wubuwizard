/* wubu_tree_accept.c -- EG-91 tree acceptance gate. Strict C11. */
#include "wubu_tree_accept.h"

#include <stdlib.h>

static int valid_tree(const int *parent, const int *match, int n)
{
    if (!parent || !match || n < 1 || parent[0] != -1 || match[0] != 1)
        return 0;
    for (int i = 1; i < n; ++i)
        if (parent[i] < 0 || parent[i] >= i ||
            (match[i] != 0 && match[i] != 1)) return 0;
    return 1;
}

int wubu_tree_rounds(int n)
{
    int r = 0;
    if (n < 1) return -1;
    while (r < 31 && (1 << r) < n) ++r;
    return r;
}

int wubu_tree_accept_jump(const int *parent, const int *match,
                          int n, int *out_path)
{
    int *up, *ok, *depth, best = 0, len = 0, at, rounds;
    long long best_key = 0;
    if (!valid_tree(parent, match, n) || !out_path) return -1;
    up = (int *)malloc((size_t)n * sizeof(int));
    ok = (int *)malloc((size_t)n * sizeof(int));
    depth = (int *)malloc((size_t)n * sizeof(int));
    if (!up || !ok || !depth) {
        free(up); free(ok); free(depth);
        return -1;
    }

    /* Doubling state: edge to parent, match of that edge, edge count. */
    for (int i = 0; i < n; ++i) {
        up[i] = i == 0 ? 0 : parent[i];
        ok[i] = i == 0 ? 1 : match[i];
        depth[i] = i == 0 ? 0 : 1;
    }
    rounds = wubu_tree_rounds(n);
    for (int r = 0; r < rounds; ++r) {
        for (int i = 1; i < n; ++i) {
            int u = up[i];
            depth[i] += depth[u];
            ok[i] = ok[i] && ok[u];
            up[i] = up[u];
        }
    }

    /* Max key: deepest fully-matching node, smallest index on ties. */
    for (int i = 1; i < n; ++i) {
        long long key = (long long)depth[i] * (n + 1) + (n - i);
        if (ok[i] && key > best_key) { best_key = key; best = i; }
    }

    /* Reconstruct only the accepted path (root first). */
    for (at = best; at >= 0; at = parent[at]) out_path[len++] = at;
    for (int i = 0; i < len / 2; ++i) {
        int t = out_path[i];
        out_path[i] = out_path[len - 1 - i];
        out_path[len - 1 - i] = t;
    }
    free(up); free(ok); free(depth);
    return len;
}

int wubu_tree_kv_permute(const int *path, int path_len, int n, int *out_slots)
{
    unsigned char *used;
    int w = 0;
    if (!path || !out_slots || n < 1 || path_len < 1 || path_len > n) return -1;
    used = (unsigned char *)calloc((size_t)n, 1);
    if (!used) return -1;
    for (int i = 0; i < path_len; ++i) {
        int s = path[i];
        if (s < 0 || s >= n || used[s]) { free(used); return -1; }
        used[s] = 1;
        out_slots[w++] = s;
    }
    for (int i = 0; i < n; ++i)
        if (!used[i]) out_slots[w++] = i;
    free(used);
    return n;
}
