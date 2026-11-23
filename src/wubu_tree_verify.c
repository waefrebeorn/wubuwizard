/*
 * wubu_tree_verify.c -- EG-21 tree verification. Pure C11.
 */
#include "wubu_tree_verify.h"

#include <stdlib.h>
#include <string.h>

static int valid_tree(const wubu_tree_node_t *nodes, int n) {
    int i;
    if (!nodes || n <= 0 || nodes[0].parent != -1) return 0;
    for (i = 1; i < n; i++)
        if (nodes[i].parent < 0 || nodes[i].parent >= i) return 0;
    return 1;
}

int wubu_tree_mask(const wubu_tree_node_t *nodes, int n_nodes,
                   unsigned char *mask, int stride) {
    int i;
    if (!valid_tree(nodes, n_nodes) || !mask || stride < n_nodes) return -1;
    memset(mask, 0, (size_t)stride * n_nodes);
    for (i = 0; i < n_nodes; i++) {
        int cur = i;
        while (cur >= 0) {
            mask[(size_t)i * stride + cur] = 1;
            cur = nodes[cur].parent;
        }
    }
    return 0;
}

int wubu_tree_verify(const wubu_tree_node_t *nodes, int n_nodes,
                     const float *uniforms, int n_uniforms,
                     int *out, int cap) {
    int best_len = 0, best_bonus = -1, i, used = 0;
    int best_acc[256];
    if (!valid_tree(nodes, n_nodes) || !uniforms || n_uniforms < 0 ||
        !out || cap < 1 || n_nodes > 256) return -1;

    /* Enumerate root-to-leaf chains by walking back from each leaf. */
    for (i = 1; i < n_nodes; i++) {
        int chain[256], acc[256], len = 0, accepted = 0, bonus = -1;
        int cur = i, child_of_i = 0, j;
        for (j = 1; j < n_nodes; j++)
            if (nodes[j].parent == i) { child_of_i = 1; break; }
        if (child_of_i) continue;        /* only complete chains */
        while (cur > 0) {
            chain[len++] = cur;
            cur = nodes[cur].parent;
        }
        for (j = len - 1; j >= 0; j--) {
            const wubu_tree_node_t *nd = &nodes[chain[j]];
            float u = used < n_uniforms ? uniforms[used] : 1.0f;
            used++;
            if (u > nd->accept_p) { bonus = nd->bonus_token; break; }
            acc[accepted++] = nd->token;
        }
        if (accepted > best_len) {
            best_len = accepted;
            best_bonus = bonus;
            memcpy(best_acc, acc, (size_t)accepted * sizeof(int));
        }
    }
    if (best_len == 0) {
        out[0] = best_bonus >= 0 ? best_bonus : nodes[0].token;
        return 1;
    }
    if (best_len + (best_bonus >= 0 ? 1 : 0) > cap) return -1;
    memcpy(out, best_acc, (size_t)best_len * sizeof(int));
    if (best_bonus >= 0) out[best_len] = best_bonus;
    return best_len + (best_bonus >= 0 ? 1 : 0);
}
