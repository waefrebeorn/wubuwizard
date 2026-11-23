/* wubu_gqa_index.c -- EG-44b GQA sparse-index grouping gate. Strict C11. */
#include "wubu_gqa_index.h"

#include <stdlib.h>

struct wubu_gqa_index {
    int q_heads, kv_heads, n_items, group;
};

wubu_gqa_index_t *wubu_gqa_index_create(int q_heads, int kv_heads, int n_items)
{
    wubu_gqa_index_t *p;
    if (q_heads < 1 || q_heads > WUBU_GQA_MAX_HEADS ||
        kv_heads < 1 || kv_heads > q_heads ||
        q_heads % kv_heads != 0 || n_items < 1 ||
        n_items > WUBU_GQA_MAX_ITEMS) return NULL;
    p = (wubu_gqa_index_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->q_heads = q_heads;
    p->kv_heads = kv_heads;
    p->n_items = n_items;
    p->group = q_heads / kv_heads;
    return p;
}

void wubu_gqa_index_free(wubu_gqa_index_t *p) { free(p); }

int wubu_gqa_index_group_size(const wubu_gqa_index_t *p)
{
    return p ? p->group : -1;
}

int wubu_gqa_index_select(const wubu_gqa_index_t *p, const float *scores,
                          int n_items, int kv_head, int k, int *out)
{
    double avg[WUBU_GQA_MAX_ITEMS];
    if (!p || !scores || !out || n_items != p->n_items ||
        kv_head < 0 || kv_head >= p->kv_heads || k < 1 || k > n_items) return -1;
    for (int j = 0; j < n_items; ++j) {
        double sum = 0.0;
        int base = kv_head * p->group;
        for (int h = 0; h < p->group; ++h)
            sum += scores[(size_t)(base + h) * n_items + j];
        avg[j] = sum / p->group;
    }
    for (int i = 0; i < k; ++i) {
        int best = -1;
        for (int j = 0; j < n_items; ++j) {
            int used = 0;
            for (int u = 0; u < i; ++u) if (out[u] == j) used = 1;
            if (used) continue;
            if (best < 0 || avg[j] > avg[best]) best = j;
        }
        out[i] = best;
    }
    return k;
}

int wubu_gqa_index_work(const wubu_gqa_index_t *p, int k)
{
    return p ? p->kv_heads * k : -1;
}

int wubu_gqa_index_work_naive(const wubu_gqa_index_t *p, int k)
{
    return p ? p->q_heads * k : -1;
}
