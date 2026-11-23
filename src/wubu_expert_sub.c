/* wubu_expert_sub.c -- EG-78 constant novice experts. Strict C11. */
#include "wubu_expert_sub.h"

#include <stdlib.h>
#include <string.h>

struct wubu_expert_sub {
    int experts, hidden;
    size_t mlp_bytes, saved;
    unsigned char sub[WUBU_EXPERT_MAX];
    float cv[WUBU_EXPERT_MAX][WUBU_HIDDEN_MAX];
};

wubu_expert_sub_t *wubu_expert_sub_create(int experts, int hidden,
                                          size_t mlp_bytes)
{
    wubu_expert_sub_t *p;
    if (experts < 1 || experts > WUBU_EXPERT_MAX ||
        hidden < 1 || hidden > WUBU_HIDDEN_MAX || mlp_bytes == 0) return NULL;
    p = (wubu_expert_sub_t *)calloc(1, sizeof(*p));
    if (p) { p->experts = experts; p->hidden = hidden; p->mlp_bytes = mlp_bytes; }
    return p;
}

void wubu_expert_sub_free(wubu_expert_sub_t *p) { free(p); }

size_t wubu_expert_sub_mark(wubu_expert_sub_t *p, int expert_id,
                            const float *const_vector)
{
    if (!p || !const_vector || expert_id < 0 || expert_id >= p->experts) return 0;
    if (p->sub[expert_id]) return 0;       /* no double counting */
    memcpy(p->cv[expert_id], const_vector, (size_t)p->hidden * sizeof(float));
    p->sub[expert_id] = 1;
    p->saved += p->mlp_bytes;
    return p->mlp_bytes;
}

int wubu_expert_sub_is_substituted(const wubu_expert_sub_t *p, int expert_id)
{
    if (!p || expert_id < 0 || expert_id >= p->experts) return -1;
    return p->sub[expert_id] ? 1 : 0;
}

size_t wubu_expert_sub_bytes_saved(const wubu_expert_sub_t *p)
{
    return p ? p->saved : 0;
}

int wubu_expert_sub_contrib(const wubu_expert_sub_t *p,
                            const double *router_weights,
                            const int *selected, int k, float *out_hidden)
{
    if (!p || !router_weights || !selected || !out_hidden ||
        k < 0 || k > p->experts) return -1;
    for (int h = 0; h < p->hidden; ++h) out_hidden[h] = 0.0f;
    for (int i = 0; i < k; ++i) {
        int e = selected[i];
        if (e < 0 || e >= p->experts) return -1;
        if (!p->sub[e]) continue;          /* normal expert keeps its own path */
        for (int h = 0; h < p->hidden; ++h)
            out_hidden[h] += (float)(router_weights[e] * (double)p->cv[e][h]);
    }
    return 0;
}
