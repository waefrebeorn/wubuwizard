/* wubu_dfly.c -- EG-81 shared-KV drafting gate. Strict C11. */
#include "wubu_dfly.h"

#include <stdlib.h>
#include <string.h>

struct wubu_dfly {
    int layers, draft_tokens, index_slots, idx_n;
    size_t kv_bytes, avoided;
    unsigned char shared[WUBU_DFLY_MAX_LAYERS];
    int idx[WUBU_DFLY_MAX_INDEX];
};

wubu_dfly_t *wubu_dfly_create(int layers, int draft_tokens,
                              size_t kv_bytes_per_token, int index_slots)
{
    wubu_dfly_t *p;
    if (layers < 1 || layers > WUBU_DFLY_MAX_LAYERS ||
        draft_tokens < 1 || kv_bytes_per_token == 0 ||
        index_slots < 1 || index_slots > WUBU_DFLY_MAX_INDEX) return NULL;
    p = (wubu_dfly_t *)calloc(1, sizeof(*p));
    if (p) {
        p->layers = layers;
        p->draft_tokens = draft_tokens;
        p->kv_bytes = kv_bytes_per_token;
        p->index_slots = index_slots;
    }
    return p;
}

void wubu_dfly_free(wubu_dfly_t *p) { free(p); }

size_t wubu_dfly_share_kv(wubu_dfly_t *p, int layer)
{
    size_t saved;
    if (!p || layer < 0 || layer >= p->layers || p->shared[layer]) return 0;
    p->shared[layer] = 1;
    saved = (size_t)p->draft_tokens * p->kv_bytes;
    p->avoided += saved;
    return saved;
}

int wubu_dfly_separate_kv_allocations(const wubu_dfly_t *p)
{
    return p ? 0 : -1;
}

int wubu_dfly_set_indices(wubu_dfly_t *p, const int *idx, int n)
{
    if (!p || !idx || n < 1 || n > p->index_slots || n > WUBU_DFLY_MAX_INDEX)
        return -1;
    memcpy(p->idx, idx, (size_t)n * sizeof(int));
    p->idx_n = n;
    return 0;
}

int wubu_dfly_indices_match(const wubu_dfly_t *p, const int *idx, int n)
{
    if (!p || !idx || n < 1 || n != p->idx_n) return -1;
    return memcmp(p->idx, idx, (size_t)n * sizeof(int)) == 0 ? 1 : 0;
}

size_t wubu_dfly_bytes_avoided(const wubu_dfly_t *p)
{
    return p ? p->avoided : 0;
}
