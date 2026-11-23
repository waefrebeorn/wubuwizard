/* wubu_index_reuse.c -- EG-76 cross-layer DSA index reuse. Strict C11. */
#include "wubu_index_reuse.h"

#include <stdlib.h>

struct wubu_index_reuse {
    int layers, top_k, dim, shared_k, published_by, reused;
    size_t elem_bytes, allocated;
    int shared[WUBU_INDEX_MAX];
};

wubu_index_reuse_t *wubu_index_reuse_create(int layers, int top_k,
                                            int dim, size_t elem_bytes)
{
    wubu_index_reuse_t *p;
    if (layers < 1 || top_k < 1 || top_k > WUBU_INDEX_MAX ||
        dim < 1 || elem_bytes == 0) return NULL;
    p = (wubu_index_reuse_t *)calloc(1, sizeof(*p));
    if (p) {
        p->layers = layers; p->top_k = top_k;
        p->dim = dim; p->elem_bytes = elem_bytes;
    }
    return p;
}

void wubu_index_reuse_free(wubu_index_reuse_t *p) { free(p); }

int wubu_index_reuse_compute(const wubu_index_reuse_t *p,
                             const double *scores, int n, int *out_idx)
{
    if (!p || !scores || !out_idx || n < p->top_k || n > WUBU_INDEX_MAX) return -1;
    for (int i = 0; i < n; ++i) out_idx[i] = i;
    for (int i = 1; i < n; ++i) {
        int key = out_idx[i], j = i - 1;
        while (j >= 0 && (scores[out_idx[j]] < scores[key] ||
              (scores[out_idx[j]] == scores[key] && out_idx[j] > key))) {
            out_idx[j + 1] = out_idx[j];
            --j;
        }
        out_idx[j + 1] = key;
    }
    return p->top_k;
}

size_t wubu_index_reuse_publish(wubu_index_reuse_t *p, int layer,
                                const int *idx, int k)
{
    if (!p || !idx || k != p->top_k || layer < 0 || layer >= p->layers) return 0;
    for (int i = 0; i < k; ++i)
        if (idx[i] < 0 || idx[i] >= WUBU_INDEX_MAX) return 0;
    for (int i = 0; i < k; ++i) p->shared[i] = idx[i];
    p->shared_k = k;
    p->published_by = layer + 1;    /* 0 means no publication */
    p->allocated = (size_t)k * (size_t)p->dim * p->elem_bytes;
    return p->allocated;
}

size_t wubu_index_reuse_bytes_allocated(const wubu_index_reuse_t *p)
{
    return p ? p->allocated : 0;
}

int wubu_index_reuse_reuse(wubu_index_reuse_t *p, int layer,
                           const int *idx, int k)
{
    if (!p || !idx || k != p->shared_k || layer < 0 || layer >= p->layers ||
        p->published_by == 0 || layer == p->published_by - 1) return -1;
    for (int i = 0; i < k; ++i)
        if (idx[i] != p->shared[i]) return -1;
    ++p->reused;
    return 0;
}

size_t wubu_index_reuse_bytes_saved(const wubu_index_reuse_t *p)
{
    return p ? (size_t)p->reused * p->allocated : 0;
}
