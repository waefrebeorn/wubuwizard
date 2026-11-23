/* wubu_sparse_retrieve.c -- EG-75 sparse-KV retrieve lifecycle. Strict C11. */
#include "wubu_sparse_retrieve.h"

#include <stdlib.h>
#include <string.h>

struct wubu_sparse_retrieve {
    int dim, max_pages, recent, n;
    float repr[WUBU_SPARSE_MAX_PAGES][WUBU_SPARSE_MAX_DIM];
};

wubu_sparse_retrieve_t *wubu_sparse_retrieve_create(int dim, int max_pages,
                                                    int recent_window)
{
    wubu_sparse_retrieve_t *p;
    if (dim < 1 || dim > WUBU_SPARSE_MAX_DIM || max_pages < 1 ||
        max_pages > WUBU_SPARSE_MAX_PAGES || recent_window < 0 ||
        recent_window > max_pages) return NULL;
    p = (wubu_sparse_retrieve_t *)calloc(1, sizeof(*p));
    if (p) { p->dim = dim; p->max_pages = max_pages; p->recent = recent_window; }
    return p;
}

void wubu_sparse_retrieve_free(wubu_sparse_retrieve_t *p) { free(p); }

int wubu_sparse_retrieve_update(wubu_sparse_retrieve_t *p,
                                const float *key, int dim)
{
    if (!p || !key || dim != p->dim) return -1;
    if (p->n == p->max_pages) {
        memmove(p->repr[0], p->repr[1],
                (size_t)(p->n - 1) * p->dim * sizeof(float));
        --p->n;
    }
    memcpy(p->repr[p->n++], key, (size_t)dim * sizeof(float));
    return 0;
}

int wubu_sparse_retrieve_page_count(const wubu_sparse_retrieve_t *p)
{
    return p ? p->n : -1;
}

int wubu_sparse_retrieve_query(const wubu_sparse_retrieve_t *p,
                               const float *q, int k,
                               int *out_pages, double *out_scores)
{
    double score[WUBU_SPARSE_MAX_PAGES];
    unsigned char taken[WUBU_SPARSE_MAX_PAGES] = {0};
    int count = 0, limit;
    if (!p || !q || !out_pages || !out_scores || k < 1 || k > p->n) return -1;
    limit = k < p->n ? k : p->n;
    for (int i = 0; i < p->n; ++i) {
        double v = 0.0;
        for (int j = 0; j < p->dim; ++j)
            v += (double)q[j] * (double)p->repr[i][j];
        score[i] = v;
    }
    for (int i = p->n - p->recent; i < p->n && count < limit; ++i) {
        taken[i] = 1;
        out_pages[count] = i;
        out_scores[count++] = score[i];
    }
    while (count < limit) {
        int best = -1;
        for (int i = 0; i < p->n; ++i) {
            if (taken[i]) continue;
            if (best < 0 || score[i] > score[best]) best = i;
        }
        taken[best] = 1;
        out_pages[count] = best;
        out_scores[count++] = score[best];
    }
    for (int i = 1; i < count; ++i) {       /* score desc, page asc */
        int pg = out_pages[i];
        double sc = out_scores[i];
        int j = i - 1;
        while (j >= 0 && (out_scores[j] < sc ||
              (out_scores[j] == sc && out_pages[j] > pg))) {
            out_scores[j + 1] = out_scores[j];
            out_pages[j + 1] = out_pages[j];
            --j;
        }
        out_scores[j + 1] = sc;
        out_pages[j + 1] = pg;
    }
    return limit;
}
