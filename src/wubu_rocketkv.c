/*
 * wubu_rocketkv.c -- EG-53 training-free page eviction + Top-K decode. Pure C11.
 */
#include "wubu_rocketkv.h"

#include <math.h>

int wubu_rocketkv_importance(const float *keys, int n_pages, int page_size,
                             int dim, const float *q, int n_q, float *score) {
    if (!keys || !q || !score || n_pages <= 0 || page_size <= 0 ||
        dim <= 0 || n_q <= 0) return -1;
    for (int p = 0; p < n_pages; p++) score[p] = 0.0f;
    for (int i = 0; i < n_q; i++) {
        const float *qi = q + (size_t)i * dim;
        float total = 0.0f;
        for (int p = 0; p < n_pages; p++) {
            float page_mass = 0.0f;
            for (int t = 0; t < page_size; t++) {
                const float *k = keys + ((size_t)p * page_size + t) * dim;
                float s = 0.0f;
                for (int d = 0; d < dim; d++) s += qi[d] * k[d];
                page_mass += expf(s);
            }
            score[p] += page_mass;
            total += page_mass;
        }
        if (total > 0.0f)
            for (int p = 0; p < n_pages; p++) score[p] /= total;
    }
    return 0;
}

int wubu_rocketkv_select(const float *score, int n_pages, float retain_frac,
                         int *keep) {
    int idx[4096], nkeep;
    if (!score || !keep || n_pages <= 0 || n_pages > 4096 ||
        !(retain_frac > 0.0f) || retain_frac > 1.0f) return -1;
    for (int i = 0; i < n_pages; i++) idx[i] = i;
    for (int i = 0; i < n_pages; i++)
        for (int j = i + 1; j < n_pages; j++)
            if (score[idx[j]] > score[idx[i]]) {
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
    nkeep = (int)ceilf(retain_frac * (float)n_pages);
    if (nkeep < 1) nkeep = 1;
    if (nkeep > n_pages) nkeep = n_pages;
    for (int i = 0; i < nkeep; i++) keep[i] = idx[i];
    return nkeep;
}

int wubu_rocketkv_decode(const float *keys, const int *keep, int n_keep,
                         int page_size, int dim, const float *q, float *out) {
    float mx = -INFINITY, denom = 0.0f;
    if (!keys || !keep || !q || !out || n_keep <= 0 || page_size <= 0 ||
        dim <= 0) return -1;
    for (int i = 0; i < n_keep; i++) {
        int p = keep[i];
        if (p < 0) return -1;
        for (int t = 0; t < page_size; t++) {
            const float *k = keys + ((size_t)p * page_size + t) * dim;
            float s = 0.0f;
            for (int d = 0; d < dim; d++) s += q[d] * k[d];
            if (s > mx) mx = s;
        }
    }
    for (int d = 0; d < dim; d++) out[d] = 0.0f;
    for (int i = 0; i < n_keep; i++) {
        int p = keep[i];
        for (int t = 0; t < page_size; t++) {
            const float *k = keys + ((size_t)p * page_size + t) * dim;
            float s = 0.0f;
            for (int d = 0; d < dim; d++) s += q[d] * k[d];
            float w = expf(s - mx);
            denom += w;
            for (int d = 0; d < dim; d++) out[d] += w * k[d];
        }
    }
    if (denom <= 0.0f) return -1;
    for (int d = 0; d < dim; d++) out[d] /= denom;
    return 0;
}
