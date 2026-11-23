/*
 * wubu_proxy_route.c -- EG-54 block summary routes. Pure C11.
 */
#include "wubu_proxy_route.h"

#include <math.h>

int wubu_proxy_summary(const float *keys, int n_blocks, int block_size, int dim,
                       float *out) {
    if (!keys || !out || n_blocks <= 0 || block_size <= 0 || dim <= 0) return -1;
    for (int b = 0; b < n_blocks; b++) {
        float *s = out + (size_t)b * dim;
        for (int d = 0; d < dim; d++) s[d] = 0.0f;
        for (int t = 0; t < block_size; t++) {
            const float *k = keys + ((size_t)b * block_size + t) * dim;
            for (int d = 0; d < dim; d++) s[d] += k[d];
        }
        for (int d = 0; d < dim; d++) s[d] /= (float)block_size;
    }
    return 0;
}

int wubu_proxy_route(const float *keys, const float *summary,
                     int n_blocks, int block_size, int dim,
                     const float *q, int topk, float proxy_weight,
                     float *out, int *selected) {
    float score[256], mx = -INFINITY, denom = 0.0f;
    int sel[256], nsel;
    if (!keys || !summary || !q || !out || !selected ||
        n_blocks <= 0 || n_blocks > 256 || block_size <= 0 || dim <= 0 ||
        topk <= 0 || topk >= n_blocks || !(proxy_weight >= 0.0f)) return -1;

    for (int b = 0; b < n_blocks; b++) {
        const float *s = summary + (size_t)b * dim;
        score[b] = 0.0f;
        for (int d = 0; d < dim; d++) score[b] += q[d] * s[d];
        sel[b] = b;
    }
    for (int i = 0; i < n_blocks; i++)
        for (int j = i + 1; j < n_blocks; j++)
            if (score[sel[j]] > score[sel[i]]) {
                int t = sel[i]; sel[i] = sel[j]; sel[j] = t;
            }
    nsel = topk;
    for (int i = 0; i < nsel; i++) selected[i] = sel[i];

    for (int i = 0; i < nsel; i++)
        for (int t = 0; t < block_size; t++) {
            const float *k = keys + ((size_t)sel[i] * block_size + t) * dim;
            float s = 0.0f;
            for (int d = 0; d < dim; d++) s += q[d] * k[d];
            if (s > mx) mx = s;
        }
    for (int d = 0; d < dim; d++) out[d] = 0.0f;

    for (int i = 0; i < nsel; i++)
        for (int t = 0; t < block_size; t++) {
            const float *k = keys + ((size_t)sel[i] * block_size + t) * dim;
            float s = 0.0f;
            for (int d = 0; d < dim; d++) s += q[d] * k[d];
            float w = expf(s - mx);
            denom += w;
            for (int d = 0; d < dim; d++) out[d] += w * k[d];
        }

    /* Unselected blocks: one precomputed mean vector stands in for the whole
     * block, with mass block_size * exp(q . mean). No extra KV is fetched. */
    if (proxy_weight > 0.0f)
        for (int i = nsel; i < n_blocks; i++) {
            int b = sel[i];
            const float *s = summary + (size_t)b * dim;
            float w = (float)block_size * expf(score[b] - mx) * proxy_weight;
            denom += w;
            for (int d = 0; d < dim; d++) out[d] += w * s[d];
        }

    if (denom <= 0.0f) return -1;
    for (int d = 0; d < dim; d++) out[d] /= denom;
    return 0;
}
