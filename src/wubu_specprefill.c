/*
 * wubu_specprefill.c -- EG-58 draft-attention prompt compression. Pure C11.
 */
#include "wubu_specprefill.h"

#include <math.h>

int wubu_specprefill_score(const float *emb, int n, int dim, int draft_dim,
                           const float *q_draft, float *score) {
    if (!emb || !q_draft || !score || n <= 0 || dim <= 0 ||
        draft_dim <= 0 || draft_dim > dim) return -1;
    for (int i = 0; i < n; i++) {
        const float *e = emb + (size_t)i * dim;
        float s = 0.0f;
        for (int d = 0; d < draft_dim; d++) s += q_draft[d] * e[d];
        score[i] = s;
    }
    return 0;
}

int wubu_specprefill_select(const float *score, int n, int keep_n, int *keep) {
    int idx[4096];
    if (!score || !keep || n <= 0 || n > 4096 || keep_n <= 0 || keep_n > n)
        return -1;
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = 0; i < keep_n; i++)
        for (int j = i + 1; j < n; j++)
            if (score[idx[j]] > score[idx[i]]) {
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
    for (int i = 0; i < keep_n; i++) keep[i] = idx[i];
    return 0;
}

int wubu_specprefill_answer(const float *emb, const int *keep, int n_keep,
                            int dim, const float *q, float *logits) {
    float mx = -INFINITY, den = 0.0f;
    if (!emb || !keep || !q || !logits || n_keep <= 0 || dim <= 0) return -1;
    for (int i = 0; i < n_keep; i++) {
        int t = keep[i];
        if (t < 0) return -1;
        {
            const float *e = emb + (size_t)t * dim;
            float s = 0.0f;
            for (int d = 0; d < dim; d++) s += q[d] * e[d];
            if (s > mx) mx = s;
        }
    }
    for (int d = 0; d < dim; d++) logits[d] = 0.0f;
    for (int i = 0; i < n_keep; i++) {
        const float *e = emb + (size_t)keep[i] * dim;
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += q[d] * e[d];
        {
            float w = expf(s - mx);
            den += w;
            for (int d = 0; d < dim; d++) logits[d] += w * e[d];
        }
    }
    if (den <= 0.0f) return -1;
    for (int d = 0; d < dim; d++) logits[d] /= den;
    return 0;
}
