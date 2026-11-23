/*
 * wubu_uno.c -- EG-55 gated-LoRA draft step + lossless Psi-Spec sampling.
 * Pure C11.
 */
#include "wubu_uno.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int wubu_uno_init(wubu_uno_t *m, int dim, int rank, int n_kv) {
    if (!m || dim <= 0 || rank <= 0 || rank > dim || n_kv <= 0) return -1;
    memset(m, 0, sizeof(*m));
    m->W = (float *)calloc((size_t)dim * dim, sizeof(float));
    m->A = (float *)calloc((size_t)dim * rank, sizeof(float));
    m->B = (float *)calloc((size_t)rank * dim, sizeof(float));
    m->kv = (float *)calloc((size_t)n_kv, sizeof(float));
    if (!m->W || !m->A || !m->B || !m->kv) {
        wubu_uno_free(m);
        return -1;
    }
    m->dim = dim;
    m->rank = rank;
    m->n_kv = n_kv;
    return 0;
}

void wubu_uno_free(wubu_uno_t *m) {
    if (!m) return;
    free(m->W);
    free(m->A);
    free(m->B);
    free(m->kv);
    memset(m, 0, sizeof(*m));
}

int wubu_uno_draft_step(const wubu_uno_t *m, const float *x, float gate,
                        float *y) {
    float g, z[64];
    if (!m || !m->W || !m->A || !m->B || !m->kv || !x || !y ||
        m->dim > 64 || m->rank > m->dim) return -1;
    for (int i = 0; i < m->dim; i++) {
        double s = 0.0;
        for (int j = 0; j < m->dim; j++)
            s += (double)m->W[(size_t)i * m->dim + j] * x[j];
        y[i] = (float)s;
    }
    for (int r = 0; r < m->rank; r++) {
        double s = 0.0;
        for (int j = 0; j < m->dim; j++)
            s += (double)m->B[(size_t)r * m->dim + j] * x[j];
        z[r] = (float)s;
    }
    g = 1.0f / (1.0f + expf(-gate));
    for (int i = 0; i < m->dim; i++) {
        double s = 0.0;
        for (int r = 0; r < m->rank; r++)
            s += (double)m->A[(size_t)i * m->rank + r] * z[r];
        y[i] += g * (float)s;
    }
    return 0;
}

int wubu_psi_sample(const float *p, const float *q, int n, int tok,
                    float u1, float u2, int *emit) {
    float r[64], z = 0.0f, alpha, acc = 0.0f;
    if (!p || !q || !emit || n <= 0 || n > 64 || tok < 0 || tok >= n ||
        !(u1 >= 0.0f && u1 < 1.0f) || !(u2 >= 0.0f && u2 < 1.0f)) return -1;
    alpha = q[tok] > 0.0f && p[tok] / q[tok] < 1.0f ? p[tok] / q[tok] : 1.0f;
    if (u1 < alpha) {
        *emit = tok;
        return 1;
    }
    /* Psi-Spec residual is the pointwise positive part (p - q)+, independent
     * of the acceptance probability of the particular drafted token. */
    for (int i = 0; i < n; i++) {
        r[i] = p[i] > q[i] ? p[i] - q[i] : 0.0f;
        z += r[i];
    }
    if (z <= 0.0f) return -1;
    for (int i = 0; i < n; i++) {
        acc += r[i] / z;
        if (u2 < acc) {
            *emit = i;
            return 0;
        }
    }
    *emit = n - 1;
    return 0;
}
