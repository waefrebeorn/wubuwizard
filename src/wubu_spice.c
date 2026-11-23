/*
 * wubu_spice.c -- EG-51 SPICE surrogate and routing. Pure C11.
 */
#include "wubu_spice.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void diff_one(int rows, int cols, const float *u,
                     const float *v, float sigma, float *R) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            R[(size_t)i * cols + j] -= sigma * u[i] * v[j];
}

static int rank_one(const float *R, int rows, int cols,
                    float *u, float *v, float *sigma) {
    double un = 0.0;
    for (int j = 0; j < cols; j++) v[j] = 1.0f / (float)sqrt((double)cols);
    for (int it = 0; it < 40; it++) {
        for (int i = 0; i < rows; i++) {
            double s = 0.0;
            for (int j = 0; j < cols; j++) s += (double)R[(size_t)i * cols + j] * v[j];
            u[i] = (float)s;
        }
        for (int i = 0; i < rows; i++) un += (double)u[i] * u[i];
        if (un <= 1e-18) return -1;
        float norm = (float)sqrt(un);
        for (int i = 0; i < rows; i++) u[i] /= norm;
        double s = 0.0;
        for (int j = 0; j < cols; j++) {
            double acc = 0.0;
            for (int i = 0; i < rows; i++) acc += (double)u[i] * R[(size_t)i * cols + j];
            v[j] = (float)acc;
            s += acc * acc;
        }
        if (s <= 1e-18) return -1;
        norm = (float)sqrt((float)s);
        for (int j = 0; j < cols; j++) v[j] /= norm;
        un = 0.0;
    }
    double sig = 0.0;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            sig += (double)u[i] * R[(size_t)i * cols + j] * v[j];
    *sigma = (float)sig;
    return 0;
}

int wubu_spice_surrogate_build(const float *M, int rows, int cols, int rank,
                               wubu_spice_surrogate_t *out) {
    float *R;
    double n2 = 0.0, e2 = 0.0;
    if (!M || !out || rows <= 0 || cols <= 0 || rank <= 0 ||
        rank > rows || rank > cols) return -1;
    memset(out, 0, sizeof(*out));
    out->sigma = (float *)calloc((size_t)rank, sizeof(float));
    out->u = (float *)calloc((size_t)rows * rank, sizeof(float));
    out->v = (float *)calloc((size_t)cols * rank, sizeof(float));
    R = (float *)malloc((size_t)rows * cols * sizeof(float));
    if (!out->sigma || !out->u || !out->v || !R) goto fail;
    memcpy(R, M, (size_t)rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) n2 += (double)M[i] * M[i];
    out->rows = rows;
    out->cols = cols;
    out->rank = rank;
    for (int k = 0; k < rank; k++) {
        float *u = out->u + (size_t)k * rows;
        float *v = out->v + (size_t)k * cols;
        if (rank_one(R, rows, cols, u, v, &out->sigma[k]) != 0) goto fail;
        diff_one(rows, cols, u, v, out->sigma[k], R);
    }
    for (int i = 0; i < rows * cols; i++) e2 += (double)R[i] * R[i];
    out->relerr = n2 > 0.0 ? (float)sqrt(e2 / n2) : 0.0f;
    free(R);
    return 0;
fail:
    free(R);
    wubu_spice_surrogate_free(out);
    return -1;
}

int wubu_spice_surrogate_apply(const wubu_spice_surrogate_t *s, int row,
                               const float *x, float *y) {
    if (!s || !s->u || !s->v || !x || !y || row < 0 || row >= s->rows) return -1;
    *y = 0.0f;
    for (int k = 0; k < s->rank; k++) {
        double proj = 0.0;
        const float *v = s->v + (size_t)k * s->cols;
        for (int j = 0; j < s->cols; j++) proj += (double)v[j] * x[j];
        *y += (float)proj * s->sigma[k] * s->u[(size_t)k * s->rows + row];
    }
    return 0;
}

void wubu_spice_surrogate_free(wubu_spice_surrogate_t *s) {
    if (!s) return;
    free(s->sigma);
    free(s->u);
    free(s->v);
    memset(s, 0, sizeof(*s));
}

int wubu_spice_route(const float *conf, int n, float relerr, float tol,
                     float lookahead_conf, int *actions) {
    if (!conf || !actions || n <= 0 || !(relerr >= 0.0f) ||
        !(tol >= 0.0f)) return -1;
    for (int i = 0; i < n; i++) {
        if (conf[i] < 0.0f || conf[i] > 1.0f) return -1;
        if (conf[i] >= lookahead_conf) actions[i] = WUBU_SPICE_LOOKAHEAD;
        else if (relerr <= tol) actions[i] = WUBU_SPICE_SURROGATE;
        else actions[i] = WUBU_SPICE_EXACT;
    }
    return 0;
}
