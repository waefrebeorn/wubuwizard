/*
 * wubu_kv_fetch.c -- LP-12: tolerance-parameterised KV fetch (9P layer).
 * Strict C11, self-contained, opaque context. See include/wubu_kv_fetch.h.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_kv_fetch.h"

struct wubu_kv_fetch {
    int n, d;
    float eps, delta;
    float *sk_k, *sk_v;      /* sketch, [n*d] */
    int *piv;                /* [n_pivots] */
    float *pv_k, *pv_v;      /* exact pivotal tokens, [n_pivots*d] */
    int n_pivots;
    float *loc_k, *loc_v;    /* sketch with pivots overwritten, [n*d] */
    float *full_k, *full_v;  /* host-provided full page, [n*d] */
    float *scores;           /* [n] */
    int n_pivots_cap;
    wubu_kv_fetch_decision_t decision;
    int host_calls;
    float last_est;
};

wubu_kv_fetch_t *wubu_kv_fetch_create(int n, int d, float eps, float delta)
{
    wubu_kv_fetch_t *f;
    size_t nd;

    if (n <= 0 || d <= 0 || eps < 0.0f || delta < eps) return NULL;
    f = (wubu_kv_fetch_t *)calloc(1, sizeof(*f));
    if (!f) return NULL;
    nd = (size_t)n * (size_t)d;
    f->n = n; f->d = d; f->eps = eps; f->delta = delta;
    f->n_pivots_cap = n;
    f->sk_k = (float *)malloc(nd * sizeof(float));
    f->sk_v = (float *)malloc(nd * sizeof(float));
    f->loc_k = (float *)malloc(nd * sizeof(float));
    f->loc_v = (float *)malloc(nd * sizeof(float));
    f->full_k = (float *)malloc(nd * sizeof(float));
    f->full_v = (float *)malloc(nd * sizeof(float));
    f->scores = (float *)malloc((size_t)n * sizeof(float));
    f->piv = (int *)malloc((size_t)n * sizeof(int));
    f->pv_k = (float *)malloc(nd * sizeof(float));
    f->pv_v = (float *)malloc(nd * sizeof(float));
    if (!f->sk_k || !f->sk_v || !f->loc_k || !f->loc_v ||
        !f->full_k || !f->full_v || !f->scores ||
        !f->piv || !f->pv_k || !f->pv_v) {
        wubu_kv_fetch_free(f);
        return NULL;
    }
    return f;
}

void wubu_kv_fetch_free(wubu_kv_fetch_t *f)
{
    if (!f) return;
    free(f->sk_k); free(f->sk_v);
    free(f->loc_k); free(f->loc_v);
    free(f->full_k); free(f->full_v);
    free(f->scores);
    free(f->piv); free(f->pv_k); free(f->pv_v);
    free(f);
}

int wubu_kv_fetch_set_local(wubu_kv_fetch_t *f,
                            const float *sketch_k, const float *sketch_v,
                            const int *pivot_idx,
                            const float *pivot_k, const float *pivot_v,
                            int n_pivots)
{
    size_t nd;
    if (!f || !sketch_k || !sketch_v) return -1;
    if (n_pivots < 0 || n_pivots > f->n_pivots_cap) return -1;
    if (n_pivots > 0 && (!pivot_idx || !pivot_k || !pivot_v)) return -1;
    for (int i = 0; i < n_pivots; i++)
        if (pivot_idx[i] < 0 || pivot_idx[i] >= f->n) return -1;

    nd = (size_t)f->n * (size_t)f->d * sizeof(float);
    memcpy(f->sk_k, sketch_k, nd);
    memcpy(f->sk_v, sketch_v, nd);
    memcpy(f->loc_k, sketch_k, nd);
    memcpy(f->loc_v, sketch_v, nd);

    f->n_pivots = n_pivots;
    for (int p = 0; p < n_pivots; p++) {
        int tok = pivot_idx[p];
        f->piv[p] = tok;
        memcpy(f->pv_k + (size_t)p * f->d, pivot_k + (size_t)p * f->d,
               (size_t)f->d * sizeof(float));
        memcpy(f->pv_v + (size_t)p * f->d, pivot_v + (size_t)p * f->d,
               (size_t)f->d * sizeof(float));
        /* Local side always keeps pivotal tokens exactly. */
        memcpy(f->loc_k + (size_t)tok * f->d, pivot_k + (size_t)p * f->d,
               (size_t)f->d * sizeof(float));
        memcpy(f->loc_v + (size_t)tok * f->d, pivot_v + (size_t)p * f->d,
               (size_t)f->d * sizeof(float));
    }
    f->decision = WUBU_KV_FETCH_NO;
    f->host_calls = 0;
    f->last_est = 0.0f;
    return 0;
}

float wubu_kv_fetch_estimate(const wubu_kv_fetch_t *f, const float *query)
{
    double qnorm2 = 0.0, worst = 0.0;
    if (!f || !query) return -1.0f;
    for (int j = 0; j < f->d; j++) qnorm2 += (double)query[j] * (double)query[j];
    for (int p = 0; p < f->n_pivots; p++) {
        int tok = f->piv[p];
        double res2 = 0.0;
        for (int j = 0; j < f->d; j++) {
            double e = (double)f->pv_k[(size_t)p * f->d + j] -
                       (double)f->sk_k[(size_t)tok * f->d + j];
            res2 += e * e;
        }
        if (res2 > worst) worst = res2;
    }
    /* Bound on the attention-logit perturbation caused by sketch error. */
    return (float)(sqrt(worst) * sqrt(qnorm2) / sqrt((double)f->d));
}

static void attention(const float *k, const float *v, const float *q,
                      int n, int d, float *scores, float *out)
{
    double mx = -1e300, z = 0.0;
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int j = 0; j < d; j++)
            s += (double)k[(size_t)i * d + j] * (double)q[j];
        s *= (double)inv_sqrt_d;
        scores[i] = (float)s;
        if (s > mx) mx = s;
    }
    for (int i = 0; i < n; i++) {
        double e = exp((double)scores[i] - mx);
        scores[i] = (float)e;
        z += e;
    }
    if (z <= 0.0) z = 1.0;
    for (int i = 0; i < n; i++) scores[i] = (float)((double)scores[i] / z);
    memset(out, 0, (size_t)d * sizeof(float));
    for (int i = 0; i < n; i++) {
        float p = scores[i];
        for (int j = 0; j < d; j++)
            out[j] += p * v[(size_t)i * d + j];
    }
}

int wubu_kv_fetch_attention(wubu_kv_fetch_t *f, const float *query,
                            wubu_kv_fetch_host_fn host, void *userdata,
                            float *out)
{
    if (!f || !query || !out) return -1;

    f->last_est = wubu_kv_fetch_estimate(f, query);
    if (f->last_est < 0.0f) return -1;

    if (f->last_est <= f->eps)
        f->decision = WUBU_KV_FETCH_NO;
    else if (f->last_est >= f->delta)
        f->decision = WUBU_KV_FETCH_FORCED;
    else
        f->decision = WUBU_KV_FETCH_UNCERTAIN;

    if (f->decision == WUBU_KV_FETCH_FORCED) {
        if (!host) return -1;
        if (host(userdata, f->full_k, f->full_v, f->n, f->d) != 0) return -1;
        f->host_calls++;
        attention(f->full_k, f->full_v, query, f->n, f->d, f->scores, out);
    } else {
        attention(f->loc_k, f->loc_v, query, f->n, f->d, f->scores, out);
    }
    return 0;
}

wubu_kv_fetch_decision_t wubu_kv_fetch_decision(const wubu_kv_fetch_t *f)
{
    return f ? f->decision : WUBU_KV_FETCH_UNCERTAIN;
}

int wubu_kv_fetch_host_calls(const wubu_kv_fetch_t *f)
{
    return f ? f->host_calls : -1;
}

float wubu_kv_fetch_last_estimate(const wubu_kv_fetch_t *f)
{
    return f ? f->last_est : -1.0f;
}
