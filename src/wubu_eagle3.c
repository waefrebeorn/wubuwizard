/*
 * wubu_eagle3.c -- EG-24 tri-layer fusion + direct prediction. Pure C11.
 */
#include "wubu_eagle3.h"

#include <math.h>

/* Fixed fusion weights; a trained drafter would learn these. */
#define W0 0.1f
#define W1 0.2f
#define W2 0.7f

int wubu_eagle3_fuse(const float *h0, const float *h1, const float *h2,
                     int dim, float *out) {
    if (!h0 || !h1 || !h2 || !out || dim <= 0) return -1;
    for (int i = 0; i < dim; i++)
        out[i] = W0 * h0[i] + W1 * h1[i] + W2 * h2[i];
    return 0;
}

int wubu_eagle3_predict(const float *fused, const float *emb,
                        int vocab, int dim, float *logits) {
    if (!fused || !emb || !logits || vocab <= 0 || dim <= 0) return -1;
    for (int v = 0; v < vocab; v++) {
        float s = 0.0f;
        const float *e = emb + (size_t)v * dim;
        for (int i = 0; i < dim; i++) s += fused[i] * e[i];
        logits[v] = s;
    }
    return 0;
}

int wubu_eagle3_top1(const float *logits, int vocab) {
    int best = 0;
    if (!logits || vocab <= 0) return -1;
    for (int v = 1; v < vocab; v++)
        if (logits[v] > logits[best]) best = v;
    return best;
}

float wubu_eagle3_confidence(const float *logits, int vocab) {
    int a = -1, b = -1;
    float mx, denom, pa, pb;
    if (!logits || vocab < 2) return 0.0f;
    mx = logits[0];
    for (int v = 1; v < vocab; v++) if (logits[v] > mx) mx = logits[v];
    denom = 0.0f;
    for (int v = 0; v < vocab; v++) denom += expf(logits[v] - mx);
    for (int v = 0; v < vocab; v++) {
        float p = expf(logits[v] - mx) / denom;
        if (a < 0 || p > pa) { b = a; pb = pa; a = v; pa = p; }
        else if (b < 0 || p > pb) { b = v; pb = p; }
    }
    return b >= 0 ? pa - pb : 1.0f;
}
