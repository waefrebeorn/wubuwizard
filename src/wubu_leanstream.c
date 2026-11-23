/*
 * wubu_leanstream.c -- EG-65 partial-result streaming. Pure C11.
 */
#include "wubu_leanstream.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void step(const float *in, float *summary, int dim, int token, float *h) {
    for (int d = 0; d < dim; d++) {
        float pred = in[d] + 0.05f * summary[d];
        h[d] = pred + 0.01f * cosf(0.13f * (float)token + 0.7f * (float)d);
        summary[d] = 0.9f * summary[d] + 0.1f * h[d];
    }
}

int wubu_leanstream_full(const float *input, int n, int dim, float *out,
                         size_t *peak_bytes) {
    float *all, summary[64];
    if (!input || !out || !peak_bytes || n <= 0 || dim <= 0 || dim > 64) return -1;
    all = (float *)malloc((size_t)n * dim * sizeof(float));
    if (!all) return -1;
    memset(summary, 0, sizeof(summary));
    for (int i = 0; i < n; i++) {
        step(input + (size_t)i * dim, summary, dim, i,
             all + (size_t)i * dim);
    }
    memcpy(out, all, (size_t)n * dim * sizeof(float));
    *peak_bytes = (size_t)n * dim * sizeof(float);
    free(all);
    return 0;
}

int wubu_leanstream_stream(const float *input, int n, int dim, int chunk,
                           float *out, size_t *peak_bytes) {
    float *partial, summary[64];
    if (!input || !out || !peak_bytes || n <= 0 || dim <= 0 || dim > 64 ||
        chunk <= 0) return -1;
    partial = (float *)malloc((size_t)chunk * dim * sizeof(float));
    if (!partial) return -1;
    memset(summary, 0, sizeof(summary));
    for (int base = 0; base < n; base += chunk) {
        int cnt = base + chunk > n ? n - base : chunk;
        for (int t = 0; t < cnt; t++) {
            float *h = partial + (size_t)t * dim;
            step(input + (size_t)(base + t) * dim, summary, dim, base + t, h);
            /* Refine and emit immediately; no full hidden-state buffer. */
            memcpy(out + (size_t)(base + t) * dim, h, (size_t)dim * sizeof(float));
        }
    }
    *peak_bytes = (size_t)chunk * dim * sizeof(float) + (size_t)dim * sizeof(float);
    free(partial);
    return 0;
}
