/*
 * wubu_imgenc.c -- Vision encoder: ViT patch embedding from scratch (CC01). C11.
 */
#include "wubu_imgenc.h"
#include <math.h>
#include <string.h>
#include "wubu_activations.h"

static float lcg_randf(unsigned *seed) {
    *seed = (*seed * 1103515245U + 12345U) & 0x7fffffff;
    return (float)((double)(*seed) / (double)0x7fffffff);
}
static float lcg_randf_range(unsigned *seed, float lo, float hi) {
    return lo + (hi - lo) * lcg_randf(seed);
}

static void matmul_2d(const float *a, const float *b, float *c, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int p = 0; p < k; p++) s += a[i * k + p] * b[p * n + j];
            c[i * n + j] = s;
        }
}

static void layernorm(float *v, int dim) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += v[i];
    mean /= dim;
    float var = 0.0f;
    for (int i = 0; i < dim; i++) { float d = v[i] - mean; var += d * d; }
    var /= dim;
    float std = sqrtf(var + 1e-5f);
    for (int i = 0; i < dim; i++) v[i] = (v[i] - mean) / std;
}

/* gelu: use wubu_activations.h */

int wubu_imgenc_init(wubu_imgenc_t *v, unsigned seed) {
    if (!v) return -1;
    unsigned s = seed ? seed : 42;
    for (int i = 0; i < WUBU_IMGENC_PATCH_DIM; i++)
        for (int j = 0; j < WUBU_IMGENC_EMBED_DIM; j++)
            v->proj[i][j] = lcg_randf_range(&s, -0.1f, 0.1f);
    for (int i = 0; i < WUBU_IMGENC_N_TOKENS; i++)
        for (int j = 0; j < WUBU_IMGENC_EMBED_DIM; j++)
            v->pos[i][j] = lcg_randf_range(&s, -0.05f, 0.05f);
    for (int j = 0; j < WUBU_IMGENC_EMBED_DIM; j++)
        v->cls[j] = lcg_randf_range(&s, -0.05f, 0.05f);
    v->init = 1;
    return 0;
}

int wubu_imgenc_encode(const wubu_imgenc_t *v, const float *img, float *out) {
    if (!v || !v->init || !img || !out) return -1;
    const int P = WUBU_IMGENC_PATCH;
    const int C = WUBU_IMGENC_CHANNELS;
    const int sz = WUBU_IMGENC_IMAGE;
    const int npatches = WUBU_IMGENC_N_PATCHES;
    float patches[WUBU_IMGENC_N_PATCHES * WUBU_IMGENC_PATCH_DIM];
    int pi = 0;
    for (int py = 0; py < sz; py += P)
        for (int px = 0; px < sz; px += P) {
            int idx = 0;
            for (int dy = 0; dy < P; dy++)
                for (int dx = 0; dx < P; dx++)
                    for (int c = 0; c < C; c++) {
                        int iy = py + dy, ix = px + dx;
                        int src_idx = (iy * sz + ix) * C + c;
                        patches[pi * WUBU_IMGENC_PATCH_DIM + idx] = img[src_idx];
                        idx++;
                    }
            pi++;
        }
    float proj_out[WUBU_IMGENC_N_PATCHES * WUBU_IMGENC_EMBED_DIM];
    matmul_2d(patches, (const float *)v->proj, proj_out,
              npatches, WUBU_IMGENC_PATCH_DIM, WUBU_IMGENC_EMBED_DIM);
    for (int j = 0; j < WUBU_IMGENC_EMBED_DIM; j++) out[j] = v->cls[j];
    for (int i = 0; i < npatches; i++)
        for (int j = 0; j < WUBU_IMGENC_EMBED_DIM; j++)
            out[(i + 1) * WUBU_IMGENC_EMBED_DIM + j] = proj_out[i * WUBU_IMGENC_EMBED_DIM + j];
    for (int t = 0; t < WUBU_IMGENC_N_TOKENS; t++) {
        float *tok = &out[t * WUBU_IMGENC_EMBED_DIM];
        for (int j = 0; j < WUBU_IMGENC_EMBED_DIM; j++)
            tok[j] = gelu(tok[j] + v->pos[t][j]);
        layernorm(tok, WUBU_IMGENC_EMBED_DIM);
    }
    return 0;
}
