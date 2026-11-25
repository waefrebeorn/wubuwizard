/* wubu_inkling2_fwd.c -- EG-118 step 3: numeric primitives and the embedding path.
 *
 * Deliberately unoptimised: every weight is dequantised to f32 and multiplied with a plain dot product.
 * That is far too slow to generate text with, but it makes the graph checkable, and each primitive here
 * is written to be validated independently before anything is built on top of it.
 *
 * Tensor layout follows ggml: dims[0] is the fastest-varying axis, so a linear weight with
 * dims (n_in, n_out) is row-major as W[o * n_in + i].
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_inkling2.h"

/* Unweighted RMS norm: x * rsqrt(mean(x^2) + eps). Used by the loop message gates. */
void in2_rmsnorm_unweighted(const float *x, int n, float eps, float *out)
{
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale;
}

/* Weighted RMS norm, the form used by attn_norm / ffn_norm / the full-layer input norms. */
void in2_rmsnorm(const float *x, const float *w, int n, float eps, float *out)
{
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

/* y[o] = dot(W + o*n_in, x) for o in [0, n_out). */
void in2_matvec(const float *W, const float *x, int n_in, int n_out, float *y)
{
    for (int o = 0; o < n_out; ++o) {
        const float *row = W + (size_t)o * n_in;
        double acc = 0.0;
        for (int i = 0; i < n_in; ++i) acc += (double)row[i] * x[i];
        y[o] = (float)acc;
    }
}

/* Dequantise a whole tensor into f32. Caller frees. Returns NULL on failure. */
float *in2_tensor_to_f32(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t *n_out)
{
    if (!ctx || !t || !t->info) return NULL;
    int64_t ne = 1;
    for (int d = 0; d < t->info->n_dims; ++d) ne *= t->info->dims[d];
    float *buf = (float *)malloc((size_t)ne * sizeof(float));
    if (!buf) return NULL;
    if (wubu_inkling2_read_f32(ctx, t, buf, ne) != WUBU_INKLING2_OK) { free(buf); return NULL; }
    if (n_out) *n_out = ne;
    return buf;
}

/* Token embedding lookup: token_embd is (n_embd, n_vocab) so token t occupies columns t.
 * Rows are the embedding axis, which is dims[0], so the gather is strided over the vocab axis. */
int in2_embed(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
              int token, float *out)
{
    if (!m->tok_embd.info) return WUBU_INKLING2_ERR_MISSING_TENSOR;
    const gguf_tensor_info *ti = m->tok_embd.info;
    if (ti->n_dims != 2 || ti->dims[0] != (int64_t)cfg->d_model) return WUBU_INKLING2_ERR_SHAPE;
    if (token < 0 || token >= cfg->vocab_size) return WUBU_INKLING2_ERR_SHAPE;

    /* The quantised rows are per-token columns; the cheapest correct path is to dequantise the whole
     * tensor once. A streaming reader will replace this, but correctness first. */
    int64_t ne = 0;
    float *emb = in2_tensor_to_f32(ctx, &m->tok_embd, &ne);
    if (!emb) return WUBU_INKLING2_ERR_IO;
    const int64_t stride = ti->dims[0];
    for (int64_t i = 0; i < stride; ++i) out[i] = emb[(int64_t)token * stride + i];
    free(emb);
    return WUBU_INKLING2_OK;
}
