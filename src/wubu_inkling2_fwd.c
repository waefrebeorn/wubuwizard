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

/* Static depthwise causal short convolution over a sequence.
 *
 * Reference semantics (Inkling2LoopShortConvolution): nn.Conv1d(kernel=k, groups=channels,
 * padding=k-1, bias=False), i.e. output at step t sees x[t-k+1..t], followed by
 *     out = (conv(x) + x) / sqrt(k + 1)
 * The residual add and the 1/sqrt(k+1) scale are part of the block, not an optimisation detail:
 * dropping them changes the magnitude by ~2.2x.
 *
 * w is (k, channels) with w[0] weighting the current token (tap index == backward distance).
 */
void in2_shortconv(const float *x, int n_tok, int channels, const float *w, int k, float *out)
{
    const float scale = 1.0f / sqrtf((float)(k + 1));
    for (int t = 0; t < n_tok; ++t) {
        for (int c = 0; c < channels; ++c) {
            float acc = 0.0f;
            for (int tap = 0; tap < k; ++tap) {
                const int src = t - tap;             /* tap 0 = current token */
                if (src < 0) continue;               /* left padding with zeros */
                acc += w[(size_t)tap * channels + c] * x[(size_t)src * channels + c];
            }
            const float res = x[(size_t)t * channels + c];
            out[(size_t)t * channels + c] = (acc + res) * scale;
        }
    }
}

/* L2 clip, optionally preceded by a learned per-(head,dim) scale.
 *
 *   x *= scale;  row = x[h*dim .. h*dim+dim];  x *= min(1, threshold / (|row| + 1e-6))
 *
 * q uses this with a learned scale (attn_q_clip), k uses it bare, and the relative-position states use it
 * with rel_clip. The scale is stored in ggml order (dim, heads), i.e. scale[h*dim + d].
 */
void in2_l2_clip(float *x, const float *scale, int n_head, int dim, float threshold)
{
    for (int h = 0; h < n_head; ++h) {
        float *row = x + (size_t)h * dim;
        if (scale) for (int d = 0; d < dim; ++d) row[d] *= scale[(size_t)h * dim + d];
        double ss = 0.0;
        for (int d = 0; d < dim; ++d) ss += (double)row[d] * row[d];
        const float norm = sqrtf((float)ss);
        float f = threshold / (norm + 1e-6f);
        if (f > 1.0f) f = 1.0f;
        for (int d = 0; d < dim; ++d) row[d] *= f;
    }
}

/* Relative-position bias table: bias[h][dist] = sum_r rel_states[h][r] * P[dist][r].
 *
 * P is attn_rel_bias, stored in ggml order (d_rel, extent) so that r is the fastest axis:
 * P[dist * d_rel + r]. Computing the whole table once and indexing it by distance is what the reference
 * does, and it is cheaper than a gemm per (query,key) pair. Distances outside [0, extent) get 0, which
 * the caller handles by skipping them.
 */
void in2_rel_bias_table(const float *rel_states, const float *P, int n_head, int d_rel, int extent, float *bias)
{
    for (int h = 0; h < n_head; ++h) {
        const float *rs = rel_states + (size_t)h * d_rel;
        float *out = bias + (size_t)h * extent;
        for (int d = 0; d < extent; ++d) {
            const float *pr = P + (size_t)d * d_rel;
            double acc = 0.0;
            for (int r = 0; r < d_rel; ++r) acc += (double)rs[r] * pr[r];
            out[d] = (float)acc;
        }
    }
}

/* Read a block-aligned element range of a tensor and dequantise it. The MoE expert blocks are
 * (2*d_ff, d_model, n_expert) with expert slices contiguous and 256-aligned, so one expert can be read
 * without touching the other 127 (2.7 GB if dequantised whole). */
int in2_dequant_slice(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t elem_off, int64_t n_elems, float *dst)
{
    if (!ctx || !t || !t->info || !dst) return WUBU_INKLING2_ERR_MISSING_TENSOR;
    if (elem_off % 256 != 0) return WUBU_INKLING2_ERR_SHAPE;
    const int ty = t->info->ggml_type;
    const int64_t byte_off = gguf_raw_size(ty, elem_off);
    const int64_t nbytes   = gguf_raw_size(ty, n_elems);
    if (byte_off < 0 || nbytes <= 0) return WUBU_INKLING2_ERR_UNKNOWN_TYPE;
    unsigned char *raw = (unsigned char *)malloc((size_t)nbytes);
    if (!raw) return WUBU_INKLING2_ERR_IO;
    wubu_in2_tensor_t shifted = *t;
    shifted.file_offset = t->file_offset + byte_off;
    int rc = wubu_inkling2_read(ctx, &shifted, raw, nbytes);
    if (rc == WUBU_INKLING2_OK) gguf_dequantize(raw, ty, n_elems, dst);
    free(raw);
    return rc;
}

/* Softcap: tanh(x / c) * c. Bounds a value into (-c, c) without a hard clip. */
static float in2_softcap(float x, float c) { return tanhf(x / c) * c; }

/* Softcapped SwiGLU: the sigmoid sees the RAW gate; only the linear factor and `up` are softcapped.
 *   act = softcap(gate, 4.0) * sigmoid(gate) * softcap(up, 25.0)
 * Applying the softcap inside the sigmoid as well is a common misreading and changes the gate
 * non-linearly, so it is spelled out here. */
void in2_expert_act(const float *gate, const float *up, int n, float *out)
{
    for (int i = 0; i < n; ++i) {
        const float g = gate[i];
        const float sig = 1.0f / (1.0f + expf(-g));
        out[i] = in2_softcap(g, 4.0f) * sig * in2_softcap(up[i], 25.0f);
    }
}

/* Router: sigmoid scores, selection by score + correction bias, weights from the UNBIASED scores
 * renormalised to sum 1 and then multiplied by the routed scaling factor (8.0).
 *
 * The distinction matters: the correction bias only changes which experts are picked, never how much
 * they contribute. Returns the number of experts selected. */
int in2_router(const float *x, const float *W, const float *corr_bias, int n_expert, int d_model,
               int top_k, float scale, int *idx, float *w)
{
    float *scores = (float *)malloc((size_t)n_expert * sizeof(float));
    float *adj    = (float *)malloc((size_t)n_expert * sizeof(float));
    if (!scores || !adj) { free(scores); free(adj); return 0; }
    for (int e = 0; e < n_expert; ++e) {
        const float *row = W + (size_t)e * d_model;
        double acc = 0.0;
        for (int i = 0; i < d_model; ++i) acc += (double)row[i] * x[i];
        const float s = 1.0f / (1.0f + expf(-(float)acc));
        scores[e] = s;
        adj[e] = s + (corr_bias ? corr_bias[e] : 0.0f);
    }
    /* partial selection sort for top_k */
    char *taken = (char *)calloc((size_t)n_expert, 1);
    for (int k = 0; k < top_k; ++k) {
        int best = -1; float bestv = -1e30f;
        for (int e = 0; e < n_expert; ++e) if (!taken[e] && adj[e] > bestv) { bestv = adj[e]; best = e; }
        if (best < 0) break;
        taken[best] = 1; idx[k] = best;
    }
    double sum = 0.0;
    for (int k = 0; k < top_k; ++k) sum += scores[idx[k]];
    const float inv = (float)(1.0 / (sum + 1e-20));
    for (int k = 0; k < top_k; ++k) w[k] = scores[idx[k]] * inv * scale;
    free(scores); free(adj); free(taken);
    return top_k;
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
