/* wubu_inkling2_model.c -- the full depth path: embed -> layers -> final mix -> norm -> logits.
 *
 * Single token for now. At one token every attention is degenerate (one key plus the sink), so this
 * exercises the depth machinery without needing a KV cache or a sliding window.
 *
 * SCOPE: runs the 24 physical layers exactly once, i.e. loop pass 1 of cfg->n_loops. The remaining passes
 * need the message gates over the previous pass's hidden and block states, which are not implemented yet,
 * so in2_forward_token REFUSES when n_loops > 1 instead of quietly returning the output of a shallower
 * model. A partial graph that answers anyway is worse than one that says what it cannot do.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_inkling2.h"

void in2_rmsnorm_unweighted(const float *x, int n, float eps, float *out);
void in2_rmsnorm(const float *x, const float *w, int n, float eps, float *out);
void in2_matvec(const float *W, const float *x, int n_in, int n_out, float *y);
int  in2_layer_sliding(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                       int il, float *h, int n_tok, int pos0);
int  in2_layer_full(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                    int il, float *h, int n_tok, int pos0, const float *blocks, int n_blocks);
int  in2_embed(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
               int token, float *out);
float *in2_tensor_to_f32(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t *n_out);

static float *global_f32(gguf_ctx *ctx, const char *name)
{
    gguf_tensor_info *info = NULL;
    if (wubu_inkling2_global_tensor(ctx, name, &info) != 1 || !info) return NULL;
    wubu_in2_tensor_t t;
    memset(&t, 0, sizeof t);
    t.info = info;
    t.file_offset = (int64_t)ctx->data_blob_offset + (int64_t)info->data_offset;
    return in2_tensor_to_f32(ctx, &t, NULL);
}

/* Depth mix: one learned query over [block_states; partial] after an unweighted RMS norm. With one state
 * and one query it degenerates to a bare norm - which is why layer 0's mlp_input is a norm in disguise. */
static void in2_mix(const float *blocks, int n_blocks, const float *partial, const float *query,
                    int d, float eps, float *out)
{
    const int n = n_blocks + 1;
    float *st = (float *)malloc((size_t)n * d * sizeof(float));
    float *nr = (float *)malloc((size_t)n * d * sizeof(float));
    float *al = (float *)malloc((size_t)n * sizeof(float));
    if (!st || !nr || !al) { free(st); free(nr); free(al); return; }
    for (int i = 0; i < n_blocks; ++i) memcpy(st + (size_t)i * d, blocks + (size_t)i * d, sizeof(float) * d);
    memcpy(st + (size_t)n_blocks * d, partial, sizeof(float) * d);
    for (int i = 0; i < n; ++i) in2_rmsnorm_unweighted(st + (size_t)i * d, d, eps, nr + (size_t)i * d);
    const float inv = 1.0f / sqrtf((float)d);
    float mx = -1e30f;
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int j = 0; j < d; ++j) acc += (double)query[j] * nr[(size_t)i * d + j];
        al[i] = (float)acc * inv;
        if (al[i] > mx) mx = al[i];
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) { al[i] = expf(al[i] - mx); sum += al[i]; }
    for (int i = 0; i < n; ++i) al[i] = (float)(al[i] / sum);
    for (int j = 0; j < d; ++j) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) acc += (double)al[i] * nr[(size_t)i * d + j];
        out[j] = (float)acc;
    }
    free(st); free(nr); free(al);
}

int in2_forward_token(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                      int token, float *h, float *logits)
{
    /* IN2_SINGLE_PASS runs pass 1 only. It exists to exercise the depth path before the message gates
     * land; anything it prints is a 1-of-3-pass number and must be labelled as such. */
    if (cfg->n_loops > 1 && !getenv("IN2_SINGLE_PASS")) {
        fprintf(stderr, "in2_forward_token: n_loops=%d, message gates not implemented (single pass only)\n",
                cfg->n_loops);
        return WUBU_INKLING2_ERR_NO_LOOP;
    }
    const int d = cfg->d_model;
    const float eps = cfg->rms_eps;

    float *emb = (float *)calloc((size_t)d, sizeof(float));
    if (!emb) return WUBU_INKLING2_ERR_IO;
    int rc = in2_embed(ctx, m, cfg, token, emb);
    if (rc != WUBU_INKLING2_OK) { free(emb); return rc; }
    float *en = global_f32(ctx, "token_embd_norm.weight");
    if (en) { in2_rmsnorm(emb, en, d, eps, h); free(en); }
    else    memcpy(h, emb, sizeof(float) * d);
    free(emb);

    float *blocks = (float *)calloc((size_t)16 * d, sizeof(float));
    int n_blocks = 0;
    if (!blocks) return WUBU_INKLING2_ERR_IO;

    for (int il = 0; il < cfg->n_layer_phys; ++il) {
        float *br = (float *)malloc(sizeof(float) * d);
        if (!br) { free(blocks); return WUBU_INKLING2_ERR_IO; }
        memcpy(br, h, sizeof(float) * d);
        /* The four full-attention layers (il % 6 == 5) are structurally different: three depth-mixed
         * states and three input norms. Sliding layers still use their single-block approximation of
         * mlp_input; that is a known gap, recorded in the port doc. */
        if (il % cfg->attn_res_block == cfg->attn_res_block - 1)
            rc = in2_layer_full(ctx, m, cfg, il, h, 1, 0, blocks, n_blocks);
        else
            rc = in2_layer_sliding(ctx, m, cfg, il, h, 1, 0);
        if (rc != WUBU_INKLING2_OK) { free(br); free(blocks); return rc; }
        if (il % cfg->attn_res_block == 0 && n_blocks < 16) {
            memcpy(blocks + (size_t)n_blocks * d, br, sizeof(float) * d);
            ++n_blocks;
        }
        free(br);
    }

    float *fin = (float *)malloc(sizeof(float) * d);
    float *fq = global_f32(ctx, "output_res_query.weight");
    if (!fin) { free(blocks); free(fq); return WUBU_INKLING2_ERR_IO; }
    if (fq) in2_mix(blocks, n_blocks, h, fq, d, eps, fin);
    else    memcpy(fin, h, sizeof(float) * d);

    float *on = global_f32(ctx, "output_norm.weight");
    if (on) { in2_rmsnorm(fin, on, d, eps, h); free(on); }
    else    memcpy(h, fin, sizeof(float) * d);

    float *Wo = global_f32(ctx, "output.weight");
    if (!Wo) { free(blocks); free(fin); free(fq); return WUBU_INKLING2_ERR_IO; }
    in2_matvec(Wo, h, d, cfg->vocab_size, logits);
    const float cap = 30.0f;
    for (int i = 0; i < cfg->vocab_size; ++i) logits[i] = tanhf(logits[i] / cap) * cap;

    free(Wo); free(blocks); free(fin); free(fq);
    return WUBU_INKLING2_OK;
}
