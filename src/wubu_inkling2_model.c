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
#include <math.h>
#include <string.h>
#include <math.h>

#include "wubu_inkling2.h"
#include "wubu_inkling2_kv.h"
int in2_matvec_quant(gguf_ctx *, const wubu_in2_tensor_t *, int64_t, const float *, int, int, float *);

void in2_rmsnorm_unweighted(const float *x, int n, float eps, float *out);
void in2_rmsnorm(const float *x, const float *w, int n, float eps, float *out);
void in2_matvec(const float *W, const float *x, int n_in, int n_out, float *y);
int  in2_layer_sliding(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                       int il, float *h, int n_tok, int pos0, const float *blocks, int n_blocks,
                       const float *msgs);
void in2_messages(const float *prev_blocks, int n_prev, const float *prev_h,
                  const float *qres, const float *qkv, int li, int d, float eps, float *out);
int  in2_layer_full_seq(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                        int il, float *h, int n_tok, int pos0, const float *blocks, int n_blocks,
                        const float *msgs);
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
static void in2_mix(const float *blocks, int n_blocks, int block_stride, int pos_off,
                    const float *partial, const float *query, int d, float eps, float *out)
{
    const int n = n_blocks + 1;
    float *st = (float *)malloc((size_t)n * d * sizeof(float));
    float *nr = (float *)malloc((size_t)n * d * sizeof(float));
    float *al = (float *)malloc((size_t)n * sizeof(float));
    if (!st || !nr || !al) { free(st); free(nr); free(al); return; }
    for (int i = 0; i < n_blocks; ++i)
        memcpy(st + (size_t)i * d, blocks + (size_t)i * block_stride + pos_off, sizeof(float) * d);
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

/* pos0 = index of the first token in `tokens`. With a cache attached the layers only compute those positions and
 * read everything earlier from the cache; with pos0 = 0 and no cache this is the original full recompute. */
int in2_forward_at(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                   const int *tokens, int n_tok, int pos0, float *h, float *logits)
{
    in2_kv_begin(pos0);
    /* IN2_SINGLE_PASS runs pass 1 only. It exists to exercise the depth path before the message gates
     * land; anything it prints is a 1-of-3-pass number and must be labelled as such. */
    /* IN2_SINGLE_PASS=1 still forces one pass, which is useful for A/B against the pre-message graph. */
    const int n_loops = getenv("IN2_SINGLE_PASS") ? 1 : cfg->n_loops;
    (void)n_loops;
    const int d = cfg->d_model;
    const float eps = cfg->rms_eps;

    const size_t ND = (size_t)n_tok * d;
    float *emb = (float *)calloc(ND, sizeof(float));
    if (!emb) return WUBU_INKLING2_ERR_IO;
    int rc = WUBU_INKLING2_OK;
    float *en = global_f32(ctx, "token_embd_norm.weight");
    for (int t = 0; t < n_tok && rc == WUBU_INKLING2_OK; ++t) {
        rc = in2_embed(ctx, m, cfg, tokens[t], emb + (size_t)t * d);
        if (rc != WUBU_INKLING2_OK) break;
        if (en) in2_rmsnorm(emb + (size_t)t * d, en, d, eps, h + (size_t)t * d);
        else    memcpy(h + (size_t)t * d, emb + (size_t)t * d, sizeof(float) * d);
    }
    free(en); free(emb);
    if (rc != WUBU_INKLING2_OK) return rc;

    /* block states hold a whole sequence per entry: 16 entries x n_tok positions */
    float *blocks = (float *)calloc((size_t)16 * ND, sizeof(float));
    int n_blocks = 0;
    if (!blocks) return WUBU_INKLING2_ERR_IO;

    float *qres = global_f32(ctx, "loop_res_query.weight");
    float *qkv  = global_f32(ctx, "loop_kv_res_query.weight");
    float *msgs = (float *)calloc((size_t)4 * d, sizeof(float));
    float *pre_h = (float *)malloc(sizeof(float) * ND);
    float *prev_h = (float *)malloc(sizeof(float) * ND);
    float *prev_blocks = (float *)calloc((size_t)16 * ND, sizeof(float));
    int pre_blocks = 0, n_prev = 0;
    if (!msgs || !pre_h || !prev_h || !prev_blocks) rc = WUBU_INKLING2_ERR_IO;

    /* helper: run one layer and bank the block residual at boundaries */
    int ltseq = 0;
    #define RUN_LAYER(IL, MSG) do {                                                    \
        float *_br = (float *)malloc(sizeof(float) * ND);                              \
        if (!_br) { rc = WUBU_INKLING2_ERR_IO; break; }                                \
        memcpy(_br, h, sizeof(float) * ND);                                            \
        if (rc == WUBU_INKLING2_OK && (IL) % cfg->attn_res_block == 0 && n_blocks < 16) { \
            memcpy(blocks + (size_t)n_blocks * ND, _br, sizeof(float) * ND); ++n_blocks; \
        }                                                                               \
        if ((IL) % cfg->attn_res_block == cfg->attn_res_block - 1)                     \
            rc = in2_layer_full_seq(ctx, m, cfg, (IL), h, n_tok, pos0, blocks, n_blocks, (MSG)); \
        else                                                                            \
            rc = in2_layer_sliding(ctx, m, cfg, (IL), h, n_tok, pos0, blocks, n_blocks, (MSG)); \
        if (rc == WUBU_INKLING2_OK && getenv("IN2_LTRACE")) {                           \
            const int _t = n_tok - 1;                                                   \
            double _s = 0.0;                                                            \
            for (int _i = 0; _i < d; ++_i) { double _v = h[(size_t)_t * d + _i]; _s += _v * _v; } \
            fprintf(stderr, "LT seq=%d il=%02d pos0=%d ntok=%d l2_last=%.12e\n", ltseq++, (IL), pos0, n_tok, _s); \
        }                                                                               \
        if (rc == WUBU_INKLING2_OK && getenv("IN2_DUMP_LAYERS")) {                     \
            char _lp[512]; snprintf(_lp, sizeof _lp, "%s/L%02d.bin", getenv("IN2_DUMP_LAYERS"), (IL)); \
            FILE *_lf = fopen(_lp, "wb");                                              \
            if (_lf) { fwrite(h, sizeof(float), (size_t)ND, _lf); fclose(_lf); }        \
        }                                                                               \
        free(_br);                                                                      \
    } while (0)

    /* pre-loop layers */
    for (int il = 0; il < cfg->loop_core_start && rc == WUBU_INKLING2_OK; ++il) RUN_LAYER(il, NULL);
    if (rc == WUBU_INKLING2_OK) { memcpy(pre_h, h, sizeof(float) * ND); pre_blocks = n_blocks; }

    /* first pass over the loop core: no messages yet */
    for (int il = cfg->loop_core_start; il < cfg->loop_core_end && rc == WUBU_INKLING2_OK; ++il)
        RUN_LAYER(il, NULL);

    /* LoopCD (arXiv 2610.02185, Eq. 2): the first recurrent state h_1 is a WEAK prediction of the same next
     * token, because it embodies less computation than h_R. Standard decoding throws it away. Keeping it lets
     * us extrapolate away from the weak prediction before the coda, at zero extra head passes:
     *     h' = h_R + w * (h_R - h_1)
     * w = 0 must reproduce the baseline bit for bit - that is this feature's instrument check. */
    float *h1 = NULL;
    if (getenv("IN2_LOOPCD") && n_loops > 1 && rc == WUBU_INKLING2_OK) {
        h1 = (float *)malloc(sizeof(float) * ND);
        if (h1) memcpy(h1, h, sizeof(float) * ND);
    }

    /* each later pass restarts from the pre-loop state and carries only the messages */
    for (int pass = 1; pass < n_loops && rc == WUBU_INKLING2_OK; ++pass) {
        memcpy(prev_h, h, sizeof(float) * ND);
        /* the reference skips the pre-loop blocks plus the first block appended by the previous pass */
        const int skip = pre_blocks + 1;
        n_prev = (n_blocks > skip) ? (n_blocks - skip) : 0;
        for (int i = 0; i < n_prev; ++i)
            memcpy(prev_blocks + (size_t)i * ND, blocks + (size_t)(skip + i) * ND, sizeof(float) * ND);

        memcpy(h, pre_h, sizeof(float) * ND);
        n_blocks = pre_blocks;
        for (int il = cfg->loop_core_start; il < cfg->loop_core_end && rc == WUBU_INKLING2_OK; ++il) {
            const int li = il - cfg->loop_core_start;
            if (qres && qkv) {
                in2_messages(prev_blocks, n_prev, prev_h, qres, qkv, li, d, eps, msgs);
                if (getenv("IN2_MTRACE") && li == 0) {
                    const int _t = n_tok - 1;
                    double _m = 0.0;
                    for (int _i = 0; _i < d; ++_i) { double _v = msgs[(size_t)_t * d + _i]; _m += _v * _v; }
                    fprintf(stderr, "MSG pass=%d li=%d n_prev=%d pos0=%d ntok=%d l2_last=%.12e\n",
                            pass, li, n_prev, pos0, n_tok, _m);
                }
            }
            RUN_LAYER(il, (qres && qkv) ? msgs : NULL);
        }
    }

    if (h1 && rc == WUBU_INKLING2_OK) {
        const float w = (float)atof(getenv("IN2_LOOPCD"));
        for (size_t i = 0; i < (size_t)ND; ++i) h[i] = h[i] + w * (h[i] - h1[i]);
        if (getenv("IN2_VERBOSE"))
            fprintf(stderr, "LoopCD-Hidden: w=%g over %d positions (h_R extrapolated away from h_1)\n", w, n_tok);
        free(h1); h1 = NULL;
    }

    /* post-loop layers */
    for (int il = cfg->loop_core_end; il < cfg->n_layer_phys && rc == WUBU_INKLING2_OK; ++il)
        RUN_LAYER(il, NULL);
    #undef RUN_LAYER

    free(qres); free(qkv); free(msgs); free(pre_h); free(prev_h); free(prev_blocks);
    if (rc != WUBU_INKLING2_OK) { free(blocks); return rc; }

    float *fin = (float *)malloc(sizeof(float) * d);
    float *fq = global_f32(ctx, "output_res_query.weight");
    if (!fin) { free(blocks); free(fq); return WUBU_INKLING2_ERR_IO; }
    /* logits are only meaningful for the last position */
    const float *last = h + (size_t)(n_tok - 1) * d;
    if (fq) in2_mix(blocks, n_blocks, n_tok * d, (n_tok - 1) * d, last, fq, d, eps, fin);
    else    memcpy(fin, last, sizeof(float) * d);

    float *on = global_f32(ctx, "output_norm.weight");
    float out_h[1];
    (void)out_h;
    if (on) { in2_rmsnorm(fin, on, d, eps, fin); free(on); }
    if (getenv("IN2_DUMP_H")) {
        const char *hp = getenv("IN2_DUMP_H");
        FILE *hf = fopen(hp, "wb");
        if (hf) { fwrite(fin, sizeof(float), (size_t)d, hf); fclose(hf); }
        fprintf(stderr, "dumped final h (%d floats) to %s\n", d, hp);
    }


    /* LM head: 201024 x 2304 = 463M MACs per token, previously done as a scalar double dot over a 1.85 GB float
     * copy of the matrix. output.weight is Q8_0, so it goes through the translated AVX2 kernel in row chunks -
     * chunked because the kernel path reads a whole row block at a time and one call over all 201024 rows would
     * malloc ~492 MB. elem_off is a multiple of both the weight block (32) and the activation block, and d = 2304
     * keeps it a multiple of 64, so every chunk is legal for the kernels. Falls back to the float path unchanged
     * when the kernels are unavailable, so the old numbers stay reproducible with IN2_NOFAST=1. */
    int head_done = 0;
    if (!getenv("IN2_NOFAST")) {
        gguf_tensor_info *oinfo = NULL;
        if (wubu_inkling2_global_tensor(ctx, "output.weight", &oinfo) == 1 && oinfo) {
            wubu_in2_tensor_t ot;
            memset(&ot, 0, sizeof ot);
            ot.info = oinfo;
            ot.file_offset = (int64_t)ctx->data_blob_offset + (int64_t)oinfo->data_offset;
            const int CHUNK = 8192;
            head_done = 1;
            for (int r = 0; r < cfg->vocab_size; r += CHUNK) {
                const int n = (r + CHUNK <= cfg->vocab_size) ? CHUNK : (cfg->vocab_size - r);
                if (in2_matvec_quant(ctx, &ot, (int64_t)r * d, fin, n, d, logits + r) != WUBU_INKLING2_OK) {
                    head_done = 0;
                    break;
                }
            }
            if (!head_done) fprintf(stderr, "lm_head kernel path failed at vocab %d, falling back\n", cfg->vocab_size);
        }
    }
    if (!head_done) {
        float *Wo = global_f32(ctx, "output.weight");
        if (!Wo) { free(blocks); free(fin); free(fq); return WUBU_INKLING2_ERR_IO; }
        in2_matvec(Wo, fin, d, cfg->vocab_size, logits);
        free(Wo);
    }
    const float cap = 30.0f;
    for (int i = 0; i < cfg->vocab_size; ++i) logits[i] = tanhf(logits[i] / cap) * cap;

    /* The output matrix has 201024 rows but only 200058 are real tokens; the rest are padding rows whose
     * embeddings are meaningless. The reference drops the padded vocab before sampling, and without this
     * the first generation run selected id 200567 - an empty padding row - as its second token. */
    if (cfg->vocab_size_unpadded > 0 && cfg->vocab_size_unpadded < cfg->vocab_size)
        for (int i = cfg->vocab_size_unpadded; i < cfg->vocab_size; ++i) logits[i] = -1e30f;

    free(blocks); free(fin); free(fq);
    return WUBU_INKLING2_OK;
}

/* Full-prefix entry point: unchanged contract, so every existing caller keeps working. */
int in2_forward_seq(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                    const int *tokens, int n_tok, float *h, float *logits)
{
    return in2_forward_at(ctx, m, cfg, tokens, n_tok, 0, h, logits);
}
