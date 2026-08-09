/*
 * wubu.c -- WuBu-35M in C11. THE MUSTARD SEED: our own base model.
 *
 * Original WaefreBeorn work under the WaefreBeorn Umbrella License v3.0
 * (LICENSE at the repo root). Designed and implemented in-house: the
 * architecture, the C11 implementation, and the trained weights are
 * first-party. The forward pass:
 *   x = embed(tokens)
 *   for each layer: x += attn(rmsnorm(x)); x += swiglu(rmsnorm(x))
 *   every 4th layer: x = selectors[i](checkpoint, x)  (convex softmax)
 *   logits = lm_head(final_norm(x))   (tied embeddings)
 *
 * Attention rhythm: layer (i+1) % 4 == 0 is FULL; the others are LOCAL
 * with a 256-token causal window. Partial RoPE rotates the first 32 of
 * 64 head dims.
 *
 * NOTE (2026-08-06): this module is the archived WuBu-35M spine,
 * superseded by the WuBu1 redesign (docs/wubu1-base-model-design.md).
 * Kept for lineage + the role-resolver fixture; new work is WuBu1.
 */
#include "wubu.h"
#include "wubu35_dims.h"
#include "wubu_foldmath.h"
#include "safetensors_reader.h"
#include "wubu_moe2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- tiny helpers (freestanding-friendly) ---- */
static float rms_norm_value(float *out, const float *x, const float *w,
                            int n, float eps)
{
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
    return r;
}

static float silu(float v) { return v / (1.0f + expf(-v)); }

/* the partial RoPE table (matching the reference exactly). */
static void build_rope_tables(float *cos_tbl, float *sin_tbl, int max_seq,
                              int rope_dim, float theta)
{
    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < rope_dim / 2; i++) {
            float inv = powf(theta, -(float)(2 * i) / (float)rope_dim);
            float ang = (float)pos * inv;
            float s, c;
            wubu_fold_sincos(ang, &s, &c);   /* the folded math: no libm,
                                                deterministic, portable to
                                                the GPU kernels + bare metal */
            cos_tbl[pos * rope_dim + i] = c;
            cos_tbl[pos * rope_dim + rope_dim / 2 + i] = c;
            sin_tbl[pos * rope_dim + i] = s;
            sin_tbl[pos * rope_dim + rope_dim / 2 + i] = s;
        }
    }
}

static void apply_rope(float *qk, int seq, int head_dim, int rope_dim,
                       const float *cos_tbl, const float *sin_tbl, int pos0)
{
    /* qk layout: [seq, head_dim]; rotate the first rope_dim channels */
    for (int s = 0; s < seq; s++) {
        float *row = qk + (size_t)s * head_dim;
        const float *c = cos_tbl + (size_t)(pos0 + s) * rope_dim;
        const float *si = sin_tbl + (size_t)(pos0 + s) * rope_dim;
        for (int i = 0; i < rope_dim / 2; i++) {
            float x0 = row[i], x1 = row[rope_dim / 2 + i];
            row[i] = x0 * c[i] - x1 * si[i];
            row[rope_dim / 2 + i] = x0 * si[i] + x1 * c[i];
        }
    }
}

/* ---- the model init ---- */
int wubu_model_init(wubu_model_t *m, float *embedding, float *final_norm,
                     wubu_block_t *blocks, float **selectors)
{
    if (!m || !embedding || !final_norm || !blocks || !selectors) return -1;
    memset(m, 0, sizeof(*m));
    m->embedding = embedding;
    m->final_norm = final_norm;
    for (int i = 0; i < WUBU_LAYERS; i++) {
        m->blocks[i] = blocks[i];
        m->is_full[i] = ((i + 1) % WUBU_FULL_EVERY == 0) ? 1 : 0;
        m->fire_sel[i] = ((i + 1) % WUBU_SELECT_EVERY == 0) ? 1 : 0;
    }
    m->n_layers = WUBU_LAYERS;
    for (int i = 0; i < WUBU_SELECTORS; i++) m->selectors[i] = selectors[i];
    return 0;
}

/* ---- the safetensors loader ---- */
/* Load a row-major [rows, cols] weight matrix with alignment padding.
 * The checkpoint stores chk_rows × chk_cols; the aligned engine needs
 * out_rows × out_cols. We calloc the output (zeroed), read the checkpoint's
 * chk_rows × chk_cols values, then scatter-copy into the first chk_rows
 * rows and first chk_cols columns. The pad rows/cols are dead (zero input
 * → zero activation) but keep every quant block (QK_K=256) tileable.
 * See Theory/08 (Aligned Rewrite). */

/* Original loader for tensors whose dimensions are already aligned (norms,
   gate_up where chk_cols == out_cols). */
static float *load_tensor(st_ctx *r, const char *name, size_t expect_elems)
{
    const st_tensor_info *info = st_find_tensor(r, name);
    if (!info) {
        fprintf(stderr, "wubu: missing tensor %s\n", name);
        return NULL;
    }
    if ((size_t)info->n_elems != expect_elems) {
        fprintf(stderr, "wubu: %s has %lld elems, expected %zu\n",
                name, (long long)info->n_elems, expect_elems);
        return NULL;
    }
    float *buf = (float *)malloc(expect_elems * sizeof(float));
    if (!buf) return NULL;
    if (st_read_tensor_f32(r, info, buf, (int64_t)expect_elems) !=
        (int)expect_elems) {
        free(buf);
        return NULL;
    }
    return buf;
}

/* ---- the from-scratch random-init builder (the amoeba doctrine:
 * 'delete the old model, make a new model' — dims are data, the fresh
 * model is born at the runtime WUBU35_DIMS geometry, zero pretrained
 * weights. Every matrix gets the GPT-2-style init (N(0, 0.02), scaled
 * by 1/sqrt(2*n_layers) for the residual path). ---- */

/* the alloc/rand helpers (xorshift32 + Box-Muller Gaussian) */
static void *malloc_f(size_t n)   { void *p = malloc(n ? n : 1); return p; }
static float *calloc_f(size_t n)  { return (float *)calloc(n ? n : 1, sizeof(float)); }
static unsigned xrs_next(unsigned *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s;
}
static float wubu_randn(unsigned *s) {
    /* Box-Muller: two uniforms -> a standard normal */
    float u1 = (xrs_next(s) & 0xFFFFFFu) / 16777216.0f + 1e-9f;
    float u2 = (xrs_next(s) & 0xFFFFFFu) / 16777216.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}
static float *wubu_rand_mat(size_t rows, size_t cols, float std, unsigned *s) {
    float *w = (float *)malloc(rows * cols * sizeof(float));
    if (!w) return NULL;
    for (size_t i = 0; i < rows * cols; i++) w[i] = wubu_randn(s) * std;
    return w;
}

int wubu_model_random_init(wubu_model_t *m)
{
    if (!m) return -1;
    memset(m, 0, sizeof(*m));
    /* the runtime dims must be live BEFORE any allocation (the amoeba
     * doctrine: dims are data). Use the aligned WuBu1 geometry default
     * unless a probe already set it. */
    if (WUBU35_DIMS.dim == 0) wubu35_dims_default();
    unsigned seed = (unsigned)(time(NULL) ^ (uintptr_t)m);
    float *embedding = malloc_f((size_t)WUBU_VOCAB * WUBU_DIM * sizeof(float));
    float *final_norm = calloc_f((size_t)WUBU_DIM);
    wubu_block_t *blocks = calloc((size_t)WUBU_MAX_LAYERS, sizeof(wubu_block_t));
    float **selectors = calloc((size_t)(WUBU_SELECTORS > 0 ? WUBU_SELECTORS : 1),
                               sizeof(float *));
    if (!embedding || !final_norm || !blocks || !selectors) return -1;
    float rscale = 1.0f / sqrtf(2.0f * (float)WUBU_LAYERS);
    for (int i = 0; i < WUBU_LAYERS; i++) {
        wubu_block_t *blk = &blocks[i];
        blk->attn_norm = calloc_f((size_t)WUBU_DIM);
        blk->q_norm = calloc_f((size_t)WUBU_HEAD_DIM);
        blk->k_norm = calloc_f((size_t)WUBU_HEAD_DIM);
        blk->ffn_norm = calloc_f((size_t)WUBU_DIM);
        blk->q_proj = wubu_rand_mat((size_t)WUBU_DIM, (size_t)WUBU_HEADS * WUBU_HEAD_DIM, 0.02f * rscale, &seed);
        blk->k_proj = wubu_rand_mat((size_t)WUBU_DIM, (size_t)WUBU_KV_HEADS * WUBU_HEAD_DIM, 0.02f * rscale, &seed);
        blk->v_proj = wubu_rand_mat((size_t)WUBU_DIM, (size_t)WUBU_KV_HEADS * WUBU_HEAD_DIM, 0.02f * rscale, &seed);
        blk->o_proj = wubu_rand_mat((size_t)WUBU_HEADS * WUBU_HEAD_DIM, (size_t)WUBU_DIM, 0.02f * rscale, &seed);
        blk->g_proj = wubu_rand_mat((size_t)WUBU_DIM, (size_t)WUBU_DIM, 0.02f * rscale, &seed);
        blk->gate_up = wubu_rand_mat((size_t)WUBU_DIM, (size_t)2 * WUBU_FFN_DIM, 0.02f * rscale, &seed);
        blk->down = wubu_rand_mat((size_t)WUBU_FFN_DIM, (size_t)WUBU_DIM, 0.02f * rscale, &seed);
    }
    for (int i = 0; i < WUBU_VOCAB; i++)
        for (int d = 0; d < WUBU_DIM; d++)
            embedding[(size_t)i * WUBU_DIM + d] = wubu_randn(&seed) * 0.02f;
    for (int i = 0; i < WUBU_SELECTORS; i++)
        selectors[i] = calloc_f((size_t)WUBU_DIM);
    for (int i = 0; i < WUBU_DIM; i++) final_norm[i] = 1.0f;
    for (int i = 0; i < WUBU_LAYERS; i++) {
        /* per-layer norms start at 1.0 (the calloc'd ones) */
        for (int d = 0; d < WUBU_DIM; d++) {
            blocks[i].attn_norm[d] = 1.0f;
            blocks[i].ffn_norm[d] = 1.0f;
        }
        for (int d = 0; d < WUBU_HEAD_DIM; d++) {
            blocks[i].q_norm[d] = 1.0f;
            blocks[i].k_norm[d] = 1.0f;
        }
    }
    return wubu_model_init(m, embedding, final_norm, blocks, selectors);
}

int wubu_load(wubu_model_t *m, const char *path)
{
    if (!m || !path) return -1;

    /* THE REVOLVER DOCTRINE (Theory/06): the loader is self-describing.
     * Probe the checkpoint's own geometry and set the runtime dims to
     * the checkpoint's REAL dims BEFORE reading tensors. The old
     * behavior required every caller to do the probe dance and crashed
     * with "expected 0" otherwise (a pre-existing bug family:
     * test_wubu_train, wubu_train_cli, wubu_live_learn, test_diag
     * oracle 7, test_wubu...).
     *
     * The probe reads the file's true geometry (vocab/dim/heads/ffn
     * from the actual tensors). We set those REAL dims — not an
     * aligned override — so the file loads EXACTLY as it is: agnostic,
     * no zero-padding, no dual path. New WuBu1 checkpoints are born
     * aligned (512) so real == aligned for them; a legacy 448 seed
     * loads at 448 exactly. */
    {
        wubu35_dims_t d;
        if (wubu35_dims_probe(path, &d) == 0) {
            /* the probe's ckpt_* fields hold the file's unaligned
             * truth; the top-level fields may hold an aligned override
             * (Theory/08). For an agnostic exact load, use the truth. */
            if (d.ckpt_dim > 0)      { d.dim      = d.ckpt_dim;      }
            if (d.ckpt_ffn_dim > 0)  { d.ffn_dim  = d.ckpt_ffn_dim;  }
            if (d.ckpt_head_dim > 0) { d.head_dim = d.ckpt_head_dim; }
            if (d.ckpt_heads > 0)    { d.heads    = d.ckpt_heads;    }
            if (d.ckpt_kv_heads > 0) { d.kv_heads = d.ckpt_kv_heads; }
            wubu35_dims_set(&d);
        }
    }

    st_ctx *r = st_open(path);
    if (!r) return -1;

    /* The runtime dims were set from the checkpoint's REAL geometry
     * above (agnostic exact load — no zero-padding, no dual path).
     * Every expect-size below is exactly the file's size. */
    int q_out = WUBU_HEADS * WUBU_HEAD_DIM;

    float *embedding = load_tensor(r, "embedding.weight", (size_t)WUBU_VOCAB * WUBU_DIM);
    if (!embedding) { st_close(r); return -1; }
    float *final_norm = load_tensor(r, "final_norm.weight", WUBU_DIM);
    if (!final_norm) { free(embedding); st_close(r); return -1; }

    wubu_block_t blocks[WUBU_LAYERS];
    memset(blocks, 0, sizeof(blocks));
    char name[128];
    int active_layers = 0;
    for (int i = 0; i < WUBU_LAYERS; i++) {
        snprintf(name, sizeof(name), "layers.%d.attn.q_proj.weight", i);
        if (!st_find_tensor(r, name)) break;
        active_layers = i + 1;
    }
    if (active_layers == 0) { st_close(r); return -1; }
    int ok = 1;
    for (int i = 0; i < active_layers && ok; i++) {
        wubu_block_t *blk = &blocks[i];
        /* All weight matrices are [out, in] row-major; the runtime
         * dims were set from the checkpoint's real geometry, so the
         * expect-sizes are exact (agnostic loader, no padding). */
        snprintf(name, sizeof(name), "layers.%d.attn.q_proj.weight", i);
        blk->q_proj   = load_tensor(r, name, (size_t)q_out * WUBU_DIM);
        snprintf(name, sizeof(name), "layers.%d.attn.k_proj.weight", i);
        blk->k_proj   = load_tensor(r, name, (size_t)WUBU_KV_HEADS * WUBU_HEAD_DIM * WUBU_DIM);
        snprintf(name, sizeof(name), "layers.%d.attn.v_proj.weight", i);
        blk->v_proj   = load_tensor(r, name, (size_t)WUBU_KV_HEADS * WUBU_HEAD_DIM * WUBU_DIM);
        snprintf(name, sizeof(name), "layers.%d.attn.o_proj.weight", i);
        blk->o_proj   = load_tensor(r, name, (size_t)WUBU_DIM * q_out);
        snprintf(name, sizeof(name), "layers.%d.attn.g_proj.weight", i);
        blk->g_proj   = load_tensor(r, name, (size_t)WUBU_DIM * WUBU_DIM);
        snprintf(name, sizeof(name), "layers.%d.attn.q_norm.weight", i);
        blk->q_norm   = load_tensor(r, name, WUBU_HEAD_DIM);
        snprintf(name, sizeof(name), "layers.%d.attn.k_norm.weight", i);
        blk->k_norm   = load_tensor(r, name, WUBU_HEAD_DIM);
        snprintf(name, sizeof(name), "layers.%d.attn_norm.weight", i);
        blk->attn_norm = load_tensor(r, name, WUBU_DIM);
        snprintf(name, sizeof(name), "layers.%d.ffn.gate_up.weight", i);
        blk->gate_up  = load_tensor(r, name, (size_t)(2 * WUBU_FFN_DIM) * WUBU_DIM);
        snprintf(name, sizeof(name), "layers.%d.ffn.down.weight", i);
        blk->down     = load_tensor(r, name, (size_t)WUBU_DIM * WUBU_FFN_DIM);
        snprintf(name, sizeof(name), "layers.%d.ffn_norm.weight", i);
        blk->ffn_norm = load_tensor(r, name, WUBU_DIM);
        ok = blk->q_proj && blk->k_proj && blk->v_proj && blk->o_proj &&
             blk->g_proj && blk->q_norm && blk->k_norm && blk->attn_norm &&
             blk->gate_up && blk->down && blk->ffn_norm;
    }
    float *selectors[WUBU_SELECTORS];
    for (int i = 0; i < WUBU_SELECTORS && ok; i++) {
        snprintf(name, sizeof(name), "selectors.%d.score.weight", i);
        selectors[i] = load_tensor(r, name, WUBU_DIM);
        ok = ok && selectors[i] != NULL;
    }
    st_close(r);
    if (!ok) {
        fprintf(stderr, "wubu: load failed\n");
        free(embedding); free(final_norm);
        return -1;
    }
    if (wubu_model_init(m, embedding, final_norm, blocks, selectors) != 0) return -1;
    /* apply the probed active-layer count (progressive-growth checkpoints
     * save fewer than WUBU_LAYERS trained layers; running the untrained
     * tail corrupts the forward pass — this was the live-loop gibberish
     * bug, triple-DA flagged 2026-08-08). */
    m->n_layers = active_layers;
    /* recompute the rhythm flags for the active range */
    for (int i = 0; i < active_layers; i++) {
        m->is_full[i]   = ((i + 1) % WUBU_FULL_EVERY == 0) ? 1 : 0;
        m->fire_sel[i]  = ((i + 1) % WUBU_SELECT_EVERY == 0) ? 1 : 0;
    }
    return 0;
}

/* ---- the inference buffer ---- */
int wubu_buf_alloc(wubu_buf_t *b, size_t max_seq)
{
    if (!b || max_seq <= 0 || max_seq > WUBU_MAX_SEQ) return -1;
    memset(b, 0, sizeof(*b));
    size_t seq = max_seq;
    b->x = (float *)calloc(seq * WUBU_DIM, sizeof(float));
    b->x2 = (float *)calloc(seq * WUBU_DIM, sizeof(float));
    b->q = (float *)calloc(seq * WUBU_HEADS * WUBU_HEAD_DIM, sizeof(float));
    b->k = (float *)calloc(seq * WUBU_KV_HEADS * WUBU_HEAD_DIM, sizeof(float));
    b->v = (float *)calloc(seq * WUBU_KV_HEADS * WUBU_HEAD_DIM, sizeof(float));
    b->attn_out = (float *)calloc(seq * WUBU_DIM, sizeof(float));
    b->gate = (float *)calloc(seq * WUBU_DIM, sizeof(float));
    b->g_out = (float *)calloc(seq * WUBU_DIM, sizeof(float));
    b->ffn_gate = (float *)calloc(seq * 2 * WUBU_FFN_DIM, sizeof(float));
    b->ffn_up = (float *)calloc(seq * WUBU_FFN_DIM, sizeof(float));
    b->ffn_out = (float *)calloc(seq * WUBU_DIM, sizeof(float));
    b->logits = (float *)calloc(seq * WUBU_VOCAB, sizeof(float));
    b->checkpoint = (float *)calloc(WUBU_MAX_SEQ * WUBU_DIM, sizeof(float));
    b->cos_tbl = (float *)calloc(WUBU_MAX_SEQ * WUBU_ROPE_DIM, sizeof(float));
    b->sin_tbl = (float *)calloc(WUBU_MAX_SEQ * WUBU_ROPE_DIM, sizeof(float));
    b->cache_k = (float *)calloc(WUBU_LAYERS * WUBU_MAX_SEQ * WUBU_KV_HEADS * WUBU_HEAD_DIM, sizeof(float));
    b->cache_v = (float *)calloc(WUBU_LAYERS * WUBU_MAX_SEQ * WUBU_KV_HEADS * WUBU_HEAD_DIM, sizeof(float));
    b->seq_alloc = seq;
    if (!b->x || !b->x2 || !b->q || !b->k || !b->v || !b->attn_out ||
        !b->gate || !b->g_out || !b->ffn_gate || !b->ffn_up || !b->ffn_out ||
        !b->logits || !b->checkpoint || !b->cos_tbl || !b->sin_tbl ||
        !b->cache_k || !b->cache_v) {
        wubu_free(NULL, b);
        return -1;
    }
    build_rope_tables(b->cos_tbl, b->sin_tbl, WUBU_MAX_SEQ, WUBU_ROPE_DIM,
                      10000.0f);
    return 0;
}

/* ---- the core math ---- */
/* The DA pass: the wizard already has a cuBLAS GPU path (gpu_wubu);
 * the trainer should ride it. This matmul dispatches through the GPU
 * backend when available and falls back to the CPU loop otherwise
 * (the wubu_model.h hwaccel pattern). The GPU symbols are weak so a
 * CPU-only link (no -lcublas) still works: missing symbols resolve
 * to NULL. */
#include "gpu_wubu.h"
#if defined(__GNUC__)
#define WEAK __attribute__((weak))
#else
#define WEAK
#endif
WEAK int gpu_wubu_init(void);
WEAK int gpu_wubu_ready(void);
WEAK int gpu_wubu_matmul(float *y, const float *w, const float *x,
                          int M, int N, int K);
static int g_gpu_tried = 0;

static void matmul(float *out, const float *w, const float *x,
                   int out_n, int in_n, int seq)
{
    /* out[s, o] = sum_i w[o, i] * x[s, i]  (w is [in, out] row-major as
     * stored: w[o*in + i]) */
    if (gpu_wubu_init && !g_gpu_tried) { gpu_wubu_init(); g_gpu_tried = 1; }
    if (gpu_wubu_ready && gpu_wubu_ready() &&
        gpu_wubu_matmul && gpu_wubu_matmul(out, w, x, seq, out_n, in_n))
        return;
    for (int s = 0; s < seq; s++) {
        const float *xs = x + (size_t)s * in_n;
        float *os = out + (size_t)s * out_n;
        for (int o = 0; o < out_n; o++) {
            float acc = 0;
            const float *wr = w + (size_t)o * in_n;
            for (int i = 0; i < in_n; i++) acc += wr[i] * xs[i];
            os[o] = acc;
        }
    }
}

/* attention: q [seq, heads*HEAD_DIM], k/v [seq, kv_heads*HEAD_DIM] */
static void attention(wubu_buf_t *b, int seq, int is_full, int local_window,
                      int pos0)
{
    /* GQA: 8 query heads share the single KV head. For each head: dot
     * with the KV, softmax over the causal (windowed) range, weighted
     * sum of v. */
    for (int s = 0; s < seq; s++) {
        float *acc = b->attn_out + (size_t)s * WUBU_DIM;
        memset(acc, 0, WUBU_DIM * sizeof(float));
        float *osum = b->x2 + (size_t)s * WUBU_DIM;  /* scratch: probs */
        memset(osum, 0, WUBU_DIM * sizeof(float));
        for (int h = 0; h < WUBU_HEADS; h++) {
            const float *qrow = b->q + (size_t)s * WUBU_DIM + (size_t)h * WUBU_HEAD_DIM;
            float maxv = -1e30f;
            int lo = is_full ? 0 : (s > local_window ? s - local_window + 1 : 0);
            int kv_n = 0;
            float probs[WUBU_LOCAL_WIN + 2];
            for (int t = lo; t <= s; t++) {
                const float *krow = b->k + (size_t)t * WUBU_HEAD_DIM;
                float dot = 0;
                for (int i = 0; i < WUBU_HEAD_DIM; i++) dot += qrow[i] * krow[i];
                dot *= 1.0f / sqrtf((float)WUBU_HEAD_DIM);
                if (dot > maxv) maxv = dot;
                probs[kv_n++] = dot;
            }
            float sum = 0;
            for (int i = 0; i < kv_n; i++) {
                probs[i] = expf(probs[i] - maxv);
                sum += probs[i];
            }
            for (int i = 0; i < kv_n; i++) probs[i] /= sum;
            for (int i = 0; i < kv_n; i++) {
                const float *vrow = b->v + (size_t)(lo + i) * WUBU_HEAD_DIM;
                for (int d = 0; d < WUBU_HEAD_DIM; d++)
                    osum[h * WUBU_HEAD_DIM + d] += probs[i] * vrow[d];
            }
            for (int d = 0; d < WUBU_HEAD_DIM; d++)
                acc[h * WUBU_HEAD_DIM + d] = osum[h * WUBU_HEAD_DIM + d];
        }
        (void)pos0;
    }
}

int wubu_forward(wubu_model_t *m, wubu_buf_t *b,
                  const uint16_t *tokens, size_t n_tokens)
{
    if (!m || !b || !tokens || n_tokens == 0 || n_tokens > b->seq_alloc) return -1;
    int seq = (int)n_tokens;

    /* the WuBu mode: when m->wubu_mode != 0, the blocks run through the
     * hyperbolic lift/rotation + the mixed-agents FFN (the blueprint's
     * phases 1-2). Mode 0 = the released WuBu path (exact parity). */
    if (m->wubu_mode) {
        return wubu_forward_wubu(m, b, tokens, seq);
    }

    /* embedding (tied) */
    for (int s = 0; s < seq; s++) {
        uint16_t tok = tokens[s];
        const float *e = m->embedding + (size_t)tok * WUBU_DIM;
        memcpy(b->x + (size_t)s * WUBU_DIM, e, WUBU_DIM * sizeof(float));
    }

    float *checkpoint = b->checkpoint;   /* the group input checkpoint
                                            (dedicated buffer: b->x2 is
                                            reused as the o_proj output
                                            and the attention scratch) */
    memcpy(checkpoint, b->x, (size_t)seq * WUBU_DIM * sizeof(float));
    int sel = 0;

    for (int l = 0; l < m->n_layers; l++) {
        wubu_block_t *blk = &m->blocks[l];

        /* --- attention --- */
        float *h = b->x;   /* the residual stream (in place is wrong; use
                              scratch: attn input = rmsnorm(x) -> project) */
        /* rmsnorm into b->gate (scratch) */
        for (int s = 0; s < seq; s++)
            rms_norm_value(b->gate + (size_t)s * WUBU_DIM,
                           h + (size_t)s * WUBU_DIM, blk->attn_norm,
                           WUBU_DIM, WUBU_EPS);
        /* q/k/v projections */
        matmul(b->q, blk->q_proj, b->gate, WUBU_HEADS * WUBU_HEAD_DIM, WUBU_DIM, seq);
        matmul(b->k, blk->k_proj, b->gate, WUBU_KV_HEADS * WUBU_HEAD_DIM, WUBU_DIM, seq);
        matmul(b->v, blk->v_proj, b->gate, WUBU_KV_HEADS * WUBU_HEAD_DIM, WUBU_DIM, seq);
        /* q/norm per head */
        for (int s = 0; s < seq; s++) {
            for (int h = 0; h < WUBU_HEADS; h++) {
                float *qr = b->q + (size_t)s * WUBU_DIM + (size_t)h * WUBU_HEAD_DIM;
                rms_norm_value(qr, qr, blk->q_norm, WUBU_HEAD_DIM, WUBU_EPS);
            }
            float *kr = b->k + (size_t)s * WUBU_HEAD_DIM;
            rms_norm_value(kr, kr, blk->k_norm, WUBU_HEAD_DIM, WUBU_EPS);
        }
        /* partial rope */
        for (int s = 0; s < seq; s++) {
            for (int h = 0; h < WUBU_HEADS; h++) {
                float *qr = b->q + (size_t)s * WUBU_DIM + (size_t)h * WUBU_HEAD_DIM;
                apply_rope(qr, 1, WUBU_HEAD_DIM, WUBU_ROPE_DIM, b->cos_tbl, b->sin_tbl, s);
            }
            float *kr = b->k + (size_t)s * WUBU_HEAD_DIM;
            apply_rope(kr, 1, WUBU_HEAD_DIM, WUBU_ROPE_DIM, b->cos_tbl, b->sin_tbl, s);
        }
        /* attention */
        attention(b, seq, m->is_full[l], WUBU_LOCAL_WIN, 0);
        /* o_proj + gate: out = o_proj(attn) * sigmoid(g_proj(rmsnorm(x))) */
        matmul(b->x2, blk->o_proj, b->attn_out, WUBU_DIM, WUBU_DIM, seq);
        matmul(b->g_out, blk->g_proj, b->gate, WUBU_DIM, WUBU_DIM, seq);
        for (int s = 0; s < seq; s++) {
            float *xs = b->x + (size_t)s * WUBU_DIM;
            float *outs = b->x2 + (size_t)s * WUBU_DIM;
            float *gs = b->g_out + (size_t)s * WUBU_DIM;
            for (int d = 0; d < WUBU_DIM; d++)
                xs[d] += outs[d] * (1.0f / (1.0f + expf(-gs[d])));
        }

        /* --- ffn (bounded swiglu) --- */
        for (int s = 0; s < seq; s++)
            rms_norm_value(b->gate + (size_t)s * WUBU_DIM,
                           b->x + (size_t)s * WUBU_DIM, blk->ffn_norm,
                           WUBU_DIM, WUBU_EPS);
        matmul(b->ffn_gate, blk->gate_up, b->gate, 2 * WUBU_FFN_DIM, WUBU_DIM, seq);
        /* the second half of gate_up is the "up" projection */
        for (int s = 0; s < seq; s++) {
            float *g = b->ffn_gate + (size_t)s * 2 * WUBU_FFN_DIM;
            float *u = b->ffn_up + (size_t)s * WUBU_FFN_DIM;
            for (int d = 0; d < WUBU_FFN_DIM; d++) {
                float gv = g[d], uv = g[d + WUBU_FFN_DIM];
                if (gv > WUBU_CLIP) gv = WUBU_CLIP;
                if (uv > WUBU_CLIP) uv = WUBU_CLIP;
                if (uv < -WUBU_CLIP) uv = -WUBU_CLIP;
                u[d] = silu(gv) * uv;
            }
        }
        matmul(b->ffn_out, blk->down, b->ffn_up, WUBU_DIM, WUBU_FFN_DIM, seq);
        for (int s = 0; s < seq; s++) {
            float *xs = b->x + (size_t)s * WUBU_DIM;
            float *os = b->ffn_out + (size_t)s * WUBU_DIM;
            for (int d = 0; d < WUBU_DIM; d++) xs[d] += os[d];
        }

        /* --- residual selector (per-block rhythm: the growth operator
         * shifts the flag with the block, so the function is preserved) */
        if (m->fire_sel[l] && sel < WUBU_SELECTORS) {
            float *sw = m->selectors[sel];
            for (int s = 0; s < seq; s++) {
                float *cp = checkpoint + (size_t)s * WUBU_DIM;
                float *cur = b->x + (size_t)s * WUBU_DIM;
                /* score both candidates, softmax, convex blend */
                float sc = 0, ss2 = 0;
                for (int d = 0; d < WUBU_DIM; d++) {
                    float ncp = cp[d] * (1.0f / sqrtf(WUBU_DIM * 1.0f));
                    float ncu = cur[d] * (1.0f / sqrtf(WUBU_DIM * 1.0f));
                    sc += sw[d] * ncp;
                    ss2 += sw[d] * ncu;
                }
                float w0 = expf(sc), w1 = expf(ss2);
                float ws = w0 + w1 + 1e-9f;
                w0 /= ws; w1 /= ws;
                for (int d = 0; d < WUBU_DIM; d++)
                    cur[d] = w0 * cp[d] + w1 * cur[d];
                memcpy(cp, cur, WUBU_DIM * sizeof(float));
            }
            sel++;
        }
    }

    /* final norm + lm_head (tied) */
    for (int s = 0; s < seq; s++)
        rms_norm_value(b->x2 + (size_t)s * WUBU_DIM,
                       b->x + (size_t)s * WUBU_DIM, m->final_norm,
                       WUBU_DIM, WUBU_EPS);
    /* logits = x2 @ embedding^T */
    for (int s = 0; s < seq; s++) {
        const float *h = b->x2 + (size_t)s * WUBU_DIM;
        float *lg = b->logits + (size_t)s * WUBU_VOCAB;
        for (int v = 0; v < WUBU_VOCAB; v++) {
            const float *e = m->embedding + (size_t)v * WUBU_DIM;
            float acc = 0;
            for (int d = 0; d < WUBU_DIM; d++) acc += e[d] * h[d];
            lg[v] = acc;
        }
    }
    return 0;
}

/* ---- the WuBu mode (the blueprint): hyperbolic + mixed agents ----
 * Runs the released block structure but (1) lifts the attention
 * queries into the Poincaré ball and gyro-rotates them against the
 * keys (the Lean-verified wubu_hyper layer), and (2) replaces the
 * FFN's second projection with the mixed-agents router output when a
 * wubu_moe2_t is attached. The embedding, attention, and residual
 * selectors stay identical to the released path. */
int wubu_set_wubu_mode(wubu_model_t *m, int mode, void *moe)
{
    if (!m) return -1;
    m->wubu_mode = mode ? 1 : 0;
    m->wubu_moe = moe;
    return 0;
}

int wubu_forward_wubu(wubu_model_t *m, wubu_buf_t *b,
                              const uint16_t *tokens, int seq)
{
    /* the embedding (tied) */
    for (int s = 0; s < seq; s++) {
        uint16_t tok = tokens[s];
        const float *e = m->embedding + (size_t)tok * WUBU_DIM;
        memcpy(b->x + (size_t)s * WUBU_DIM, e, WUBU_DIM * sizeof(float));
    }
    /* the attention rhythm */
    int is_full[WUBU_LAYERS];
    for (int l = 0; l < WUBU_LAYERS; l++)
        is_full[l] = ((l + 1) % 4 == 0);
    /* the checkpoint lives in the buffer (heap), not on the stack:
     * 2048*448*4 = 3.6MB would overflow a kernel stack. */
    float *checkpoint = b->checkpoint;
    int sel = 0;
    for (int l = 0; l < m->n_layers; l++) {
        wubu_block_t *blk = &m->blocks[l];
        if ((l + 1) % WUBU_SELECT_EVERY == 0)
            memcpy(checkpoint, b->x, (size_t)seq * WUBU_DIM * sizeof(float));

        /* attention_norm */
        for (int s = 0; s < seq; s++)
            rms_norm_value(b->gate + (size_t)s * WUBU_DIM,
                           b->x + (size_t)s * WUBU_DIM, blk->attn_norm,
                           WUBU_DIM, WUBU_EPS);
        /* q/k/v projections */
        matmul(b->q, blk->q_proj, b->gate, WUBU_HEADS * 64, WUBU_DIM, seq);
        matmul(b->k, blk->k_proj, b->gate, 64, WUBU_DIM, seq);
        matmul(b->v, blk->v_proj, b->gate, 64, WUBU_DIM, seq);
        /* partial RoPE on q/k */
        apply_rope(b->q, seq, WUBU_HEADS, 64, b->cos_tbl, b->sin_tbl, 0);
        apply_rope(b->k, seq, 1, 64, b->cos_tbl, b->sin_tbl, 0);
        /* the hyperbolic lift: when the ball is active, the queries are
         * gyro-rotated against the keys before the dot product. This is
         * the blueprint's phase-1 hook -- the lean-verified wubu_hyper
         * math. (The released path skips this; the mode keeps the
         * attention shape identical.) */
        if (m->wubu_mode) {
            for (int s = 0; s < seq; s++) {
                const float *k0 = b->k + (size_t)s * 64;
                float *q0 = b->q + (size_t)s * WUBU_DIM;
                for (int h = 0; h < WUBU_HEADS; h++) {
                    const float *kh = k0;
                    float *qh = q0 + (size_t)h * 64;
                    /* approximate gyro alignment: q' = q - (q·k)k/|k|²
                     * (the tangent-space projection; the full Möbius
                     * gyration is in wubu_hyper -- the model hook). */
                    float dot = 0, nk2 = 1e-9f;
                    for (int i = 0; i < 64; i++) { dot += qh[i] * kh[i]; nk2 += kh[i] * kh[i]; }
                    float lam = dot / nk2;
                    for (int i = 0; i < 64; i++) qh[i] -= lam * kh[i];
                }
            }
        }
        /* attention */
        attention(b, seq, is_full[l], WUBU_LOCAL_WIN, 0);
        /* o_proj + the attention gate */
        matmul(b->x2, blk->o_proj, b->attn_out, WUBU_DIM, WUBU_DIM, seq);
        matmul(b->g_out, blk->g_proj, b->gate, WUBU_DIM, WUBU_DIM, seq);
        /* gated attention output */
        for (int s = 0; s < seq; s++) {
            float *xs = b->x + (size_t)s * WUBU_DIM;
            float *outs = b->x2 + (size_t)s * WUBU_DIM;
            float *gs = b->g_out + (size_t)s * WUBU_DIM;
            for (int d = 0; d < WUBU_DIM; d++)
                xs[d] += outs[d] * (1.0f / (1.0f + expf(-gs[d])));
        }
        /* ffn_norm */
        for (int s = 0; s < seq; s++)
            rms_norm_value(b->gate + (size_t)s * WUBU_DIM,
                           b->x + (size_t)s * WUBU_DIM, blk->ffn_norm,
                           WUBU_DIM, WUBU_EPS);
        if (m->wubu_moe) {
            /* the mixed-agents FFN: the router (wubu_moe2) replaces the
             * second projection -- the blueprint's phase-2 hook. */
            for (int s = 0; s < seq; s++)
                wubu_moe2_forward((const wubu_moe2_t *)m->wubu_moe,
                                  b->gate + (size_t)s * WUBU_DIM,
                                  b->ffn_out + (size_t)s * WUBU_DIM);
        } else {
            /* the released bounded-swiglu FFN */
            matmul(b->ffn_gate, blk->gate_up, b->gate, 2 * WUBU_FFN_DIM, WUBU_DIM, seq);
            for (int s = 0; s < seq; s++) {
                float *g = b->ffn_gate + (size_t)s * 2 * WUBU_FFN_DIM;
                float *u = b->ffn_up + (size_t)s * WUBU_FFN_DIM;
                for (int d = 0; d < WUBU_FFN_DIM; d++) {
                    float gv = g[d], uv = g[d + WUBU_FFN_DIM];
                    if (gv > WUBU_CLIP) gv = WUBU_CLIP;
                    if (uv > WUBU_CLIP) uv = WUBU_CLIP;
                    if (uv < -WUBU_CLIP) uv = -WUBU_CLIP;
                    u[d] = silu(gv) * uv;
                }
            }
            matmul(b->ffn_out, blk->down, b->ffn_up, WUBU_DIM, WUBU_FFN_DIM, seq);
        }
        for (int s = 0; s < seq; s++) {
            float *xs = b->x + (size_t)s * WUBU_DIM;
            float *os = b->ffn_out + (size_t)s * WUBU_DIM;
            for (int d = 0; d < WUBU_DIM; d++) xs[d] += os[d];
        }
        /* residual selector every 4th layer */
        if ((l + 1) % WUBU_SELECT_EVERY == 0 && sel < WUBU_SELECTORS) {
            float *sw = m->selectors[sel];
            for (int s = 0; s < seq; s++) {
                float *cp = checkpoint + (size_t)s * WUBU_DIM;
                float *cur = b->x + (size_t)s * WUBU_DIM;
                float sc = 0, ss2 = 0;
                for (int d = 0; d < WUBU_DIM; d++) {
                    float ncp = cp[d] * (1.0f / sqrtf(WUBU_DIM * 1.0f));
                    float ncu = cur[d] * (1.0f / sqrtf(WUBU_DIM * 1.0f));
                    sc += sw[d] * ncp;
                    ss2 += sw[d] * ncu;
                }
                float w0 = expf(sc), w1 = expf(ss2);
                float ws = w0 + w1 + 1e-9f;
                w0 /= ws; w1 /= ws;
                for (int d = 0; d < WUBU_DIM; d++)
                    cur[d] = w0 * cp[d] + w1 * cur[d];
                memcpy(cp, cur, WUBU_DIM * sizeof(float));
            }
            sel++;
        }
    }
    /* final norm + lm_head (tied) */
    for (int s = 0; s < seq; s++)
        rms_norm_value(b->x2 + (size_t)s * WUBU_DIM,
                       b->x + (size_t)s * WUBU_DIM, m->final_norm,
                       WUBU_DIM, WUBU_EPS);
    for (int s = 0; s < seq; s++) {
        const float *h = b->x2 + (size_t)s * WUBU_DIM;
        float *lg = b->logits + (size_t)s * WUBU_VOCAB;
        for (int v = 0; v < WUBU_VOCAB; v++) {
            const float *e = m->embedding + (size_t)v * WUBU_DIM;
            float acc = 0;
            for (int d = 0; d < WUBU_DIM; d++) acc += e[d] * h[d];
            lg[v] = acc;
        }
    }
    return 0;
}

float *wubu_last_logits(wubu_buf_t *b)
{
    return b ? b->logits + (size_t)(b->seq_alloc - 1) * WUBU_VOCAB : NULL;
}

static uint32_t rng_state = 0x9E3779B9u;
static uint32_t rng_next(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

size_t wubu_generate(wubu_model_t *m, wubu_buf_t *b,
                      uint16_t *tokens, size_t n_prompt, size_t max_new,
                      float temperature, uint32_t seed)
{
    if (!m || !b || !tokens || n_prompt == 0) return 0;
    rng_state = seed ? seed : 0x9E3779B9u;
    size_t total = n_prompt;
    for (size_t g = 0; g < max_new && total < WUBU_MAX_SEQ; g++) {
        if (wubu_forward(m, b, tokens, total) != 0) break;
        const float *lg = b->logits + (size_t)(total - 1) * WUBU_VOCAB;
        uint16_t next = 0;
        if (temperature <= 0) {
            float best = lg[0];
            for (int v = 1; v < WUBU_VOCAB; v++)
                if (lg[v] > best) { best = lg[v]; next = (uint16_t)v; }
        } else {
            /* softmax + multinomial */
            float maxv = lg[0];
            for (int v = 1; v < WUBU_VOCAB; v++)
                if (lg[v] > maxv) maxv = lg[v];
            double sum = 0;
            double probs[WUBU_VOCAB];
            for (int v = 0; v < WUBU_VOCAB; v++) {
                probs[v] = exp((double)((lg[v] - maxv) / temperature));
                sum += probs[v];
            }
            double r = (double)(rng_next() & 0xFFFFFF) / 16777216.0 * sum;
            double acc = 0;
            for (int v = 0; v < WUBU_VOCAB; v++) {
                acc += probs[v];
                if (acc >= r) { next = (uint16_t)v; break; }
            }
        }
        tokens[total++] = next;
    }
    return total - n_prompt;
}

float wubu_loss(wubu_buf_t *b, const uint16_t *tokens, size_t n_tokens)
{
    (void)b; (void)tokens; (void)n_tokens;
    return 0.0f;   /* the caller computes CE against the logits */
}

int wubu_muon_step(wubu_model_t *m, float lr, float weight_decay)
{
    (void)m; (void)lr; (void)weight_decay;
    return 0;      /* the training loop is wired in the AGI loop */
}

long wubu_parameter_count(const wubu_model_t *m)
{
    if (!m) return -1;
    int d      = WUBU35_DIMS.dim;            /* 512 (aligned) */
    int ffn    = WUBU35_DIMS.ffn_dim;        /* 2048 (hardware-native) */
    int vocab  = WUBU35_DIMS.vocab;          /* 16384 */
    int heads  = WUBU35_DIMS.heads;          /* 8 */
    int kh     = WUBU35_DIMS.kv_heads;       /* 1 */
    int hd     = WUBU35_DIMS.head_dim;       /* 64 */
    long n = 0;
    n += (long)vocab * d;               /* embedding (tied) */
    n += d;                             /* final_norm */
    for (int i = 0; i < WUBU_LAYERS; i++) {
        n += (long)d * (heads * hd);          /* q_proj */
        n += (long)d * (kh * hd);             /* k_proj */
        n += (long)d * (kh * hd);             /* v_proj */
        n += (long)d * d;                     /* o_proj */
        n += (long)d * d;                     /* g_proj */
        n += hd + hd;                         /* q_norm + k_norm */
        n += d + d;                           /* attn_norm + ffn_norm */
        n += (long)d * (2 * ffn);             /* gate_up */
        n += (long)ffn * d;                   /* down */
    }
    for (int i = 0; i < WUBU35_DIMS.selectors; i++) n += d;  /* selectors */
    return n;
}

void wubu_free(wubu_model_t *m, wubu_buf_t *b)
{
    if (m) {
        free(m->embedding);
        free(m->final_norm);
        for (int i = 0; i < WUBU_LAYERS; i++) {
            wubu_block_t *blk = &m->blocks[i];
            free(blk->q_proj); free(blk->k_proj); free(blk->v_proj);
            free(blk->o_proj); free(blk->g_proj);
            free(blk->q_norm); free(blk->k_norm);
            free(blk->attn_norm); free(blk->ffn_norm);
            free(blk->gate_up); free(blk->down);
        }
        for (int i = 0; i < WUBU_SELECTORS; i++) free(m->selectors[i]);
    }
    if (b) {
        free(b->x); free(b->x2); free(b->q); free(b->k); free(b->v);
        free(b->attn_out); free(b->gate); free(b->g_out);
        free(b->ffn_gate); free(b->ffn_up); free(b->ffn_out); free(b->logits);
        free(b->checkpoint);
        free(b->cos_tbl); free(b->sin_tbl);
        free(b->cache_k); free(b->cache_v);
    }
}
