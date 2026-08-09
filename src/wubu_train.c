/*
 * wubu_train.c -- the WuBu training core (the AGI brain-cluster loop).
 *
 * The wizard becomes a TRAINING engine. The mustard seed (WuBu-35M)
 * grows here: the REAL backprop (wubu_backprop) + the REAL Muon
 * (Newton-Schulz 5) + AdamW for the 1-D params, next-token
 * cross-entropy, the confirmed reference recipe. The gradient of
 * EVERY parameter is computed analytically through the full chain:
 * attention q/k/v/o/g, qk-norm, rope, softmax, the gated residual,
 * the bounded SwiGLU, the residual selectors, the final norm and the
 * tied head. This module owns the trainer state + the micro-batch
 * loop; the deep math lives in wubu_backprop.c.
 */
#include "wubu_train.h"
#include "wubu_backprop.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float *calloc_f(size_t n)
{
    return (float *)calloc(n ? n : 1, sizeof(float));
}

static void free_mat(float *p) { free(p); }

int wubu_train_init(wubu_train_t *tr, const wubu_model_t *m)
{
    if (!tr || !m) return -1;
    memset(tr, 0, sizeof(*tr));
    for (int i = 0; i < WUBU_LAYERS; i++) {
        /* runtime geometry (the revolver): buffer sizes follow the
         * model's ACTUAL dims, not the old compile-time 448 (the
         * hardcoded 448/64/1228/2456/16384 were a pre-existing bug:
         * any model with different dims crashed in backprop) */
        const size_t hd = WUBU_HEADS * WUBU_HEAD_DIM;        /* q/o/g */
        const size_t khd = WUBU_KV_HEADS * WUBU_HEAD_DIM;    /* k/v */
        const size_t d = WUBU_DIM, f = WUBU_FFN_DIM;
        tr->q_proj_g[i] = calloc_f(d * hd);
        tr->k_proj_g[i] = calloc_f(d * khd);
        tr->v_proj_g[i] = calloc_f(d * khd);
        tr->o_proj_g[i] = calloc_f(d * hd);
        tr->g_proj_g[i] = calloc_f(d * hd);
        tr->gate_up_g[i] = calloc_f(d * (2 * f));
        tr->down_g[i] = calloc_f(f * d);
        tr->q_proj_m[i] = calloc_f(d * hd);
        tr->k_proj_m[i] = calloc_f(d * khd);
        tr->v_proj_m[i] = calloc_f(d * khd);
        tr->o_proj_m[i] = calloc_f(d * hd);
        tr->g_proj_m[i] = calloc_f(d * hd);
        tr->gate_up_m[i] = calloc_f(d * (2 * f));
        tr->down_m[i] = calloc_f(f * d);
        if (!tr->q_proj_g[i] || !tr->k_proj_g[i] || !tr->v_proj_g[i] ||
            !tr->o_proj_g[i] || !tr->g_proj_g[i] || !tr->gate_up_g[i] ||
            !tr->down_g[i] || !tr->q_proj_m[i] || !tr->k_proj_m[i] ||
            !tr->v_proj_m[i] || !tr->o_proj_m[i] || !tr->g_proj_m[i] ||
            !tr->gate_up_m[i] || !tr->down_m[i]) {
            wubu_train_free(tr);
            return -1;
        }
    }
    tr->emb_g = calloc_f(WUBU_VOCAB * WUBU_DIM);
    tr->emb_m = calloc_f(WUBU_VOCAB * WUBU_DIM);
    tr->emb_v = calloc_f(WUBU_VOCAB * WUBU_DIM);
    if (!tr->emb_g || !tr->emb_m || !tr->emb_v) { wubu_train_free(tr); return -1; }
    /* the 1-D AdamW slots: the per-layer norms, the final norm, the
     * selectors (sizes per the WUBU_NORM_SLOTS layout).
     * q_norm/k_norm are PER-HEAD norms -> WUBU_HEAD_DIM floats (NOT
     * rope_dim — that's the partial-rotation width, a different
     * concept; a pre-existing bug under-allocated them and corrupted
     * the heap on any model where rope_dim != head_dim). */
    for (int i = 0; i < WUBU_NORM_SLOTS; i++) {
        int sz = (i % 4 == 2 || i % 4 == 3) && (i < 4 * WUBU_LAYERS)
                 ? WUBU_HEAD_DIM : WUBU_DIM;
        tr->norm_g[i] = calloc_f((size_t)sz);
        tr->norm_m[i] = calloc_f((size_t)sz);
        tr->norm_v[i] = calloc_f((size_t)sz);
        if (!tr->norm_g[i] || !tr->norm_m[i] || !tr->norm_v[i]) {
            wubu_train_free(tr);
            return -1;
        }
    }
    tr->bp_rec = NULL;   /* allocated lazily on first micro-batch */
    return 0;
}

void wubu_train_free(wubu_train_t *tr)
{
    if (!tr) return;
    for (int i = 0; i < WUBU_LAYERS; i++) {
        free_mat(tr->q_proj_g[i]); free_mat(tr->k_proj_g[i]);
        free_mat(tr->v_proj_g[i]); free_mat(tr->o_proj_g[i]);
        free_mat(tr->g_proj_g[i]); free_mat(tr->gate_up_g[i]);
        free_mat(tr->down_g[i]);
        free_mat(tr->q_proj_m[i]); free_mat(tr->k_proj_m[i]);
        free_mat(tr->v_proj_m[i]); free_mat(tr->o_proj_m[i]);
        free_mat(tr->g_proj_m[i]); free_mat(tr->gate_up_m[i]);
        free_mat(tr->down_m[i]);
    }
    for (int i = 0; i < WUBU_NORM_SLOTS; i++) {
        free_mat(tr->norm_g[i]); free_mat(tr->norm_m[i]);
        free_mat(tr->norm_v[i]);
    }
    free_mat(tr->emb_g); free_mat(tr->emb_m); free_mat(tr->emb_v);
    if (tr->bp_rec) { wubu_bp_free(tr->bp_rec); free(tr->bp_rec); }
    memset(tr, 0, sizeof(*tr));
}

int wubu_train_zero_grad(wubu_train_t *tr)
{
    if (!tr) return -1;
    for (int i = 0; i < WUBU_LAYERS; i++) {
        memset(tr->q_proj_g[i], 0, WUBU_DIM * (WUBU_HEADS * WUBU_HEAD_DIM) * sizeof(float));
        memset(tr->k_proj_g[i], 0, WUBU_DIM * (WUBU_KV_HEADS * WUBU_HEAD_DIM) * sizeof(float));
        memset(tr->v_proj_g[i], 0, WUBU_DIM * (WUBU_KV_HEADS * WUBU_HEAD_DIM) * sizeof(float));
        memset(tr->o_proj_g[i], 0, WUBU_DIM * (WUBU_HEADS * WUBU_HEAD_DIM) * sizeof(float));
        memset(tr->g_proj_g[i], 0, WUBU_DIM * (WUBU_HEADS * WUBU_HEAD_DIM) * sizeof(float));
        memset(tr->gate_up_g[i], 0, WUBU_DIM * (2 * WUBU_FFN_DIM) * sizeof(float));
        memset(tr->down_g[i], 0, WUBU_FFN_DIM * WUBU_DIM * sizeof(float));
    }
    memset(tr->emb_g, 0, WUBU_VOCAB * WUBU_DIM * sizeof(float));
    for (int i = 0; i < WUBU_NORM_SLOTS; i++) {
        int sz = (i % 4 == 2 || i % 4 == 3) && (i < 4 * WUBU_LAYERS)
                 ? WUBU_HEAD_DIM : WUBU_DIM;
        memset(tr->norm_g[i], 0, (size_t)sz * sizeof(float));
    }
    tr->micro_steps = 0;
    tr->grad_norm_sum = 0;
    tr->loss_sum = 0;
    return 0;
}

/* ensure the recorder can hold n_tokens (grow on demand) */
static int ensure_bp(wubu_train_t *tr, int n_tokens)
{
    if (tr->bp_rec && tr->bp_rec->cap_seq >= n_tokens) return 0;
    if (tr->bp_rec) { wubu_bp_free(tr->bp_rec); free(tr->bp_rec); tr->bp_rec = NULL; }
    tr->bp_rec = (wubu_bp_t *)calloc(1, sizeof(wubu_bp_t));
    if (!tr->bp_rec) return -1;
    if (wubu_bp_alloc(tr->bp_rec, n_tokens) != 0) {
        free(tr->bp_rec);
        tr->bp_rec = NULL;
        return -1;
    }
    return 0;
}

float wubu_train_microbatch(wubu_model_t *m, wubu_train_t *tr,
                             wubu_buf_t *b, const uint16_t *tokens,
                             size_t n_tokens)
{
    if (!m || !tr || !b || !tokens || n_tokens < 2) return 0;
    if (ensure_bp(tr, (int)n_tokens) != 0) return 0;
    float loss = wubu_bp_forward(m, b, tr->bp_rec, tokens, (int)n_tokens);
    wubu_bp_backward(m, b, tr->bp_rec, tr, tokens, (int)n_tokens);
    return loss;
}

float wubu_train_lr(const wubu_train_cfg_t *cfg, uint32_t step)
{
    if (!cfg) return 1e-4f;
    float lr;
    if (step < cfg->warmup_steps) {
        lr = cfg->lr * ((float)step / (float)(cfg->warmup_steps ? cfg->warmup_steps : 1));
    } else {
        float t = (float)(step - cfg->warmup_steps) /
                  (float)(cfg->max_steps - cfg->warmup_steps + 1);
        lr = cfg->lr * 0.5f * (1.0f + cosf(acosf(-1.0f) * t));
    }
    return lr;
}

int wubu_train_step(wubu_model_t *m, wubu_train_t *tr,
                     const wubu_train_cfg_t *cfg, uint32_t step)
{
    if (!m || !tr || !cfg) return -1;
    return wubu_bp_muon_step(m, tr, cfg, step);
}

float wubu_train_step_loop(wubu_model_t *m, wubu_train_t *tr,
                            wubu_buf_t *b, const uint16_t *tokens,
                            size_t n_tokens, const wubu_train_cfg_t *cfg,
                            uint32_t step)
{
    if (!m || !tr || !b || !tokens || !cfg || n_tokens < 2) return 0;
    wubu_train_zero_grad(tr);
    /* micro-batch: the whole sequence is one micro-batch in the seed
     * loop (the reference used 48 sequences of 2048; we chunk). */
    float loss = wubu_train_microbatch(m, tr, b, tokens, n_tokens);
    wubu_train_step(m, tr, cfg, step);
    return loss;
}
