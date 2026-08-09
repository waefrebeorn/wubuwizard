/*
 * wubu_train.h -- the WuBu training core (the AGI brain-cluster loop).
 *
 * The wizard is an INFERENCE engine today; this module makes it a
 * TRAINING engine too. WuBu-35M is the mustard seed: the training
 * loop grows it -- more tokens from the research repos, more
 * parameters, more knowledge -- all designed and trained in-house.
 *
 * The reference recipe (reproduced faithfully):
 *   - Muon optimizer (the reference used it for the final 4B tokens:
 *     peak lr 1e-4, weight decay 0.1, batch 48, seq 2048)
 *   - cross-entropy on the next token
 *   - the hybrid local/global attention forward (wubu)
 *   - weight-decay applied via the Muon Newton-Schulz style update
 *
 * Training is memory-bound: we accumulate gradients over micro-batches
 * and update with Muon. Pure C11, no third-party deps.
 */
#ifndef WUBU_TRAIN_H
#define WUBU_TRAIN_H

#include "wubu.h"
/* NOTE: wubu.h defines wubu_model_t (the direct-weights training model).
 * wubu_model.h is the gguf-loader's OPAQUE model — do NOT include both. */

/* The training config (the reference recipe). */
typedef struct {
    float  lr;             /* 1e-4 peak (both groups when the splits
                              below are 0) */
    float  muon_lr;        /* the Muon group LR (2e-2); 0 -> lr */
    float  adam_lr;        /* the AdamW group LR (2e-3); 0 -> lr */
    float  weight_decay;   /* 0.1 */
    float  grad_clip;      /* global-norm clip; <= 0 -> no clip */
    uint32_t batch_size;   /* 48 (reference) -- we micro-batch */
    uint32_t seq_len;      /* 2048 */
    uint32_t warmup_steps;
    uint32_t max_steps;
    float  muon_momentum;  /* 0.95 */
    int    adam_for_embed; /* the embedding + norms use AdamW, the
                              matrices use Muon (reference split) */
} wubu_train_cfg_t;

/* The 1-D parameter slots trained with AdamW (norms + selectors):
 *   [4*l + 0] = attn_norm, [4*l + 1] = ffn_norm,
 *   [4*l + 2] = q_norm,    [4*l + 3] = k_norm      (l in 0..L-1)
 *   [4*L]     = final_norm
 *   [4*L+1+i] = selectors[i]                        (i in 0..S-1)
 */
#define WUBU_NORM_SLOTS (4 * WUBU_LAYERS + 1 + WUBU_SELECTORS)
#define WUBU_MAX_LAYERS 64    /* struct pointer-array cap (runtime L <= 64) */
#define WUBU_MAX_NORM_SLOTS (4 * WUBU_MAX_LAYERS + 1 + 8)

/* Gradient accumulators: one float per weight, only for the Muon-updated
 * matrices (the big ones). The norms + embeddings + selectors use
 * AdamW states. */
typedef struct wubu_train {
    /* per-block matrix gradients */
    float *q_proj_g[WUBU_MAX_LAYERS];  /* [448,448] */
    float *k_proj_g[WUBU_MAX_LAYERS];  /* [448,64] */
    float *v_proj_g[WUBU_MAX_LAYERS];
    float *o_proj_g[WUBU_MAX_LAYERS];
    float *g_proj_g[WUBU_MAX_LAYERS];
    float *gate_up_g[WUBU_MAX_LAYERS]; /* [448,2456] */
    float *down_g[WUBU_MAX_LAYERS];    /* [1228,448] */
    /* AdamW states for the embedding */
    float *emb_g;   /* [16384,448] the gradient accumulator */
    float *emb_m;   /* [16384,448] the AdamW first moment */
    float *emb_v;   /* [16384,448] the AdamW second moment */
    /* the 1-D params (norms + selectors) -> AdamW: gradient + states */
    float *norm_g[WUBU_MAX_NORM_SLOTS];
    float *norm_m[WUBU_MAX_NORM_SLOTS];
    float *norm_v[WUBU_MAX_NORM_SLOTS];
    /* Muon states: the momentum per matrix (Newton-Schulz iteration) */
    float *q_proj_m[WUBU_MAX_LAYERS];
    float *k_proj_m[WUBU_MAX_LAYERS];
    float *v_proj_m[WUBU_MAX_LAYERS];
    float *o_proj_m[WUBU_MAX_LAYERS];
    float *g_proj_m[WUBU_MAX_LAYERS];
    float *gate_up_m[WUBU_MAX_LAYERS];
    float *down_m[WUBU_MAX_LAYERS];
    /* the REAL backprop recorder (owned by the trainer; allocated on
     * first use, grown as the sequence grows) */
    struct wubu_bp_t *bp_rec;
    /* telemetry */
    uint32_t micro_steps;
    double   grad_norm_sum;
    double   loss_sum;
} wubu_train_t;

/* T1: initialize the training state (allocates the gradient buffers). */
int wubu_train_init(wubu_train_t *tr, const wubu_model_t *m);

/* T2: zero the accumulated gradients. */
int wubu_train_zero_grad(wubu_train_t *tr);

/* T3: accumulate the gradient of one micro-batch. Runs the recording
 * forward (wubu_bp_forward) + the REAL analytic backward
 * (wubu_bp_backward) -- chain rule through EVERY path (attention
 * q/k/v/o/g, qk-norm, rope, softmax, gated residual, SwiGLU, the
 * residual selectors, final norm, tied head). Every parameter gets
 * its own gradient. Returns the loss. */
float wubu_train_microbatch(wubu_model_t *m, wubu_train_t *tr,
                             wubu_buf_t *b, const uint16_t *tokens,
                             size_t n_tokens);

/* T4: the Muon+AdamW optimizer step. */
int wubu_train_step(wubu_model_t *m, wubu_train_t *tr,
                     const wubu_train_cfg_t *cfg, uint32_t step);

/* T5: the full training loop -- one step = batch of micro-batches. */
float wubu_train_step_loop(wubu_model_t *m, wubu_train_t *tr,
                            wubu_buf_t *b, const uint16_t *tokens,
                            size_t n_tokens, const wubu_train_cfg_t *cfg,
                            uint32_t step);

/* T6: the learning-rate schedule (warmup + cosine decay). */
float wubu_train_lr(const wubu_train_cfg_t *cfg, uint32_t step);

/* T7: free the training state. */
void wubu_train_free(wubu_train_t *tr);

#endif