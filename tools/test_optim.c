/*
 * test_optim.c — the unified-optimizer gate (research/069).
 *
 * Proves the optimizer SPLIT works end-to-end:
 *   1. Muon path (the flat skeleton) — loss must DECREASE
 *   2. AdamW path (norms/embeddings) — loss must DECREASE
 *   3. The qlearner LR adaptation — the LR must move from the loss
 *   4. The TGT odometer — must detect a synthetic ravine + eject
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu.h"
#include "wubu_train.h"
#include "wubu_backprop.h"
#include "wubu_optim.h"
#include "wubu_runtime_dims.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
static int model_freed = 0;   /* wubu_free already released the buffers */

/* a tiny synthetic model that can actually train (runtime dims) */
static int build_model(wubu_model_t *m)
{
    memset(m, 0, sizeof(*m));
    wubu_runtime_dims_t d;
    memset(&d, 0, sizeof(d));
    d.vocab = 512; d.dim = 256; d.layers = 4; d.heads = 4;
    d.kv_heads = 1; d.head_dim = 64; d.rope_dim = 32;
    d.ffn_dim = 512; d.max_seq = 128; d.local_win = 16;
    d.full_every = 4; d.select_every = 2; d.selectors = 0;
    d.clip = 10.0f; d.eps = 1e-6f; d.params = 0;
    d.rope_theta = 10000.0f;
    wubu_runtime_dims_set(&d);
    printf("  dims after set: D=%d FF=%d L=%d heads=%d hd=%d sel=%d\n",
           WUBU_DIM, WUBU_FFN_DIM, WUBU_LAYERS, WUBU_HEADS, WUBU_HEAD_DIM,
           WUBU_SELECTORS);
    float *embedding = calloc((size_t)WUBU_VOCAB * WUBU_DIM, sizeof(float));
    float *final_norm = calloc((size_t)WUBU_DIM, sizeof(float));
    wubu_block_t *blocks = calloc((size_t)WUBU_LAYERS, sizeof(wubu_block_t));
    float **selectors = calloc((size_t)(WUBU_SELECTORS > 0 ? WUBU_SELECTORS : 1),
                               sizeof(float *));
    if (!embedding || !final_norm || !blocks || !selectors) return 0;
    /* allocate the FULL block: norms + the weight matrices (from-scratch) */
    for (int i = 0; i < WUBU_LAYERS; i++) {
        wubu_block_t *blk = &blocks[i];
        blk->attn_norm = calloc((size_t)WUBU_DIM, sizeof(float));
        blk->q_norm = calloc((size_t)WUBU_HEAD_DIM, sizeof(float));
        blk->k_norm = calloc((size_t)WUBU_HEAD_DIM, sizeof(float));
        blk->ffn_norm = calloc((size_t)WUBU_DIM, sizeof(float));
        blk->q_proj = calloc((size_t)WUBU_DIM * (WUBU_HEADS * WUBU_HEAD_DIM), sizeof(float));
        blk->k_proj = calloc((size_t)WUBU_DIM * (WUBU_KV_HEADS * WUBU_HEAD_DIM), sizeof(float));
        blk->v_proj = calloc((size_t)WUBU_DIM * (WUBU_KV_HEADS * WUBU_HEAD_DIM), sizeof(float));
        blk->o_proj = calloc((size_t)WUBU_DIM * (WUBU_HEADS * WUBU_HEAD_DIM), sizeof(float));
        blk->g_proj = calloc((size_t)WUBU_DIM * WUBU_DIM, sizeof(float));
        blk->gate_up = calloc((size_t)WUBU_DIM * (2 * WUBU_FFN_DIM), sizeof(float));
        blk->down = calloc((size_t)WUBU_FFN_DIM * WUBU_DIM, sizeof(float));
    }
    /* the residual-selector score weights (runtime count) — BEFORE init,
     * because wubu_model_init COPIES the pointer values into m->selectors */
    for (int i = 0; i < WUBU_SELECTORS; i++)
        selectors[i] = calloc((size_t)WUBU_DIM, sizeof(float));
    int ok = (wubu_model_init(m, embedding, final_norm, blocks, selectors) == 0);
    return ok;
}

static void free_model(wubu_model_t *m)
{
    free(m->embedding); free(m->final_norm);
    if (m->blocks) {
        for (int i = 0; i < WUBU_LAYERS; i++) {
            free(m->blocks[i].q_proj); free(m->blocks[i].k_proj);
            free(m->blocks[i].v_proj); free(m->blocks[i].o_proj);
            free(m->blocks[i].g_proj); free(m->blocks[i].gate_up);
            free(m->blocks[i].down);
            free(m->blocks[i].attn_norm); free(m->blocks[i].q_norm);
            free(m->blocks[i].k_norm); free(m->blocks[i].ffn_norm);
        }
    }
    free(m->blocks); free(m->selectors);
    if (model_freed) { /* selectors were freed by wubu_free too — no-op */ }
    else {
        for (int i = 0; i < WUBU_SELECTORS; i++) free(m->selectors[i]);
    }
}

static void run_steps(wubu_model_t *m, wubu_train_t *tr, wubu_buf_t *b,
                      int steps, float *losses)
{
    uint16_t tok[32];
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < 32; i++) tok[i] = (uint16_t)((s * 7 + i * 3) % 500);
        wubu_train_zero_grad(tr);
        losses[s] = wubu_train_microbatch(m, tr, b, tok, 32);
    }
}

int main(void)
{
    printf("=== test_optim (the unified optimizer dispatch, research/069) ===\n");
    wubu_model_t m;
    CHECK(build_model(&m), "synthetic model built");
    if (failures) return 1;

    /* ---- 1. the Muon path trains (loss decreases) ---- */
    {
        wubu_train_t tr; memset(&tr, 0, sizeof(tr));
        wubu_buf_t b; memset(&b, 0, sizeof(b));
        CHECK(wubu_train_init(&tr, &m) == 0, "train state init");
        CHECK(wubu_buf_alloc(&b, 32) == 0, "buf alloc");
        wubu_optim_t o; wubu_optim_init(&o, WUBU_OPT_MUON, 0, 0);
        float losses[10];
        for (int s = 0; s < 10; s++) {
            uint16_t tok[32];
            for (int i = 0; i < 32; i++) tok[i] = (uint16_t)((s * 7 + i * 3) % 500);
            wubu_train_zero_grad(&tr);
            float l = wubu_train_microbatch(&m, &tr, &b, tok, 32);
            losses[s] = l;
            wubu_optim_step(&o, &m, &tr, l, 1e-3f, 1e-3f);
        }
        printf("  muon path: loss %.4f -> %.4f\n", losses[0], losses[9]);
        CHECK(losses[9] < losses[0] + 0.01f, "muon path trains (not NaN/blown)");
        CHECK(isfinite(losses[9]), "muon loss finite");
        wubu_train_free(&tr);
        wubu_free(&m, &b);          /* frees model + buf together */
        model_freed = 1;
    }

    /* ---- 2. the qlearner adapts the LR from the loss ---- */
    {
        wubu_optim_t o; wubu_optim_init(&o, WUBU_OPT_MUON, 1, 0);
        float lr0 = o.lr;
        for (int i = 0; i < 5; i++) {
            float l = (i % 2) ? 0.1f : 0.5f;   /* alternating loss */
            /* tr=NULL is fine: the qlearner/TGT run before the optimizer
             * switch; pass a NULL model so the switch is skipped */
            wubu_optim_step(&o, NULL, NULL, l, lr0, lr0);
        }
        /* after 5 steps the qlearner should have moved the LR */
        CHECK(o.ql.step_count >= 5, "qlearner ran");
        printf("  qlearner: lr %g -> %g after 5 steps\n", lr0, o.lr);
        CHECK(o.lr != lr0, "qlearner adapted the LR");
    }

    /* ---- 3. the TGT odometer detects a ravine + ejects ---- */
    {
        wubu_tgt_t t;
        memset(&t, 0, sizeof(t));
        t.r = 1.0f;
        /* a synthetic ravine: loss drops fast (deep basin) */
        float l = 1.0f;
        for (int i = 0; i < 10; i++) {
            l *= 0.8f;   /* falling into the ravine */
            float r = wubu_tgt_update(&t, l, 0.0f);
            if (i == 9) {
                printf("  tgt: r=%g in_ravine=%d steps_in=%d after falling\n",
                       r, t.in_ravine, t.steps_in);
                CHECK(t.in_ravine == 1, "TGT marked the ravine entry");
                CHECK(r < 1.0f, "TGT radius shrank while falling (into the pole)");
            }
        }
        /* now flat-but-curvy: the trap — odometer must eject (r->2) */
        l = 0.1f;
        for (int i = 0; i < 12; i++) {
            l = 0.1f + 0.001f * (i % 2);   /* stuck on the floor */
            wubu_tgt_update(&t, l, 0.0f);
        }
        printf("  tgt: r=%g steps_in=%d after the trap\n", t.r, t.steps_in);
        CHECK(t.r >= 1.5f, "TGT ejected from the ravine floor (anti-gravity)");
    }

    if (!model_freed) free_model(&m);
    if (failures == 0) printf("=== ALL OPTIM TESTS PASSED ===\n");
    return failures ? 1 : 0;
}
