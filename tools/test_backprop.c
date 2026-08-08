/*
 * test_backprop.c -- the REAL backward pass + REAL Muon, verified the
 * DA way: analytic gradients vs FINITE DIFFERENCES on a tiny random
 * model. Tests != correct -- so we perturb a real weight, re-run the
 * forward, and compare (L(w+h) - L(w-h)) / 2h against the analytic
 * dL/dw for EVERY parameter type (q/k/v/o/g/gate_up/down projections,
 * the qk/attn/ffn/final norms, the selectors, the embedding).
 *
 * Also proves the recording forward matches the RELEASED forward
 * (wubu_forward) loss for the same tokens -- the trainer and the
 * inference engine must agree on the model they see.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu.h"
#include "wubu_train.h"
#include "wubu_backprop.h"

/* Weak GPU dispatch (same pattern as wubu_backprop.c): on boxes without
 * the CUDA backend these symbols are NULL and the FD perturb skips the
 * weight-cache dirty mark; with the backend they are strong and the
 * check fires. gpu_wubu.h (strong) would make the address checks dead. */
#ifdef __GNUC__
#define BP_WEAK __attribute__((weak))
#else
#define BP_WEAK
#endif
BP_WEAK void gpu_wubu_mark_weights_dirty(void);

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

/* deterministic xorshift */
static uint32_t rng_state = 0x12345678u;
static uint32_t rng_next(void)
{
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    return x;
}
static float frand(void)
{
    return ((float)(rng_next() & 0xFFFF) / 32767.5f) - 1.0f;
}

static void fill_rand(float *p, size_t n, float scale)
{
    for (size_t i = 0; i < n; i++) p[i] = frand() * scale;
}

/* build a small random model (no safetensors needed) */
static int make_model(wubu_model_t *m)
{
    float *emb = (float *)malloc(16384 * 448 * sizeof(float));
    float *final_norm = (float *)malloc(448 * sizeof(float));
    if (!emb || !final_norm) return -1;
    fill_rand(emb, 16384 * 448, 0.02f);
    for (int i = 0; i < 448; i++) final_norm[i] = 1.0f + 0.01f * frand();

    wubu_block_t blocks[WUBU_LAYERS];
    memset(blocks, 0, sizeof(blocks));
    for (int l = 0; l < WUBU_LAYERS; l++) {
        wubu_block_t *blk = &blocks[l];
        blk->q_proj   = (float *)malloc(448 * 448 * sizeof(float));
        blk->k_proj   = (float *)malloc(448 * 64 * sizeof(float));
        blk->v_proj   = (float *)malloc(448 * 64 * sizeof(float));
        blk->o_proj   = (float *)malloc(448 * 448 * sizeof(float));
        blk->g_proj   = (float *)malloc(448 * 448 * sizeof(float));
        blk->gate_up  = (float *)malloc(448 * 2456 * sizeof(float));
        blk->down     = (float *)malloc(1228 * 448 * sizeof(float));
        blk->q_norm   = (float *)malloc(64 * sizeof(float));
        blk->k_norm   = (float *)malloc(64 * sizeof(float));
        blk->attn_norm= (float *)malloc(448 * sizeof(float));
        blk->ffn_norm = (float *)malloc(448 * sizeof(float));
        if (!blk->q_proj || !blk->k_proj || !blk->v_proj || !blk->o_proj ||
            !blk->g_proj || !blk->gate_up || !blk->down || !blk->q_norm ||
            !blk->k_norm || !blk->attn_norm || !blk->ffn_norm) return -1;
        fill_rand(blk->q_proj, 448 * 448, 0.02f);
        fill_rand(blk->k_proj, 448 * 64, 0.02f);
        fill_rand(blk->v_proj, 448 * 64, 0.02f);
        fill_rand(blk->o_proj, 448 * 448, 0.02f);
        fill_rand(blk->g_proj, 448 * 448, 0.02f);
        fill_rand(blk->gate_up, 448 * 2456, 0.02f);
        fill_rand(blk->down, 1228 * 448, 0.02f);
        fill_rand(blk->q_norm, 64, 0.1f);
        fill_rand(blk->k_norm, 64, 0.1f);
        for (int i = 0; i < 448; i++) {
            blk->attn_norm[i] = 1.0f + 0.01f * frand();
            blk->ffn_norm[i]  = 1.0f + 0.01f * frand();
        }
    }
    float *selectors[WUBU_SELECTORS];
    for (int i = 0; i < WUBU_SELECTORS; i++) {
        selectors[i] = (float *)malloc(448 * sizeof(float));
        fill_rand(selectors[i], 448, 0.02f);
    }
    return wubu_model_init(m, emb, final_norm, blocks, selectors);
}

/* the mean-reduced CE from the REFERENCE forward's logits (parity) */
static float ref_loss(const wubu_model_t *m, const wubu_buf_t *b,
                      const uint16_t *tokens, int n)
{
    float loss = 0, n_pos = (float)(n - 1);
    for (int s = 0; s < n - 1; s++) {
        const float *lg = b->logits + (size_t)s * WUBU_VOCAB;
        float maxv = lg[0];
        for (int v = 1; v < WUBU_VOCAB; v++) if (lg[v] > maxv) maxv = lg[v];
        double sum = 0;
        for (int v = 0; v < WUBU_VOCAB; v++) sum += exp((double)(lg[v] - maxv));
        loss += (float)(((double)maxv + log(sum) - (double)lg[tokens[s + 1]]) / n_pos);
    }
    return loss;
}

/* one FD check: perturb *pw by h, re-forward, compare with ana */
static int fd_check(wubu_model_t *m, wubu_buf_t *b, wubu_bp_t *bp,
                    const uint16_t *tokens, int n,
                    float *pw, float ana, const char *name)
{
    const float h = 1e-2f;
    float orig = *pw;
    /* the perturbation is a weight change: tell the GPU weight cache
     * after EVERY perturb, or the second forward reuses the cached
     * first value and lp == lm exactly (the DA catch) */
    *pw = orig + h;
    if (gpu_wubu_mark_weights_dirty) gpu_wubu_mark_weights_dirty();
    float lp = wubu_bp_forward(m, b, bp, tokens, n);
    *pw = orig - h;
    if (gpu_wubu_mark_weights_dirty) gpu_wubu_mark_weights_dirty();
    float lm = wubu_bp_forward(m, b, bp, tokens, n);
    *pw = orig;
    float num = (lp - lm) / (2.0f * h);
    float diff = fabsf(num - ana);
    float tol = fmaxf(1e-3f, 0.1f * fabsf(ana));
    int ok = diff <= tol;
    if (!ok)
        printf("  FD MISMATCH %-22s ana=% .6f num=% .6f diff=% .6f (tol %g)\n",
               name, ana, num, diff, tol);
    return ok;
}

int main(void)
{
    printf("=== test_backprop (the REAL backward pass + Muon, FD-verified) ===\n");

    wubu_model_t m;
    if (make_model(&m) != 0) { printf("  FAIL: cannot build the random model\n"); return 1; }

    wubu_buf_t b;
    CHECK(wubu_buf_alloc(&b, 64) == 0, "buf alloc");

    wubu_bp_t bp;
    CHECK(wubu_bp_alloc(&bp, 64) == 0, "bp alloc");

    wubu_train_t tr;
    CHECK(wubu_train_init(&tr, &m) == 0, "train init");

    /* a small varied token sequence: the RNG advances per token (the
     * DA catch -- the old loop reused the same state and every token
     * was identical, so the embedding INPUT path was never exercised).
     * ids span 10..40, some repeated, so rows are both inputs and
     * (tied) head targets. */
    uint16_t tok[24];
    for (int i = 0; i < 24; i++) tok[i] = (uint16_t)(10 + (rng_next() % 31));
    {
        int uniq = 0;
        for (int i = 0; i < 24; i++) {
            int seen = 0;
            for (int j = 0; j < i; j++) if (tok[j] == tok[i]) seen = 1;
            if (!seen) uniq++;
        }
        printf("  token diversity: %d unique / 24 (first: %u %u %u %u)\n",
               uniq, (unsigned)tok[0], (unsigned)tok[1],
               (unsigned)tok[2], (unsigned)tok[3]);
        CHECK(uniq > 8, "tokens are varied (not a degenerate stream)");
    }

    /* ---- forward parity: the recording forward == the released forward */
    float loss_bp = wubu_bp_forward(&m, &b, &bp, tok, 24);
    wubu_forward(&m, &b, tok, 24);
    float loss_ref = ref_loss(&m, &b, tok, 24);
    printf("  loss bp %.6f vs released %.6f\n", loss_bp, loss_ref);
    CHECK(fabsf(loss_bp - loss_ref) < 1e-3f, "recording forward == released forward");

    /* ---- the analytic backward ---- */
    wubu_train_zero_grad(&tr);
    float loss_b = wubu_bp_backward(&m, &b, &bp, &tr, tok, 24);
    CHECK(fabsf(loss_bp - loss_b) < 1e-4f, "forward/backward loss agree");
    CHECK(tr.micro_steps == 1, "micro_steps counted");
    CHECK(tr.loss_sum > 0, "loss positive");

    /* ---- layers must specialize (the old bug gave them all the same
     * gradient -- the shared-proxy path). Real backprop gives each
     * layer its own. ---- */
    {
        double d = 0;
        for (size_t i = 0; i < 448 * 448; i++)
            d += fabsf(tr.q_proj_g[0][i] - tr.q_proj_g[1][i]);
        CHECK(d > 1e-6, "layers get DIFFERENT gradients (no shared proxy)");
    }

    /* ---- finite differences: one element of every parameter type ---- */
    int ok = 1;
    wubu_bp_forward(&m, &b, &bp, tok, 24);   /* fresh activations */

    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[0].q_proj[7 * 448 + 13], tr.q_proj_g[0][7 * 448 + 13], "q_proj[0]a");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[0].q_proj[400 * 448 + 300], tr.q_proj_g[0][400 * 448 + 300], "q_proj[0]b");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[0].k_proj[3 * 448 + 5],  tr.k_proj_g[0][3 * 448 + 5],  "k_proj[0]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[0].v_proj[2 * 448 + 9],  tr.v_proj_g[0][2 * 448 + 9],  "v_proj[0]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[0].o_proj[11 * 448 + 4], tr.o_proj_g[0][11 * 448 + 4], "o_proj[0]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[0].g_proj[6 * 448 + 17], tr.g_proj_g[0][6 * 448 + 17], "g_proj[0]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[0].gate_up[100 * 448 + 3], tr.gate_up_g[0][100 * 448 + 3], "gate_up[0]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[0].down[50 * 1228 + 21], tr.down_g[0][50 * 1228 + 21], "down[0]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[7].q_proj[9 * 448 + 8], tr.q_proj_g[7][9 * 448 + 8], "q_proj[7]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[7].o_proj[3 * 448 + 60], tr.o_proj_g[7][3 * 448 + 60], "o_proj[7]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[7].v_proj[10 * 448 + 4], tr.v_proj_g[7][10 * 448 + 4], "v_proj[7]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[11].down[3 * 1228 + 77], tr.down_g[11][3 * 1228 + 77], "down[11]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[3].attn_norm[10], tr.norm_g[4 * 3 + 0][10], "attn_norm[3]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[7].attn_norm[200], tr.norm_g[4 * 7 + 0][200], "attn_norm[7]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[7].ffn_norm[22], tr.norm_g[4 * 7 + 1][22], "ffn_norm[7]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[3].ffn_norm[300], tr.norm_g[4 * 3 + 1][300], "ffn_norm[3]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[0].q_norm[0], tr.norm_g[4 * 0 + 2][0], "q_norm[0]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[7].q_norm[50], tr.norm_g[4 * 7 + 2][50], "q_norm[7]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[0].k_norm[31], tr.norm_g[4 * 0 + 3][31], "k_norm[0]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.blocks[5].k_norm[10], tr.norm_g[4 * 5 + 3][10], "k_norm[5]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.final_norm[3], tr.norm_g[4 * WUBU_LAYERS][3], "final_norm");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.selectors[1][4], tr.norm_g[4 * WUBU_LAYERS + 1 + 1][4], "selectors[1]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.selectors[2][100], tr.norm_g[4 * WUBU_LAYERS + 1 + 2][100], "selectors[2]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.embedding[13 * 448 + 25], tr.emb_g[13 * 448 + 25], "embedding[13]");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.embedding[27 * 448 + 300], tr.emb_g[27 * 448 + 300], "embedding[27]");
    /* the INPUT path: rows that ARE input tokens (both input grad and,
     * when they are also targets, the tied-head grad) */
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.embedding[(size_t)tok[0] * 448 + 77],
                   tr.emb_g[(size_t)tok[0] * 448 + 77], "embedding[tok0] (input path)");
    ok &= fd_check(&m, &b, &bp, tok, 24, &m.embedding[(size_t)tok[3] * 448 + 300],
                   tr.emb_g[(size_t)tok[3] * 448 + 300], "embedding[tok3] (input path)");
    CHECK(ok, "all finite-difference gradient checks");
    if (ok) printf("  FD: all %d parameter checks match the numeric gradient\n", 26);

    /* ---- the Muon step must be finite and move the weights ---- */
    wubu_train_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.lr = 1e-3f;
    cfg.muon_momentum = 0.95f;
    cfg.weight_decay = 0.1f;
    cfg.warmup_steps = 2;
    cfg.max_steps = 10;
    float w_before = m.blocks[0].q_proj[0];
    CHECK(wubu_bp_muon_step(&m, &tr, &cfg, 3) == 0, "muon step runs");
    float w_after = m.blocks[0].q_proj[0];
    CHECK(w_after != w_before, "muon step moved the weights");
    CHECK(w_after == w_after, "weights finite after muon");

    /* the grads are consumed by the step */
    double consumed = 0;
    for (size_t i = 0; i < 448 * 448; i++) consumed += fabsf(tr.q_proj_g[0][i]);
    CHECK(consumed == 0.0, "gradients consumed by the step");

    /* ---- DA: the grad-clip path must work (recipe: clip 1.0) ---- */
    {
        wubu_train_zero_grad(&tr);
        wubu_bp_forward(&m, &b, &bp, tok, 24);
        wubu_bp_backward(&m, &b, &bp, &tr, tok, 24);
        wubu_train_cfg_t cc;
        memset(&cc, 0, sizeof(cc));
        cc.lr = 1e-3f; cc.weight_decay = 0.1f; cc.muon_momentum = 0.95f;
        cc.grad_clip = 1e-6f;    /* absurdly tight: everything gets scaled */
        cc.warmup_steps = 2; cc.max_steps = 10;
        CHECK(wubu_bp_muon_step(&m, &tr, &cc, 3) == 0, "grad-clipped step runs");
        int finite = 1;
        for (size_t i = 0; i < 448 * 448; i++)
            if (m.blocks[0].q_proj[i] != m.blocks[0].q_proj[i]) finite = 0;
        CHECK(finite, "weights finite after the clipped step");
        double cl = 0;
        for (size_t i = 0; i < 448 * 448; i++) cl += fabsf(tr.q_proj_g[0][i]);
        CHECK(cl == 0.0, "clipped step consumed the grads");
    }

    wubu_bp_free(&bp);
    wubu_train_free(&tr);
    wubu_free(&m, &b);

    if (failures == 0) printf("ALL BACKPROP TESTS PASSED -- the gradients are real\n");
    else printf("%d BACKPROP FAILURES\n", failures);
    return failures ? 1 : 0;
}
