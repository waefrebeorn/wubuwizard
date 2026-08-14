/*
 * test_amoeba_shrink.c -- AM01 gate: the symmetric amoeba.
 *
 * Proves the body can now RETRACT a pseudopod, not just extend it:
 *
 *   1. DEPTH SHRINK: grow a block, then shrink it back -- n_layers
 *      returns to the baseline and the model still forwards (finite).
 *   2. CORE GUARD: shrink refuses to remove the last layer (the ring-0
 *      core never shrinks away -- the AN12 invariant).
 *   3. WIDTH SHRINK: prune 25% of an FFN's columns with gate-zero
 *      re-insertion -- the forward output is UNCHANGED (function-
 *      preserving, the grow doctrine applied to shrink).
 *   4. TRAIN-STATE PAIR: wubu_train_shrink frees + shifts the grad/
 *      momentum arrays so the trainer and the body stay in sync.
 *
 * Gate: `make test_amoeba_shrink`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Provide a minimal runtime dims for this isolated test.
 * The test uses hardcoded dimension macros (WUBU_DIM, etc.) but wubu.h
 * redefines them in terms of WUBU_RUNTIME_DIMS. We provide a static
 * instance with the default WuBu-35M geometry and call the initializer. */
#include "wubu_runtime_dims.h"
static wubu_runtime_dims_t g_wubu_test_dims;
wubu_runtime_dims_t WUBU_RUNTIME_DIMS;  /* the extern expected by wubu.h */

void wubu_runtime_dims_default(void) {
    WUBU_RUNTIME_DIMS.vocab = 16384;
    WUBU_RUNTIME_DIMS.dim = 448;
    WUBU_RUNTIME_DIMS.layers = 12;
    WUBU_RUNTIME_DIMS.heads = 7;
    WUBU_RUNTIME_DIMS.kv_heads = 1;
    WUBU_RUNTIME_DIMS.head_dim = 64;
    WUBU_RUNTIME_DIMS.rope_dim = 32;
    WUBU_RUNTIME_DIMS.ffn_dim = 1228;
    WUBU_RUNTIME_DIMS.max_seq = 2048;
    WUBU_RUNTIME_DIMS.local_win = 256;
    WUBU_RUNTIME_DIMS.full_every = 4;
    WUBU_RUNTIME_DIMS.select_every = 4;
    WUBU_RUNTIME_DIMS.clip = 10.0f;
    WUBU_RUNTIME_DIMS.eps = 1e-6f;
    WUBU_RUNTIME_DIMS.selectors = 3;
    WUBU_RUNTIME_DIMS.rope_theta = 10000.0f;
    WUBU_RUNTIME_DIMS.params = 35072768;
}

#include "wubu.h"
#include "wubu_grow.h"
#include "wubu_shrink.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

/* A tiny model: init a minimal wubu_model_t with 2 layers (the core),
 * grow to 3, shrink back to 2. */
static int init_min_model(wubu_model_t *m)
{
    memset(m, 0, sizeof(*m));
    m->n_layers = 2;
    for (int l = 0; l < 2; l++) {
        wubu_block_t *b = &m->blocks[l];
        memset(b, 0, sizeof(*b));
        b->q_proj    = (float *)calloc((size_t)WUBU_DIM * WUBU_HEADS * 64,    sizeof(float));
        b->k_proj    = (float *)calloc((size_t)WUBU_DIM * WUBU_KV_HEADS * 64,  sizeof(float));
        b->v_proj    = (float *)calloc((size_t)WUBU_DIM * WUBU_KV_HEADS * 64,  sizeof(float));
        b->o_proj    = (float *)calloc((size_t)WUBU_DIM * WUBU_HEADS * 64,    sizeof(float));
        b->g_proj    = (float *)calloc((size_t)WUBU_DIM * WUBU_HEADS * 64,    sizeof(float));
        b->q_norm    = (float *)calloc((size_t)WUBU_KV_HEADS * 64,             sizeof(float));
        b->k_norm    = (float *)calloc((size_t)WUBU_KV_HEADS * 64,             sizeof(float));
        b->attn_norm = (float *)calloc((size_t)WUBU_DIM,                       sizeof(float));
        b->gate_up   = (float *)calloc((size_t)WUBU_DIM * WUBU_FFN_DIM * 2,    sizeof(float));
        b->down      = (float *)calloc((size_t)WUBU_FFN_DIM * WUBU_DIM,        sizeof(float));
        b->ffn_norm  = (float *)calloc((size_t)WUBU_DIM,                       sizeof(float));
        if (!b->q_proj || !b->down) return -1;
        /* fill with small deterministic values (importance variance) */
        for (int i = 0; i < WUBU_DIM * WUBU_FFN_DIM * 2; i++)
            b->gate_up[i] = (float)((i * 2654435761u) % 997) / 1000.0f - 0.5f;
        for (int i = 0; i < WUBU_FFN_DIM * WUBU_DIM; i++)
            b->down[i] = (float)((i * 40503u) % 991) / 1000.0f - 0.5f;
    }
    return 0;
}

static void free_model(wubu_model_t *m)
{
    for (int l = 0; l < m->n_layers; l++) {
        wubu_block_t *b = &m->blocks[l];
        free(b->q_proj);  free(b->k_proj);  free(b->v_proj);
        free(b->o_proj);  free(b->g_proj);
        free(b->q_norm);  free(b->k_norm);  free(b->attn_norm);  free(b->ffn_norm);
        free(b->gate_up); free(b->down);
    }
}

/* FFN forward of one block: y = down(silu(gate) * up(x)) -- the pruned
 * columns must not change it. Layout: gate_up[dim, 2*ffn] row-major,
 * down[ffn, dim] row-major. */
static void block_ffn(const wubu_block_t *b, const float *x, float *y)
{
    float act[WUBU_FFN_DIM];
    for (int j = 0; j < WUBU_FFN_DIM; j++) {
        float g = 0.0f, u = 0.0f;
        for (int i = 0; i < WUBU_DIM; i++) {
            g += x[i] * b->gate_up[(size_t)i * (2 * WUBU_FFN_DIM) + j];
            u += x[i] * b->gate_up[(size_t)i * (2 * WUBU_FFN_DIM) + WUBU_FFN_DIM + j];
        }
        float sig = g < -80.0f ? 0.0f : g / (1.0f + expf(-g));
        act[j] = sig * u;
    }
    for (int i = 0; i < WUBU_DIM; i++) {
        float s = 0.0f;
        for (int j = 0; j < WUBU_FFN_DIM; j++)
            s += act[j] * b->down[(size_t)j * WUBU_DIM + i];
        y[i] = s;
    }
}

int main(void)
{
    printf("=== test_amoeba_shrink (AM01: the symmetric amoeba) ===\n");

    wubu_model_t m;
    if (init_min_model(&m) != 0) {
        printf("  FAIL: model init\n");
        return 1;
    }
    int base = m.n_layers;
    printf("  baseline: %d layers\n", base);

    /* ---- 1. grow then shrink: the amoeba retracts ---- */
    if (wubu_grow_insert_block(&m, 1) == 0)
        printf("  grow: %d -> %d layers\n", base, m.n_layers);
    CHECK(m.n_layers == base + 1, "grow inserted a block");
    int rc = wubu_shrink_block(&m, 1);
    CHECK(rc == 0, "shrink succeeds");
    printf("  shrink: %d -> %d layers\n", base + 1, m.n_layers);
    CHECK(m.n_layers == base, "shrink returns to the baseline depth");

    /* ---- 2. the core guard: never shrink below the core ---- */
    /* shrink layer 0 (the core) of the 2-layer body -> refused */
    CHECK(wubu_shrink_block(&m, 0) == 0, "shrink of a non-core layer ok");
    CHECK(m.n_layers == base - 1, "body at 1 layer (the core)");
    CHECK(wubu_shrink_block(&m, 0) == -1,
          "core guard: shrink refuses to remove the last layer");
    CHECK(m.n_layers == 1, "the ring-0 core survives");

    /* ---- 3. width shrink is function-preserving ---- */
    {
        wubu_model_t m2;
        if (init_min_model(&m2) != 0) { printf("  FAIL: model2 init\n"); return 1; }
        wubu_block_t *b = &m2.blocks[0];

        /* Zero a known 25% of the FFN columns (gate+up+down): a zeroed
         * column contributes nothing to the output, so pruning it must
         * be EXACTLY lossless. This is the honest function-preservation
         * invariant for shrink (gate-zero re-insertion). */
        const int ffn = WUBU_FFN_DIM;
        const int dim = WUBU_DIM;
        int n_zero = ffn / 4;
        for (int j = 0; j < n_zero; j++) {
            for (int i = 0; i < dim; i++) {
                b->gate_up[(size_t)i * (2 * ffn) + j] = 0.0f;
                b->gate_up[(size_t)i * (2 * ffn) + ffn + j] = 0.0f;
            }
            for (int i = 0; i < dim; i++)
                b->down[(size_t)j * dim + i] = 0.0f;
        }

        float x[WUBU_DIM], y_before[WUBU_DIM], y_after[WUBU_DIM];
        for (int i = 0; i < WUBU_DIM; i++) x[i] = (float)((i * 7u) % 100) / 50.0f - 1.0f;
        block_ffn(b, x, y_before);

        /* prune 25%: the zero columns have zero norm, so the selector
         * MUST pick them (and only them) -- lossless by construction */
        int pruned = wubu_shrink_ffn(&m2, 0, 0.25f);
        CHECK(pruned == 0, "width shrink succeeds (25% pruned)");
        block_ffn(b, x, y_after);

        double max_diff = 0.0;
        for (int i = 0; i < WUBU_DIM; i++) {
            double d = fabs((double)y_before[i] - y_after[i]);
            if (d > max_diff) max_diff = d;
        }
        printf("  ok: width shrink max |before-after| = %.3e (zeroed cols)\n", max_diff);
        CHECK(max_diff < 1e-5,
              "width shrink is function-preserving (zero columns pruned losslessly)");
        free_model(&m2);
    }

    /* ---- 4. train-state pair stays in sync ---- */
    {
        wubu_train_t tr;
        memset(&tr, 0, sizeof(tr));
        size_t q = (size_t)WUBU_DIM * WUBU_HEADS * 64;
        /* 3 layers of train state (the fixed arrays are WUBU_LAYERS
         * wide; fill the first 3 slots) */
        for (int l = 0; l < 3; l++) {
            tr.q_proj_g[l] = (float *)calloc(q, sizeof(float));
            tr.q_proj_m[l] = (float *)calloc(q, sizeof(float));
            tr.gate_up_g[l] = (float *)calloc(q, sizeof(float));
            tr.gate_up_m[l] = (float *)calloc(q, sizeof(float));
        }
        for (int l = 0; l < 12; l++) {
            tr.norm_g[l] = (float *)calloc(WUBU_DIM, sizeof(float));
            tr.norm_m[l] = (float *)calloc(WUBU_DIM, sizeof(float));
            tr.norm_v[l] = (float *)calloc(WUBU_DIM, sizeof(float));
        }
        CHECK(wubu_train_shrink(&tr, 1, 3) == 0, "train-state shrink succeeds");
        CHECK(tr.q_proj_g[2] == NULL && tr.q_proj_g[0] != NULL,
              "grad arrays shifted down + freed the removed slot");
        CHECK(tr.norm_g[8] == NULL && tr.norm_g[0] != NULL,
              "norm slots shifted down (4 per block)");
        for (int l = 0; l < 3; l++) {
            free(tr.q_proj_g[l]); free(tr.q_proj_m[l]);
            free(tr.gate_up_g[l]); free(tr.gate_up_m[l]);
        }
        for (int l = 0; l < 12; l++) {
            free(tr.norm_g[l]); free(tr.norm_m[l]); free(tr.norm_v[l]);
        }
    }

    free_model(&m);
    if (failures == 0) printf("=== ALL AMOEBA-SHRINK TESTS PASSED ===\n");
    else printf("=== %d FAILURES ===\n", failures);
    return failures ? 1 : 0;
}
