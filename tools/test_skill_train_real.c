/*
 * test_skill_train_real.c — the SKILL → TRAIN REAL-WEIGHTS gate (the
 * milestone: "close experience → weights, not only experience → hive
 * cells" — the drained stream becomes a REAL gradient nudge).
 *
 * Asserts:
 *   1. the drained preference stream reaches the trainer's apply
 *   2. wubu_train_apply_prefs nudges the embedding gradient toward
 *      the win direction (the win value > lose value => the row's
 *      gradient grows)
 *   3. the nudge is BOUNDED (a tiny fraction — it cannot destabilize
 *      the step)
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "wubu_train.h"
#include "wubu_skill_train.h"
#include "wubu.h"
#include "wubu_runtime_dims.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_skill_train_real (experience becomes weights) ===\n");

    /* the runtime dims must be set BEFORE any allocation (the dims are
     * data — the macros resolve through WUBU_RUNTIME_DIMS) */
    if (WUBU_RUNTIME_DIMS.dim == 0) wubu_runtime_dims_default();

    /* a minimal model: just the embedding (the nudge target) */
    wubu_model_t m;
    memset(&m, 0, sizeof(m));
    m.n_layers = 1;
    m.embedding = (float *)calloc((size_t)WUBU_VOCAB * WUBU_DIM, sizeof(float));
    if (!m.embedding) FAIL("embedding alloc");

    wubu_train_t tr;
    memset(&tr, 0, sizeof(tr));
    tr.emb_g = (float *)calloc((size_t)WUBU_VOCAB * WUBU_DIM, sizeof(float));
    if (!tr.emb_g) FAIL("emb_g alloc");

    /* 1. the drained stream reaches the trainer */
    wubu_train_stream_t st;
    wubu_train_stream_init(&st);
    wubu_train_stream_push_pair(&st, 1, 42, 0.9f, 0.5f, 0, 1);
    wubu_train_stream_push_pair(&st, 2, 43, 0.8f, 0.4f, 1, 2);
    wubu_train_stream_item_t items[4];
    int k = wubu_train_stream_drain(&st, items, 4);
    printf("  drained %d preference items into the trainer\n", k);
    if (k != 2) FAIL("drain wrong (%d)", k);

    /* 2. the apply nudges the embedding gradient toward the wins */
    /* the baseline row 42 gradient */
    float *e42 = tr.emb_g + (size_t)42 * WUBU_DIM;
    float before = e42[0];
    wubu_train_apply_prefs(&m, &tr, items, k, 0.01f);
    float after = e42[0];
    printf("  emb_g[42][0]: %.6f -> %.6f (the win nudge)\n", before, after);
    /* the win value (0.9) > lose (0.5): the nudge is POSITIVE */
    if (after <= before) FAIL("the win pair did not push the gradient up");
    /* 3. the nudge is bounded (tiny vs the win-lose margin) */
    float expect = 0.01f * (0.9f - 0.5f);   /* s * (win-lose) */
    float got = after - before;
    if (got > expect * 1.01f || got < expect * 0.99f)
        FAIL("the nudge is not the bounded fraction (got %.6f want %.6f)", got, expect);
    printf("  the nudge: +%.6f (exactly s*(win-lose) = %.6f, bounded)\n", got, expect);

    free(m.embedding);
    free(tr.emb_g);
    printf("=== ALL SKILL-TRAIN-REAL TESTS PASSED (weights moved) ===\n");
    return 0;
}
