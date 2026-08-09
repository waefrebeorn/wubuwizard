/* wubu_shrink.c -- the model-shrink operators (AM01: the symmetric amoeba).
 *
 * Mirrors wubu_grow.c exactly: grow shifts arrays UP and allocates;
 * shrink shifts DOWN and frees. The body can now retract a pseudopod.
 *
 * The doctrine: grow is function-preserving (forward-before ==
 * forward-after); shrink is FITNESS-GATED (BI-informed choice + loss
 * tolerance + prover). The shrink operator itself does the mechanical
 * removal; the validate/commit loop (wubu_amoeba_validate) gates it.
 *
 * C11.
 */
#include "wubu_shrink.h"

#include <stdlib.h>
#include <string.h>

/* ---- depth shrink (ShortGPT) ---- */

int wubu_shrink_block(wubu_model_t *m, int pos)
{
    if (!m || pos < 0 || pos >= m->n_layers) return -1;
    if (m->n_layers <= 1) return -1;   /* never shrink below the core */

    /* Free the removed block's buffers (same per-field free as grow). */
    wubu_block_t *victim = &m->blocks[pos];
    free(victim->q_proj);  free(victim->k_proj);  free(victim->v_proj);
    free(victim->o_proj);  free(victim->g_proj);
    free(victim->q_norm);  free(victim->k_norm);
    free(victim->attn_norm); free(victim->ffn_norm);
    free(victim->gate_up); free(victim->down);

    /* Shift the later blocks down (struct copy; the freed victim's slot
     * is overwritten by the first shift). */
    for (int l = pos; l < m->n_layers - 1; l++)
        m->blocks[l] = m->blocks[l + 1];
    memset(&m->blocks[m->n_layers - 1], 0, sizeof(wubu_block_t));

    m->n_layers--;
    return 0;
}

/* ---- width shrink (low-norm prune, gate-zero re-insert) ---- */

int wubu_shrink_ffn(wubu_model_t *m, int pos, float prune_frac)
{
    if (!m || pos < 0 || pos >= m->n_layers) return -1;
    if (prune_frac <= 0.0f || prune_frac >= 1.0f) return -1;
    wubu_block_t *b = &m->blocks[pos];
    if (!b->gate_up || !b->down) return -1;

    const int dim = WUBU_DIM;
    const int ffn = WUBU_FFN_DIM;
    const int n_prune = (int)(ffn * prune_frac);
    if (n_prune < 1) return 0;

    /* Score the FFN columns by their gate_up + down norm (importance).
     * LAYOUT: gate_up is [dim, 2*ffn] row-major (gate cols 0..ffn-1,
     * up cols ffn..2*ffn-1); down is [ffn, dim] row-major. */
    float *score = (float *)malloc((size_t)ffn * sizeof(float));
    if (!score) return -1;
    for (int j = 0; j < ffn; j++) {
        float s = 0.0f;
        for (int i = 0; i < dim; i++) {
            float g = b->gate_up[(size_t)i * (2 * ffn) + j];
            float u = b->gate_up[(size_t)i * (2 * ffn) + ffn + j];
            s += g * g + u * u;
        }
        for (int i = 0; i < dim; i++) {
            float d = b->down[(size_t)j * dim + i];
            s += d * d;
        }
        score[j] = s;
    }

    /* Find the n_prune lowest-norm columns (repeated min selection). */
    uint8_t *prune = (uint8_t *)calloc((size_t)ffn, 1);
    if (!prune) { free(score); return -1; }
    for (int p = 0; p < n_prune; p++) {
        int worst = -1;
        for (int j = 0; j < ffn; j++)
            if (!prune[j] && (worst < 0 || score[j] < score[worst]))
                worst = j;
        if (worst >= 0) prune[worst] = 1;
    }
    free(score);

    /* Zero the pruned columns' weights (gate, up, down). Function-
     * preserving: the forward before == after for the pruned path (a
     * zeroed column contributes nothing to the FFN output). */
    for (int j = 0; j < ffn; j++) {
        if (!prune[j]) continue;
        for (int i = 0; i < dim; i++) {
            b->gate_up[(size_t)i * (2 * ffn) + j] = 0.0f;
            b->gate_up[(size_t)i * (2 * ffn) + ffn + j] = 0.0f;
        }
        for (int i = 0; i < dim; i++)
            b->down[(size_t)j * dim + i] = 0.0f;
    }
    free(prune);
    return 0;
}

/* ---- train-state pair (the mirror of wubu_train_grow) ---- */

int wubu_train_shrink(wubu_train_t *tr, int pos, int n_layers)
{
    if (!tr || pos < 0 || pos >= n_layers || n_layers <= 1) return -1;

    /* free the removed slot, then shift the arrays DOWN (mirror of
     * wubu_train_grow's SHIFT_ARR which shifts up + allocates) */
#define SHIFT_DOWN(ARR) do {                                            \
        if (ARR[pos]) free(ARR[pos]);                                   \
        for (int l = pos; l < n_layers - 1; l++) ARR[l] = ARR[l + 1];   \
        ARR[n_layers - 1] = NULL;                                       \
    } while (0)
    SHIFT_DOWN(tr->q_proj_g); SHIFT_DOWN(tr->k_proj_g);
    SHIFT_DOWN(tr->v_proj_g); SHIFT_DOWN(tr->o_proj_g);
    SHIFT_DOWN(tr->g_proj_g); SHIFT_DOWN(tr->gate_up_g);
    SHIFT_DOWN(tr->down_g);
    SHIFT_DOWN(tr->q_proj_m); SHIFT_DOWN(tr->k_proj_m);
    SHIFT_DOWN(tr->v_proj_m); SHIFT_DOWN(tr->o_proj_m);
    SHIFT_DOWN(tr->g_proj_m); SHIFT_DOWN(tr->gate_up_m);
    SHIFT_DOWN(tr->down_m);
#undef SHIFT_DOWN

    /* the AdamW norm slots [4l+0..3]: free + shift down (g, m, v) */
#define SHIFT_NORM(ARR) do {                                            \
        for (int e = 0; e < 4; e++) {                                   \
            if (ARR[4 * pos + e]) free(ARR[4 * pos + e]);               \
            for (int l = pos; l < n_layers - 1; l++)                    \
                ARR[4 * l + e] = ARR[4 * (l + 1) + e];                  \
            ARR[4 * (n_layers - 1) + e] = NULL;                         \
        }                                                               \
    } while (0)
    SHIFT_NORM(tr->norm_g); SHIFT_NORM(tr->norm_m); SHIFT_NORM(tr->norm_v);
#undef SHIFT_NORM
    return 0;
}
