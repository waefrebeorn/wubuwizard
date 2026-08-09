/* wubu1_block_bridge.c — lossless conversion: loaded wubu_layer_t → canonical
 * wubu1_block_t.
 *
 * The WuBu1 direction (COHESIVE_DIRECTION.md P1 Step 2): one canonical block,
 * end to end. Until the forward pass consumes wubu1_block_t directly, this
 * bridge converts a role-resolved wubu_layer_t (the current loader output)
 * into the canonical form — proving the mapping is lossless for every shared
 * weight pointer and per-layer dim.
 *
 * C11, minimal includes. Opaque-free: the block is the seam.
 */
#include "wubu1_block.h"
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include <string.h>

/* Fill *out from a loaded layer. Returns 1 on success, 0 if layer is NULL.
 * Every shared pointer/dim is copied; geometry/KV/hive fields get the
 * block's defaults (they are set by the grow operators / checkpoint). */
int wubu1_block_fill(wubu1_block_t *out, const wubu_layer_t *layer) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!layer) return 0;

    /* norms (shared by both GQA and SSM paths) */
    out->attn_norm = layer->attn_norm_weight;
    out->post_attn_norm = layer->post_attn_norm_weight;

    if (layer->is_ssm) {
        const ssm_layer_weights *s = &layer->ssm;
        /* SSM has no per-head GQA Q/K/V — the fused QKV is the attention.
         * Map it into the canonical q_proj slot (type + blob pointer). */
        out->q_proj = s->attn_qkv_weight;
        out->q_proj_q = s->attn_qkv_weight_q;
        out->q_proj_q_type = s->attn_qkv_weight_type;
        out->g_proj = s->attn_gate_weight;
        out->g_proj_q = s->attn_gate_weight_q;
        out->g_proj_q_type = s->attn_gate_weight_type;
        out->down = s->ssm_out_weight;
        out->down_q = s->ssm_out_weight_q;
        out->down_q_type = s->ssm_out_weight_type;
        /* dims: SSM layers have no GQA q/kv heads. */
        out->q_dim = 0;
        out->kv_dim = 0;
        out->out_dim = 0;
    } else {
        const gqa_layer_weights *g = &layer->gqa;
        out->q_proj = g->attn_q_weight;
        out->q_proj_q = g->attn_q_weight_q;
        out->q_proj_q_type = g->attn_q_weight_type;
        out->k_proj = g->attn_k_weight;
        out->k_proj_q = g->attn_k_weight_q;
        out->k_proj_q_type = g->attn_k_weight_type;
        out->v_proj = g->attn_v_weight;
        out->v_proj_q = g->attn_v_weight_q;
        out->v_proj_q_type = g->attn_v_weight_type;
        out->o_proj = g->attn_output_weight;
        out->o_proj_q = g->attn_output_weight_q;
        out->o_proj_q_type = g->attn_output_weight_type;
        out->q_norm = g->attn_q_norm_weight;
        out->k_norm = g->attn_k_norm_weight;
        /* dims (runtime geometry, Theory/08 aligned) */
        out->q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
        out->kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
        out->out_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
    }

    /* MoE: keep the routed-expert pointers; the canonical dense FFN fields
     * stay NULL when the layer is MoE (the forward migrates to wubu1_block_t
     * with expert fields in the next slice). */
    if (layer->moe.loaded) {
        /* Dense FFN-first: MoE routed experts do not map into the dense
         * gate_up/down slots. Recorded via the router only. */
        out->gate_up = NULL;
        out->down = NULL;
    }

    /* hive tissue: slot id + rhythm defaults (grown later). */
    out->slot = layer->layer_idx;
    out->is_full = 0;
    out->fire_sel = 0;
    out->curvature = 1.0f;      /* unit ball default */
    out->kv_curvature = 1.0f;
    out->kv_precision = 2;      /* F16 default (Escha ladder) */
    return 1;
}
