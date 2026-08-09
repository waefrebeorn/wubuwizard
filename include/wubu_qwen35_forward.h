/*
 * wubu_qwen35_forward.h — the QWEN3.5 HYBRID FORWARD (AN28 "build
 * next", after the GGUF-corrected AN61 role math). C11.
 *
 * One hybrid layer forward, config-driven, consuming the tensors the
 * loader placed. Two branches:
 *
 *   gated-attn branch (the UNFUSED GGUF layout):
 *     qg = x @ attn_q [d, 2*d_out]          -> q (first d_out), gate
 *     k  = x @ attn_k [d, kv*hd]
 *     v  = x @ attn_v [d, kv*hd]
 *     q = rmsnorm(q, q_norm[hd]); k = rmsnorm(k, k_norm[hd])
 *     q,k = rope (partial 0.25, theta 1e7); GQA softmax
 *     attn = attn * sigmoid(gate); out = attn @ W_o [d_out, d]
 *
 *   GDN branch (Gated DeltaNet, the fused GGUF layout):
 *     qkv = x @ attn_qkv [d, qk+v]; z = x @ attn_gate [d, value_dim]
 *     mixed = causal conv1d(qkv, kernel 4)  (the qk+v width)
 *     the delta rule: state *= exp(-a); state += beta*outer(k, v - s@k)
 *     out = (state @ q) * silu(z) @ ssm_out [value_dim, d]
 *
 * The recurrence (the conv state + the delta state) is the ssm
 * module's; this module exercises the projection + gate + attention
 * paths shape-correctly so the loader wiring can be verified before
 * the parity harness.
 */
#ifndef WUBU_QWEN35_FORWARD_H
#define WUBU_QWEN35_FORWARD_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_qwen35.h"

/* the per-layer weights (the loader fills these from the GGUF) */
typedef struct {
    /* the shared input norm + the post-attention norm */
    const float *attn_norm;       /* [d] RMSNorm (1e-6) */
    const float *post_norm;       /* [d] */
    /* the gated-attn branch (present for full_attention) */
    const float *attn_q;          /* [d, 2*d_out] q_and_gate */
    const float *attn_k;          /* [d, kv*hd] */
    const float *attn_v;          /* [d, kv*hd] */
    const float *attn_output;     /* [d_out, d] W_o */
    const float *q_norm;          /* [head_dim] shared per-head Q norm */
    const float *k_norm;          /* [head_dim] shared per-head K norm */
    /* the GDN branch (present for linear_attention) */
    const float *ssm_qkv;         /* [d, gdn_qkv_total] the fused qk+v */
    const float *ssm_gate;        /* [d, value_dim] the z gate */
    const float *ssm_conv;        /* [lin_conv_kernel, gdn_conv_total] */
    const float *ssm_alpha;       /* [d, lin_num_key_heads] */
    const float *ssm_beta;        /* [d, lin_num_key_heads] */
    const float *ssm_dt_bias;     /* [lin_num_key_heads] */
    const float *ssm_norm;        /* [lin_value_head_dim] per-value RMS */
    const float *ssm_out;         /* [gdn_out_rows, d] */
    const float *ssm_a;           /* [lin_num_key_heads] the decay */
} wubu_q35_weights_t;

/* F1: one hybrid layer forward. x is [seq, d]; out is [seq, d].
 * The recurrent GDN state is the ssm module's (NULL here = the
 * stateless projection path). Returns 0 on success. */
int wubu_q35_layer_forward(const wubu_q35_cfg_t *cfg,
                           const wubu_q35_split_t *split,
                           const wubu_q35_weights_t *w,
                           wubu_q35_layer_kind_t kind,
                           const float *x, int seq, float *out,
                           float *gdn_state);

/* F2: the gated-attn branch alone (testable in isolation). */
int wubu_q35_gated_attn_forward(const wubu_q35_cfg_t *cfg,
                                const wubu_q35_split_t *split,
                                const wubu_q35_weights_t *w,
                                const float *x, int seq, float *out_a);

#endif
