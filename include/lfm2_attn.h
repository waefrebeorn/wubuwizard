#ifndef LFM2_ATTN_H
#define LFM2_ATTN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* LFM2.5 GQA attention with q/k layernorm + RoPE + KV cache.
 * Self-contained: owns the RoPE, GQA group pooling, and softmax.
 *
 * x: [T, d_model] (already operator_norm'd).
 * kv_cache_layer: [2, kv_max_t, kv_dim] interleaved (K then V) or NULL.
 * start_pos: number of positions already in the cache (prefill offset).
 * Writes attn_out: [T, d_model]. */

void lfm2_gqa(const float *q_proj, const float *k_proj, const float *v_proj,
               const float *o_proj, const float *q_ln, const float *k_ln,
               int n_q_heads, int n_kv_heads, int head_dim, int d_model,
               float rope_theta, const float *x, int T,
               float *kv_cache_layer, int kv_max_t, int start_pos,
               float *attn_out);

void lfm2_gqa_q(const float *q_proj, const float *k_proj, const float *v_proj,
                const float *o_proj, const float *q_ln, const float *k_ln,
                int n_q_heads, int n_kv_heads, int head_dim, int d_model,
                float rope_theta, const float *x, int T,
                float *kv_cache_layer, int kv_max_t, int start_pos,
                float *attn_out,
                const uint8_t *q_q_proj, int q_q_t,
                const uint8_t *q_k_proj, int q_k_t,
                const uint8_t *q_v_proj, int q_v_t,
                const uint8_t *q_o_proj, int q_o_t);

#ifdef __cplusplus
}
#endif

#endif /* LFM2_ATTN_H */
