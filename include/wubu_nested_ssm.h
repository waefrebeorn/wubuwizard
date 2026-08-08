#ifndef WUBU_NESTED_SSM_H
#define WUBU_NESTED_SSM_H

#include "wubu_ssm.h"
#include "wubu_mobius.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Nested SSM: Product of K Poincaré balls with K curvatures.
 *
 * Generalizes the single-ball Poincaré SSM to K balls with different radii:
 *   h[t] = (h_1[t], ..., h_K[t])  where each h_k lives in a ball of radius R_k
 *
 * Each ball operates independently on the full state dimension:
 *   h_k[t] = mobius_add(scalar_mul(exp(gate), h_k[t-1]),
 *                        exp_map((k ⊗ (log_map(v) - log_map(h_k[t-1] @ k))) * beta, R_k))
 *
 * The gating mechanism uses the same beta/gate for all K balls, but a learned
 * per-ball weight w_k (softmax-normalized) controls each ball's contribution
 * to the final output.
 */

#define NESTED_SSM_MAX_K 16

/**
 * Nested SSM state: K independent Poincaré ball state matrices.
 *
 * Each ball has its own state matrix [SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
 * and operates with its curvature radius R_k.
 *
 * Layout: states[k * HEAD_STATE_SIZE + (vh * SSM_D_STATE + i) * SSM_D_STATE + j]
 *   where HEAD_STATE_SIZE = SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE
 */
typedef struct {
    int K;          // Number of balls (1 <= K <= NESTED_SSM_MAX_K)
    float R[NESTED_SSM_MAX_K];  // Curvatures [K]
    float *states;  // [K * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE] — flat array
} wubu_nested_ssm_state_t;

/**
 * Nested SSM gating: per-ball learned weights for combining outputs.
 * Pass NULL for uniform weighting (1/K each).
 */
typedef struct {
    float ball_weights[NESTED_SSM_MAX_K];  // [K] — will be softmax-normalized at runtime
} wubu_nested_ssm_gating_t;

/**
 * Forward pass for nested Poincaré SSM (K balls, K curvatures).
 */
void wubu_nested_ssm_forward(const float *x, int B, int T,
                              const ssm_layer_weights *weights,
                              wubu_nested_ssm_state_t *nested_state,
                              float *conv_state,
                              const wubu_nested_ssm_gating_t *gating,
                              float *output);

/**
 * Saved intermediates for nested SSM backward pass.
 * All arrays [B*T x dim] unless noted.
 *
 * Ball deltas and per-timestep state trajectory are saved for BPTT.
 */
typedef struct {
    // Standard SSM intermediates (same as ssm_fwd_save_t but nested)
    float *qkv_all;        // [N, CONV_DIM]
    float *z_all;          // [N, VALUE_DIM]
    float *beta_raw;       // [N, DT_RANK]
    float *alpha_raw;      // [N, DT_RANK]
    float *conv_post_silu; // [N, CONV_DIM]
    float *q_conv;         // [N, KEY_DIM]
    float *k_conv;         // [N, KEY_DIM]
    float *v_conv;         // [N, VALUE_DIM]
    float *q_norm;         // [N, KEY_DIM]
    float *k_norm;         // [N, KEY_DIM]
    float *delta_out;      // [N, VALUE_DIM] (pre-gated-norm, combined)
    float *z_silu;         // [N, VALUE_DIM]
    float *beta_flat;      // [N, DT_RANK] sigmoid(beta_raw)
    float *gate_flat;      // [N, DT_RANK] alpha_softplus * ssm_a
    float *conv_state_copy;// [B, CONV_KERNEL-1, CONV_DIM]

    // Nested SSM specific: number of balls (for safe cleanup)
    int K;

    // Per-ball deltas (pre-combination)
    float **ball_deltas;   // [K][N, VALUE_DIM]

    // Per-timestep state trajectory for BPTT
    // Layout: states_t[t][k][HEAD_STATE_SZ] where t=0..T, t=0 is initial state
    // HEAD_STATE_SZ = SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE
    float *states_t;       // [(T+1) * K * HEAD_STATE_SZ]

    // Per-ball output before gated norm (for backprop through ball combination)
    float *ball_delta_flat; // [K * N * VALUE_DIM] flattened version of ball_deltas

    // Softmax-normalized gating weights [K]
    float w_norm[NESTED_SSM_MAX_K];
} nested_ssm_fwd_save_t;

/**
 * Nested SSM forward with save (pass save=NULL for standard forward).
 */
void wubu_nested_ssm_forward_save(const float *x, int B, int T,
                                   const ssm_layer_weights *weights,
                                   wubu_nested_ssm_state_t *nested_state,
                                   float *conv_state,
                                   const wubu_nested_ssm_gating_t *gating,
                                   float *output,
                                   nested_ssm_fwd_save_t *save);

/**
 * Nested SSM backward pass.
 * Uses saved intermediates from wubu_nested_ssm_forward_save.
 *
 * BPTT through K independent Poincaré ball recurrences.
 */
void wubu_nested_ssm_backward(
    int B, int T,
    const float *x,                    // [B*T, D_MODEL] forward input
    const float *output,               // [B*T, D_MODEL] forward output
    const float *d_output,             // [B*T, D_MODEL] upstream gradient
    const float *ball_weights_raw,     // [K] pre-softmax ball weights (or NULL for uniform)
    const wubu_nested_ssm_state_t *nested_state,  // FINAL state (after all timesteps)
    const nested_ssm_fwd_save_t *save, // Saved intermediates from forward_save
    const ssm_layer_weights *w,        // SSM layer weights
    float *d_x,                        // [B*T, D_MODEL] gradient w.r.t. input
    // Weight gradients (all can be NULL to skip weight updates)
    float *d_qkv_weight,               // [D_MODEL, CONV_DIM]
    float *d_gate_weight,              // [D_MODEL, VALUE_DIM]
    float *d_beta_weight,              // [D_MODEL, DT_RANK]
    float *d_alpha_weight,             // [D_MODEL, DT_RANK]
    float *d_conv1d_weight,            // [CONV_KERNEL, CONV_DIM]
    float *d_ssm_out_weight,           // [VALUE_DIM, D_MODEL]
    float *d_ssm_norm_weight,          // [SSM_D_STATE]
    float *d_state_init_grad,          // [K * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE]
    float *d_ball_weights_raw          // [K] gradient w.r.t. pre-softmax ball weights (or NULL)
);

/**
 * Free memory allocated in a nested_ssm_fwd_save_t struct.
 */
void wubu_nested_ssm_fwd_save_free(nested_ssm_fwd_save_t *save);

/**
 * Initialize a nested SSM state with K balls and given curvatures.
 */
int wubu_nested_ssm_init(wubu_nested_ssm_state_t *state, int K, const float *R);

/**
 * Free memory allocated by wubu_nested_ssm_init.
 */
void wubu_nested_ssm_free(wubu_nested_ssm_state_t *state);

/**
 * Check that all K balls' states are valid (no NaN, within radius).
 * Returns 0 if valid, 1 if any invalid state found.
 */
int wubu_nested_ssm_validate(const wubu_nested_ssm_state_t *state);

#ifdef __cplusplus
}
#endif

#endif // WUBU_NESTED_SSM_H
