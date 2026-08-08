#include "wubu_poincare_gqa.h"
#include "wubu_mobius.h"
#include "gguf_reader.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================
// Helper: forward declarations for static helpers
// ============================================================
static void wubu_poincare_gqa_backward_attention(
    int B, int T, float R,
    const float *Q_ball,      // [N, GQA_Q_HEADS * GQA_HEAD_DIM]
    const float *K_ball,      // [N, GQA_KV_HEADS * GQA_HEAD_DIM]
    const float *V_ball,      // [N, GQA_KV_HEADS * GQA_HEAD_DIM]
    const float *d_attn_out,  // [N, GQA_Q_HEADS * GQA_HEAD_DIM] upstream grad
    float *d_Q_ball,          // [N, GQA_Q_HEADS * GQA_HEAD_DIM] output
    float *d_K_ball,          // [N, GQA_KV_HEADS * GQA_HEAD_DIM] output
    float *d_V_ball);         // [N, GQA_KV_HEADS * GQA_HEAD_DIM] output

// ============================================================
// exp_map backward: d_input from d_output
// exp_map(v): output[i] = tanh(||v||/R) * R/||v|| * v[i]
// ============================================================
static inline void exp_map_backward(const float *v, int d, float R,
                                    const float *d_output, float *d_input) {
    float nv = 0.0f;
    for (int i = 0; i < d; i++) nv += v[i] * v[i];
    nv = sqrtf(nv);
    if (nv < 1e-12f) {
        // exp_map(0) ≈ 0, gradient ≈ identity
        memcpy(d_input, d_output, d * sizeof(float));
        return;
    }
    // output[i] = g(n) * v[i], g(n) = tanh(n/R) * R/n
    float ratio = nv / R;
    if (ratio > 0.99f) ratio = 0.99f;
    float th = tanhf(ratio);
    float g = th * R / nv;

    // g'(n) = (sech²(n/R)*R/n - tanh(n/R)*R/n²) = (sech²(n/R) * n - th * R) * R / n² / R? Let me be careful.
    // g(n) = tanh(n/R) * R/n
    // g'(n) = sech²(n/R) * (1/R) * R/n + tanh(n/R) * (-R/n²) -- product rule
    // g'(n) = sech²(n/R)/n - tanh(n/R)*R/n²
    // g'(n) = (sech²(n/R)*n - tanh(n/R)*R) / n²
    float sech2 = 1.0f - th * th;  // sech² = 1 - tanh²
    float gp = (sech2 * nv - th * R) / (nv * nv);

    float dot = 0.0f;
    for (int i = 0; i < d; i++) dot += d_output[i] * v[i];

    float factor = gp / nv;
    for (int i = 0; i < d; i++) {
        d_input[i] = d_output[i] * g + factor * v[i] * dot;
    }
}

// ============================================================
// log_map backward: d_input from d_output
// log_map(x): output[i] = R * atanh(||x||/R) / ||x|| * x[i]
// ============================================================
static inline void log_map_backward(const float *x, int d, float R,
                                    const float *d_output, float *d_input) {
    float nx = 0.0f;
    for (int i = 0; i < d; i++) nx += x[i] * x[i];
    nx = sqrtf(nx);
    if (nx < 1e-12f) {
        // log_map(0) ≈ 0, gradient ≈ identity
        memcpy(d_input, d_output, d * sizeof(float));
        return;
    }

    // output[i] = f(n) * x[i], f(n) = R*atanh(n/R)/n
    float ratio = nx / R;
    if (ratio > 0.999f) ratio = 0.999f;
    float atanh_r = 0.5f * logf((1.0f + ratio) / (1.0f - ratio));
    float f = R * atanh_r / nx;

    // f'(n) = R * [n/(R²-n²) * n - atanh(n/R)] / n²? Let me re-derive:
    // f(n) = R * atanh(n/R) / n
    // Let a(n) = atanh(n/R)
    // a'(n) = R/(R²-n²)
    // f'(n) = R * [a'(n)*n - a(n)] / n² = [R²*n/(R²-n²) - R*atanh(n/R)] / n²

    float R2 = R * R;
    float denom = R2 - nx * nx;
    if (denom < 1e-12f) denom = 1e-12f;
    float fp_num = R2 * nx / denom - R * atanh_r;
    float fp = fp_num / (nx * nx);

    float dot = 0.0f;
    for (int i = 0; i < d; i++) dot += d_output[i] * x[i];

    float factor = fp / nx;
    for (int i = 0; i < d; i++) {
        d_input[i] = d_output[i] * f + factor * x[i] * dot;
    }
}

// ============================================================
// Möbius addition gradient: backprop through z = mobius_add(x, y)
// Given d_z (gradient w.r.t. output z), compute d_x and d_y.
//
// Uses the O(d) dot-product formulation:
//   z_i = (β * x_i + γ * y_i) / α
// where α, β, γ are scalar functions of x, y.
// ============================================================
static void mobius_add_gradient(const float *x, const float *y, int d, float R,
                                const float *z,         // output of mobius_add
                                const float *d_z,       // upstream gradient
                                float *d_x,             // output (add to existing)
                                float *d_y) {           // output (add to existing)
    float c = 1.0f / (R * R);

    // Compute dot products and squared norms
    float dot_xy = 0.0f, nx2 = 0.0f, ny2 = 0.0f;
    for (int i = 0; i < d; i++) {
        dot_xy += x[i] * y[i];
        nx2 += x[i] * x[i];
        ny2 += y[i] * y[i];
    }

    // Edge cases: if either input is near-zero, mobius_add ≈ identity
    if (nx2 < 1e-30f) {
        // x ≈ 0: z ≈ y, ∂z/∂y ≈ I, ∂z/∂x ≈ 0
        for (int i = 0; i < d; i++) d_y[i] += d_z[i];
        return;
    }
    if (ny2 < 1e-30f) {
        // y ≈ 0: z ≈ x, ∂z/∂x ≈ I, ∂z/∂y ≈ 0
        for (int i = 0; i < d; i++) d_x[i] += d_z[i];
        return;
    }

    // Scalar terms
    float cny2 = c * ny2;
    float cnx2 = c * nx2;
    float c2nx2ny2 = c * cnx2 * ny2;
    float two_c_dot = 2.0f * c * dot_xy;

    float alpha = 1.0f + two_c_dot + c2nx2ny2;
    if (fabsf(alpha) < 1e-30f) alpha = 1e-30f;  // safety clamp
    float beta = 1.0f + two_c_dot + cny2;
    float gamma = 1.0f - cnx2;

    // Compute dot products needed for O(d) backprop
    float S_xz = 0.0f, S_yz = 0.0f, S_zz = 0.0f;
    for (int i = 0; i < d; i++) {
        S_xz += d_z[i] * x[i];
        S_yz += d_z[i] * y[i];
        S_zz += d_z[i] * z[i];
    }

    float inv_alpha = 1.0f / alpha;
    float two_c_inv_alpha = 2.0f * c * inv_alpha;
    float two_c2_ny2_inv_alpha = 2.0f * c * c * ny2 * inv_alpha;

    // d_x_i = beta*d_z_i/alpha + 2c*y_i*S_xz/alpha - 2c*x_i*S_yz/alpha
    //       - (2c*y_i + 2c²||y||²*x_i)*S_zz/alpha
    for (int i = 0; i < d; i++) {
        float dx = beta * d_z[i] * inv_alpha
                 + two_c_inv_alpha * y[i] * S_xz
                 - two_c_inv_alpha * x[i] * S_yz
                 - (two_c_inv_alpha * y[i] + two_c2_ny2_inv_alpha * x[i]) * S_zz;
        d_x[i] += dx;
    }

    // d_y_i = gamma*d_z_i/alpha + 2c*(x_i+y_i)*S_xz/alpha
    //       - 2c*x_i*S_zz/alpha - 2c²||x||²*y_i*S_zz/alpha
    float two_c2_nx2_inv_alpha = 2.0f * c * c * nx2 * inv_alpha;
    for (int i = 0; i < d; i++) {
        float dy = gamma * d_z[i] * inv_alpha
                 + two_c_inv_alpha * (x[i] + y[i]) * S_xz
                 - two_c_inv_alpha * x[i] * S_zz
                 - two_c2_nx2_inv_alpha * y[i] * S_zz;
        d_y[i] += dy;
    }
}

// ============================================================
// Poincaré GQA: Hyperbolic Attention Backward
//
// Forward: for each (b,t_q,h_q):
//   q = Q_ball[b,t_q,h_q], k[t_k] = K_ball[b,t_k,h_kv]
//   dist[t_k] = R*atanh(||(-q)⊕k[t_k]||/R)
//   score[t_k] = -dist[t_k]/tau
//   w[t_k] = softmax(score)[t_k]
//   out_ball = exp_map(Σ w[t_k]*log_map(V_ball[t_k]))
//   out_vec = log_map(out_ball)
//
// Backward: use straight-through for log_map∘exp_map ≈ identity.
// For V_ball backward through log_map, use exact gradient.
// For distance backward, use exact Möbius add gradient.
// ============================================================
static void wubu_poincare_gqa_backward_attention(
    int B, int T, float R,
    const float *Q_ball,
    const float *K_ball,
    const float *V_ball,
    const float *d_attn_out,  // [N, GQA_Q_HEADS * GQA_HEAD_DIM] upstream grad
    float *d_Q_ball,
    float *d_K_ball,
    float *d_V_ball)
{
    const int hd = GQA_HEAD_DIM;   // 256
    const int n_q = GQA_Q_HEADS;   // 16
    const int n_kv = GQA_KV_HEADS; // 2
    const int q_per_kv = n_q / n_kv; // 8
    const float tau = 1.0f;

    // Pre-allocate log_mapped V for efficiency
    float *logV = (float *)malloc(T * n_kv * hd * sizeof(float));
    if (!logV) return;

    // Pre-compute log_map of all V_ball vectors
    for (int i = 0; i < T * n_kv; i++) {
        wubu_log_map(V_ball + i * hd, hd, R, logV + i * hd);
    }

    for (int b = 0; b < B; b++) {
        for (int t_q = 0; t_q < T; t_q++) {
            #pragma omp parallel for if(T > 32)
            for (int h_q = 0; h_q < n_q; h_q++) {
                int h_kv = h_q / q_per_kv;
                int max_t = t_q + 1;  // causal

                const float *q_vec = Q_ball + ((b*T + t_q) * n_q + h_q) * hd;
                const float *d_out = d_attn_out + ((b*T + t_q) * n_q + h_q) * hd;
                float *dq = d_Q_ball + ((b*T + t_q) * n_q + h_q) * hd;

                // Stack arrays for attention weights and distances
                float dists[4096], scores[4096];
                float neg_q[256], z_vec[256];

                // Step a: compute distances
                for (int t_k = 0; t_k < max_t; t_k++) {
                    const float *k_vec = K_ball + ((b*T + t_k) * n_kv + h_kv) * hd;
                    // z = (-q) ⊕ k
                    for (int i = 0; i < hd; i++) neg_q[i] = -q_vec[i];
                    wubu_mobius_add(neg_q, k_vec, hd, R, z_vec);
                    float nz = 0.0f;
                    for (int i = 0; i < hd; i++) nz += z_vec[i] * z_vec[i];
                    nz = sqrtf(nz);
                    float ratio = nz / R;
                    if (ratio >= 0.9999f) ratio = 0.9999f;
                    dists[t_k] = R * 0.5f * logf((1.0f + ratio) / (1.0f - ratio));
                    scores[t_k] = -dists[t_k] / tau;
                }

                // Step b-c: softmax
                float max_score = -1e30f;
                for (int t_k = 0; t_k < max_t; t_k++)
                    if (scores[t_k] > max_score) max_score = scores[t_k];

                float sum_exp = 0.0f;
                for (int t_k = 0; t_k < max_t; t_k++) {
                    scores[t_k] = expf(scores[t_k] - max_score);
                    sum_exp += scores[t_k];
                }
                float inv_sum = (sum_exp > 1e-30f) ? 1.0f / sum_exp : 0.0f;

                float attn_w[4096];
                for (int t_k = 0; t_k < max_t; t_k++)
                    attn_w[t_k] = scores[t_k] * inv_sum;

                // Backward starts here: d_out is gradient w.r.t. attn_out

                // === Step 1: Backprop through log_map(out_ball) ∘ exp_map(tangent_sum) ===
                // Forward: tangent_sum = Σ attn_w[t_k] * logV[t_k]
                //          out_ball = exp_map(tangent_sum)
                //          out_vec = log_map(out_ball)
                // Backward: d_tangent_sum = exp_map_backward(tangent_sum,
                //                                 log_map_backward(out_ball, d_out))
                float d_tangent_sum[256];
                {
                    float tangent_sum[256] = {0};
                    for (int t_k = 0; t_k < max_t; t_k++) {
                        const float *lv = logV + ((b*T + t_k) * n_kv + h_kv) * hd;
                        for (int i = 0; i < hd; i++)
                            tangent_sum[i] += attn_w[t_k] * lv[i];
                    }
                    float out_ball[256];
                    wubu_exp_map(tangent_sum, hd, R, out_ball);
                    float d_out_ball[256];
                    log_map_backward(out_ball, hd, R, d_out, d_out_ball);
                    exp_map_backward(tangent_sum, hd, R, d_out_ball, d_tangent_sum);
                }

                // Step 1a: d_attn_weights[t_k] = d_tangent_sum · log_map(V_ball[t_k])
                float d_attn_w[4096];
                float d_log_sum = 0.0f;
                for (int t_k = 0; t_k < max_t; t_k++) {
                    const float *lv = logV + ((b*T + t_k) * n_kv + h_kv) * hd;
                    double dot = 0.0;
                    for (int i = 0; i < hd; i++)
                        dot += (double)d_tangent_sum[i] * (double)lv[i];
                    d_attn_w[t_k] = (float)dot;
                    d_log_sum += d_attn_w[t_k] * attn_w[t_k];

                    // Also backprop to V_ball: d_V_ball from log_map backward
                    float *dv = d_V_ball + ((b*T + t_k) * n_kv + h_kv) * hd;
                    float d_log_v[256];
                    for (int i = 0; i < hd; i++)
                        d_log_v[i] = attn_w[t_k] * d_tangent_sum[i];

                    // Backprop through log_map(V_ball[t_k]) → d_V_ball[t_k] += d_log_v
                    log_map_backward(V_ball + ((b*T + t_k) * n_kv + h_kv) * hd,
                                     hd, R, d_log_v, dv);
                }

                // Step 2: Softmax backward
                // d_score[t_k] = w[t_k] * (d_w[t_k] - Σ d_w[j]*w[j])
                for (int t_k = 0; t_k < max_t; t_k++) {
                    float d_score = attn_w[t_k] * (d_attn_w[t_k] - d_log_sum);
                    float d_dist = -d_score / tau;

                    if (fabsf(d_dist) < 1e-15f) continue;

                    // Step 3: Backprop through distance
                    // d = R * atanh(||z||/R) where z = (-q) ⊕ k_vec
                    const float *k_vec = K_ball + ((b*T + t_k) * n_kv + h_kv) * hd;
                    float *dk = d_K_ball + ((b*T + t_k) * n_kv + h_kv) * hd;

                    // Recompute z = (-q) ⊕ k
                    for (int i = 0; i < hd; i++) neg_q[i] = -q_vec[i];
                    wubu_mobius_add(neg_q, k_vec, hd, R, z_vec);

                    // ∂dist/∂z_i = R² * z_i / (||z|| * (R² - ||z||²))
                    float nz = 0.0f;
                    for (int i = 0; i < hd; i++) nz += z_vec[i] * z_vec[i];
                    nz = sqrtf(nz);
                    if (nz < 1e-12f) continue;  // zero distance → no gradient

                    float R2 = R * R;
                    float denom = nz * (R2 - nz * nz);
                    if (denom < 1e-15f) denom = 1e-15f;
                    float d_z[256];
                    for (int i = 0; i < hd; i++)
                        d_z[i] = d_dist * R2 * z_vec[i] / denom;

                    // Backprop through mobius_add(x=neg_q, y=k)
                    // x = -q, y = k
                    // d_q = -d_x (since x = -q), d_k = d_y
                    float d_neg_q[256] = {0};
                    float d_k_temp[256] = {0};
                    mobius_add_gradient(neg_q, k_vec, hd, R, z_vec, d_z,
                                        d_neg_q, d_k_temp);

                    // d_q = -d_neg_q (since x = -q originally)
                    for (int i = 0; i < hd; i++) {
                        dq[i] -= d_neg_q[i];
                        dk[i] += d_k_temp[i];
                    }
                }
            }
        }
    }

    free(logV);
}

// ============================================================
// MatMul backward helper (copied from wubu_ssm.c — static there)
// d_input += d_output @ W^T   [N,Din] += [N,Dout] @ [Din,Dout]^T
// dW += input^T @ d_output     [Din,Dout] if dW non-NULL
// ============================================================
static void backward_matmul_nt(int N, int Din, int Dout,
                               const float *input, const float *d_output,
                               const float *W, float *d_input, float *dW) {
    for (int s = 0; s < N; s++) {
        for (int i = 0; i < Din; i++) {
            double sum = 0.0;
            for (int j = 0; j < Dout; j++)
                sum += (double)d_output[s * Dout + j] * (double)W[i * Dout + j];
            d_input[s * Din + i] += (float)sum;
        }
    }
    if (dW) {
        for (int i = 0; i < Din; i++) {
            for (int j = 0; j < Dout; j++) {
                double sum = 0.0;
                for (int s = 0; s < N; s++)
                    sum += (double)input[s * Din + i] * (double)d_output[s * Dout + j];
                dW[i * Dout + j] += (float)sum;
            }
        }
    }
}

// ============================================================
// Full Poincaré GQA Backward (matches interface of wubu_gqa_backward)
// ============================================================
void wubu_poincare_gqa_backward(
    int B, int T,
    const float *x,
    const float *Q_norm,
    const float *Q_raw,
    const float *K_norm,
    const float *K_raw,
    const float *V,
    const float *Q_ball,
    const float *K_ball,
    const float *V_ball,
    const float *gate,
    const float *gate_sig,
    const float *attn_out,
    const float *output,
    const float *d_output,
    const gqa_layer_weights *w,
    float R,
    float *d_x,
    float *d_q_weight,
    float *d_k_weight,
    float *d_v_weight,
    float *d_q_norm_weight,
    float *d_k_norm_weight,
    float *d_out_weight)
{
    const int N = B * T;
    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;   // 16*256 = 4096
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;  // 2*256 = 512

    // Allocate temp buffers
    float *d_attn_out = (float *)calloc(N * q_dim, sizeof(float));
    float *d_gate = (float *)calloc(N * q_dim, sizeof(float));
    float *d_Q_norm = (float *)calloc(N * q_dim, sizeof(float));
    float *d_K_norm = (float *)calloc(N * kv_dim, sizeof(float));
    float *d_V = (float *)calloc(N * kv_dim, sizeof(float));
    float *d_Q_raw = (float *)calloc(N * q_dim, sizeof(float));
    float *d_K_raw = (float *)calloc(N * kv_dim, sizeof(float));
    float *d_Q_full = (float *)calloc(N * q_dim * 2, sizeof(float));
    float *d_Q_ball = (float *)calloc(N * q_dim, sizeof(float));
    float *d_K_ball = (float *)calloc(N * kv_dim, sizeof(float));
    float *d_V_ball = (float *)calloc(N * kv_dim, sizeof(float));

    if (!d_attn_out || !d_gate || !d_Q_norm || !d_K_norm || !d_V ||
        !d_Q_raw || !d_K_raw || !d_Q_full || !d_Q_ball || !d_K_ball || !d_V_ball) {
        fprintf(stderr, "Poincare GQA backward: alloc failed\n");
        goto cleanup;
    }

    // === Step 7: Output projection backward ===
    wubu_ssm_backward_output_proj(attn_out, d_output, w->attn_output_weight,
                                   w->attn_output_weight_q, w->attn_output_weight_type,
                                   d_attn_out, d_out_weight, N);

    // === Step 6: Gate backward ===
    // Forward: attn_out_post = attn_out_pre * sigmoid(gate)
    for (int i = 0; i < N * q_dim; i++) {
        float sig = gate_sig[i];
        float sig_grad = sig * (1.0f - sig);
        float attn_pre = (sig > 1e-7f) ? attn_out[i] / sig : 0.0f;
        d_gate[i] += d_attn_out[i] * attn_pre * sig_grad;
        d_attn_out[i] *= sig;  // d_attn_pre = d_attn_post * sig
    }

    // === Step 5: Hyperbolic attention backward ===
    wubu_poincare_gqa_backward_attention(B, T, R,
                                         Q_ball, K_ball, V_ball,
                                         d_attn_out,
                                         d_Q_ball, d_K_ball, d_V_ball);

    // === Backprop V_ball → V (through exp_map) ===
    for (int i = 0; i < N * GQA_KV_HEADS; i++) {
        exp_map_backward(V + i * GQA_HEAD_DIM, GQA_HEAD_DIM, R,
                         d_V_ball + i * GQA_HEAD_DIM,
                         d_V + i * GQA_HEAD_DIM);
    }

    // === Backprop Q_ball → Q_norm (through exp_map) ===
    for (int i = 0; i < N * GQA_Q_HEADS; i++) {
        exp_map_backward(Q_norm + i * GQA_HEAD_DIM, GQA_HEAD_DIM, R,
                         d_Q_ball + i * GQA_HEAD_DIM,
                         d_Q_norm + i * GQA_HEAD_DIM);
    }

    // === Backprop K_ball → K_norm (through exp_map) ===
    for (int i = 0; i < N * GQA_KV_HEADS; i++) {
        exp_map_backward(K_norm + i * GQA_HEAD_DIM, GQA_HEAD_DIM, R,
                         d_K_ball + i * GQA_HEAD_DIM,
                         d_K_norm + i * GQA_HEAD_DIM);
    }

    // === Step 4: Q/K RMSNorm backward ===
    // Q RMSNorm
    for (int s = 0; s < N; s++) {
        for (int h = 0; h < GQA_Q_HEADS; h++) {
            const float *x_h = Q_raw + (s * GQA_Q_HEADS + h) * GQA_HEAD_DIM;
            const float *do_h = d_Q_norm + (s * GQA_Q_HEADS + h) * GQA_HEAD_DIM;
            float *dx_h = d_Q_raw + (s * GQA_Q_HEADS + h) * GQA_HEAD_DIM;

            double sum_sq = 0.0;
            for (int i = 0; i < GQA_HEAD_DIM; i++)
                sum_sq += (double)x_h[i] * (double)x_h[i];
            float rms = sqrtf((float)(sum_sq / GQA_HEAD_DIM) + 1e-6f);
            float r = 1.0f / rms;
            float r3 = r * r * r;

            double inner = 0.0;
            for (int j = 0; j < GQA_HEAD_DIM; j++)
                inner += (double)do_h[j] * (double)w->attn_q_norm_weight[j] * (double)x_h[j];

            for (int i = 0; i < GQA_HEAD_DIM; i++) {
                float grad = do_h[i] * w->attn_q_norm_weight[i] * r;
                grad -= (r3 / GQA_HEAD_DIM) * x_h[i] * (float)inner;
                dx_h[i] += grad;
            }
        }
    }
    // K RMSNorm
    for (int s = 0; s < N; s++) {
        for (int h = 0; h < GQA_KV_HEADS; h++) {
            const float *x_h = K_raw + (s * GQA_KV_HEADS + h) * GQA_HEAD_DIM;
            const float *do_h = d_K_norm + (s * GQA_KV_HEADS + h) * GQA_HEAD_DIM;
            float *dx_h = d_K_raw + (s * GQA_KV_HEADS + h) * GQA_HEAD_DIM;

            double sum_sq = 0.0;
            for (int i = 0; i < GQA_HEAD_DIM; i++)
                sum_sq += (double)x_h[i] * (double)x_h[i];
            float rms = sqrtf((float)(sum_sq / GQA_HEAD_DIM) + 1e-6f);
            float r = 1.0f / rms;
            float r3 = r * r * r;

            double inner = 0.0;
            for (int j = 0; j < GQA_HEAD_DIM; j++)
                inner += (double)do_h[j] * (double)w->attn_k_norm_weight[j] * (double)x_h[j];

            for (int i = 0; i < GQA_HEAD_DIM; i++) {
                float grad = do_h[i] * w->attn_k_norm_weight[i] * r;
                grad -= (r3 / GQA_HEAD_DIM) * x_h[i] * (float)inner;
                dx_h[i] += grad;
            }
        }
    }

    // === Step 3: Split backward ===
    // d_Q_full = [d_Q_raw | d_gate]
    for (int s = 0; s < N; s++) {
        memcpy(d_Q_full + s * q_dim * 2, d_Q_raw + s * q_dim, q_dim * sizeof(float));
        memcpy(d_Q_full + s * q_dim * 2 + q_dim, d_gate + s * q_dim, q_dim * sizeof(float));
    }

    // === Steps 1-3: MatMul backward for Q+gate, K, V ===
    backward_matmul_nt(N, D_MODEL, q_dim * 2, x, d_Q_full,
                       w->attn_q_weight, d_x, d_q_weight);

    backward_matmul_nt(N, D_MODEL, kv_dim, x, d_K_raw,
                       w->attn_k_weight, d_x, d_k_weight);

    backward_matmul_nt(N, D_MODEL, kv_dim, x, d_V,
                       w->attn_v_weight, d_x, d_v_weight);

cleanup:
    free(d_attn_out); free(d_gate);
    free(d_Q_norm); free(d_K_norm); free(d_V);
    free(d_Q_raw); free(d_K_raw); free(d_Q_full);
    free(d_Q_ball); free(d_K_ball); free(d_V_ball);
}

// ============================================================
// Forward Save: same as forward but saves intermediates
// ============================================================
void wubu_poincare_gqa_forward_save(const float *x, int B, int T,
                                     const gqa_layer_weights *weights,
                                     float R,
                                     float *output,
                                     poincare_gqa_fwd_save_t *save) {
    const int N = B * T;
    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;

    // Allocate buffers
    float *Q_full   = (float *)malloc(N * q_dim * sizeof(float));
    float *gate     = (float *)malloc(N * q_dim * sizeof(float));
    float *K        = (float *)malloc(N * kv_dim * sizeof(float));
    float *V        = (float *)malloc(N * kv_dim * sizeof(float));
    float *Q_norm   = (float *)malloc(N * q_dim * sizeof(float));
    float *K_norm   = (float *)malloc(N * kv_dim * sizeof(float));
    float *attn_out = (float *)malloc(N * q_dim * sizeof(float));

    if (!Q_full || !gate || !K || !V || !Q_norm || !K_norm || !attn_out) {
        fprintf(stderr, "Poincare GQA forward_save: alloc failed\n");
        free(Q_full); free(gate); free(K); free(V);
        free(Q_norm); free(K_norm); free(attn_out);
        return;
    }

    // Step 1: Q + gate fused projection
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        for (int j = 0; j < q_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)weights->attn_q_weight[i * (q_dim * 2) + j];
            Q_full[s * q_dim + j] = (float)sum;
        }
        for (int j = 0; j < q_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)weights->attn_q_weight[i * (q_dim * 2) + (j + q_dim)];
            gate[s * q_dim + j] = (float)sum;
        }
    }

    // Step 2: K and V projections
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        for (int j = 0; j < kv_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)weights->attn_k_weight[i * kv_dim + j];
            K[s * kv_dim + j] = (float)sum;
        }
        for (int j = 0; j < kv_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)weights->attn_v_weight[i * kv_dim + j];
            V[s * kv_dim + j] = (float)sum;
        }
    }

    // Step 3: Q/K RMSNorm — save Q_raw, K_raw (pre-norm copies)
    float *Q_only = (float *)malloc(N * q_dim * sizeof(float));
    if (!Q_only) {
        free(Q_full); free(gate); free(K); free(V);
        free(Q_norm); free(K_norm); free(attn_out);
        return;
    }
    memcpy(Q_only, Q_full, N * q_dim * sizeof(float));
    // Save Q_raw = Q_only copy for backward
    if (save->Q_raw) memcpy(save->Q_raw, Q_only, N * q_dim * sizeof(float));
    // Save K_raw
    if (save->K_raw) memcpy(save->K_raw, K, N * kv_dim * sizeof(float));

    wubu_rms_norm(B, T * GQA_Q_HEADS, GQA_HEAD_DIM,
                  Q_only, weights->attn_q_norm_weight, 1e-6f, Q_norm);
    wubu_rms_norm(B, T * GQA_KV_HEADS, GQA_HEAD_DIM,
                  K, weights->attn_k_norm_weight, 1e-6f, K_norm);

    // Save Q_norm, K_norm
    if (save->Q_norm) memcpy(save->Q_norm, Q_norm, N * q_dim * sizeof(float));
    if (save->K_norm) memcpy(save->K_norm, K_norm, N * kv_dim * sizeof(float));
    // Save V (raw, pre-exp_map)
    if (save->V) memcpy(save->V, V, N * kv_dim * sizeof(float));
    // Save gate (pre-sigmoid)
    if (save->gate) memcpy(save->gate, gate, N * q_dim * sizeof(float));

    free(Q_only);

    // Step 5: Ball-space computation
    float *Q_ball = (float *)malloc(N * GQA_Q_HEADS * GQA_HEAD_DIM * sizeof(float));
    float *K_ball = NULL;  // set below
    float *V_ball = NULL;  // set below

    if (!Q_ball) {
        free(Q_ball);
        free(Q_full); free(gate); free(K); free(V);
        free(Q_norm); free(K_norm); free(attn_out);
        return;
    }

    for (int i = 0; i < N * GQA_Q_HEADS; i++)
        wubu_exp_map(Q_norm + i * GQA_HEAD_DIM, GQA_HEAD_DIM, R,
                     Q_ball + i * GQA_HEAD_DIM);

    // Determine whether to use the hyperbolic KV cache
    int use_cache = (save->cache != NULL);
    poincare_kv_cache_t *cache_ptr = save->cache;  // saved for later use in attention
    float *K_ball_for_attn = NULL;
    float *V_ball_for_attn = NULL;
    float *K_ball_new = NULL;
    float *V_ball_new = NULL;
    float *K_ball_combined = NULL;
    float *V_ball_combined = NULL;
    int total_T = T;  // total timesteps visible to attention (per-batch)

    if (use_cache) {
        poincare_kv_cache_t *cache = save->cache;
        // Allocate buffers for the NEW tokens' ball-space K/V
        K_ball_new = (float *)malloc((int64_t)N * GQA_KV_HEADS * GQA_HEAD_DIM * sizeof(float));
        V_ball_new = (float *)malloc((int64_t)N * GQA_KV_HEADS * GQA_HEAD_DIM * sizeof(float));
        if (!K_ball_new || !V_ball_new) {
            free(K_ball_new); free(V_ball_new);
            free(Q_ball);
            free(Q_full); free(gate); free(K); free(V);
            free(Q_norm); free(K_norm); free(attn_out);
            return;
        }

        // Compute exp_map for the current batch's K_norm and V
        for (int i = 0; i < N * GQA_KV_HEADS; i++)
            wubu_exp_map(K_norm + i * GQA_HEAD_DIM, GQA_HEAD_DIM, R,
                         K_ball_new + i * GQA_HEAD_DIM);
        for (int i = 0; i < N * GQA_KV_HEADS; i++)
            wubu_exp_map(V + i * GQA_HEAD_DIM, GQA_HEAD_DIM, R,
                         V_ball_new + i * GQA_HEAD_DIM);

        // Build combined array: [cached K_ball | new K_ball]
        total_T = cache->current_T + T;
        int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
        int64_t combined_sz = (int64_t)total_T * kv_dim * (int64_t)sizeof(float);
        K_ball_combined = (float *)malloc((size_t)combined_sz);
        V_ball_combined = (float *)malloc((size_t)combined_sz);
        if (!K_ball_combined || !V_ball_combined) {
            free(K_ball_combined); free(V_ball_combined);
            free(K_ball_new); free(V_ball_new);
            free(Q_ball);
            free(Q_full); free(gate); free(K); free(V);
            free(Q_norm); free(K_norm); free(attn_out);
            return;
        }

        // Copy cached data into combined arrays
        int64_t cached_elems = (int64_t)cache->current_T * kv_dim;
        memcpy(K_ball_combined, cache->K_ball_cached, (size_t)(cached_elems * sizeof(float)));
        memcpy(V_ball_combined, cache->V_ball_cached, (size_t)(cached_elems * sizeof(float)));
        // Copy new data after cached
        int64_t new_elems = (int64_t)N * kv_dim;
        memcpy(K_ball_combined + cached_elems, K_ball_new, (size_t)(new_elems * sizeof(float)));
        memcpy(V_ball_combined + cached_elems, V_ball_new, (size_t)(new_elems * sizeof(float)));

        K_ball_for_attn = K_ball_combined;
        V_ball_for_attn = V_ball_combined;

        // Save only the NEW tokens' ball-space data for backward
        if (save->K_ball) memcpy(save->K_ball, K_ball_new, (size_t)(new_elems * sizeof(float)));
        if (save->V_ball) memcpy(save->V_ball, V_ball_new, (size_t)(new_elems * sizeof(float)));

        // Append to cache: grow if needed, then copy
        int needed_T = total_T;
        if (needed_T > cache->max_T) {
            int grow = cache->max_T > 0 ? cache->max_T : 1;
            while (needed_T > grow) grow *= 2;
            poincare_kv_cache_resize(cache, grow);
        }
        memcpy(cache->K_ball_cached + cached_elems, K_ball_new, (size_t)(new_elems * sizeof(float)));
        memcpy(cache->V_ball_cached + cached_elems, V_ball_new, (size_t)(new_elems * sizeof(float)));
        cache->current_T = total_T;

        // K_ball and V_ball for the rest of the function = the new buffers
        K_ball = K_ball_new;
        V_ball = V_ball_new;
    } else {
        // Original path: no cache
        K_ball = (float *)malloc((int64_t)N * GQA_KV_HEADS * GQA_HEAD_DIM * sizeof(float));
        V_ball = (float *)malloc((int64_t)N * GQA_KV_HEADS * GQA_HEAD_DIM * sizeof(float));
        if (!K_ball || !V_ball) {
            free(K_ball); free(V_ball);
            free(Q_ball);
            free(Q_full); free(gate); free(K); free(V);
            free(Q_norm); free(K_norm); free(attn_out);
            return;
        }

        for (int i = 0; i < N * GQA_KV_HEADS; i++)
            wubu_exp_map(K_norm + i * GQA_HEAD_DIM, GQA_HEAD_DIM, R,
                         K_ball + i * GQA_HEAD_DIM);
        for (int i = 0; i < N * GQA_KV_HEADS; i++)
            wubu_exp_map(V + i * GQA_HEAD_DIM, GQA_HEAD_DIM, R,
                         V_ball + i * GQA_HEAD_DIM);

        K_ball_for_attn = K_ball;
        V_ball_for_attn = V_ball;

        // Save ball-space vectors for backward
        if (save->Q_ball) memcpy(save->Q_ball, Q_ball, (int64_t)N * GQA_Q_HEADS * GQA_HEAD_DIM * sizeof(float));
        if (save->K_ball) memcpy(save->K_ball, K_ball, (int64_t)N * GQA_KV_HEADS * GQA_HEAD_DIM * sizeof(float));
        if (save->V_ball) memcpy(save->V_ball, V_ball, (int64_t)N * GQA_KV_HEADS * GQA_HEAD_DIM * sizeof(float));
    }

    // Hyperbolic attention
    const float tau = 1.0f;
    // When cache is used, the combined arrays have total_T tokens;
    // each batch's effective max_t position differs.
    for (int b = 0; b < B; b++) {
        for (int t_q = 0; t_q < T; t_q++) {
            #pragma omp parallel for if(T > 32 || (B > 1 && T > 4))
            for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
                int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);
                const float *q_ball = Q_ball +
                    ((b * T + t_q) * GQA_Q_HEADS + h_q) * GQA_HEAD_DIM;
                float *out_vec = attn_out +
                    ((b * T + t_q) * GQA_Q_HEADS + h_q) * GQA_HEAD_DIM;

                float attn_weights[4096];
                float max_score = -1e30f;

                if (use_cache) {
                    // Combined array: [cached(0..cache_ptr->current_T-1) | new(b*T..b*T+T-1)]
                    // Query at effective position q_eff = cache_ptr->current_T + b*T + t_q
                    // Attends to all tokens [0 .. q_eff] (causal)
                    int q_eff = cache_ptr->current_T + b * T + t_q;
                    for (int t_k = 0; t_k <= q_eff; t_k++) {
                        const float *k_ball = K_ball_for_attn +
                            (t_k * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                        float dist = wubu_poincare_dist(q_ball, k_ball, GQA_HEAD_DIM, R);
                        float score = -dist / tau;
                        attn_weights[t_k] = score;
                        if (score > max_score) max_score = score;
                    }

                    float sum_exp = 0.0f;
                    for (int t_k = 0; t_k <= q_eff; t_k++) {
                        attn_weights[t_k] = expf(attn_weights[t_k] - max_score);
                        sum_exp += attn_weights[t_k];
                    }
                    if (sum_exp > 1e-30f) {
                        float inv_sum = 1.0f / sum_exp;
                        for (int t_k = 0; t_k <= q_eff; t_k++)
                            attn_weights[t_k] *= inv_sum;
                    } else {
                        float inv_n = 1.0f / (q_eff + 1);
                        for (int t_k = 0; t_k <= q_eff; t_k++)
                            attn_weights[t_k] = inv_n;
                    }

                    const float *v_ptrs[4096];
                    for (int t_k = 0; t_k <= q_eff; t_k++) {
                        v_ptrs[t_k] = V_ball_for_attn +
                            (t_k * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                    }

                    float out_ball[GQA_HEAD_DIM];
                    wubu_poincare_linear_comb(v_ptrs, attn_weights, q_eff + 1,
                                               GQA_HEAD_DIM, R, out_ball);
                    wubu_log_map(out_ball, GQA_HEAD_DIM, R, out_vec);
                } else {
                    // Original path: no cache, standard b*T + t_k indexing
                    for (int t_k = 0; t_k <= t_q; t_k++) {
                        const float *k_ball = K_ball +
                            ((b * T + t_k) * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                        float dist = wubu_poincare_dist(q_ball, k_ball, GQA_HEAD_DIM, R);
                        float score = -dist / tau;
                        attn_weights[t_k] = score;
                        if (score > max_score) max_score = score;
                    }

                    float sum_exp = 0.0f;
                    for (int t_k = 0; t_k <= t_q; t_k++) {
                        attn_weights[t_k] = expf(attn_weights[t_k] - max_score);
                        sum_exp += attn_weights[t_k];
                    }
                    if (sum_exp > 1e-30f) {
                        float inv_sum = 1.0f / sum_exp;
                        for (int t_k = 0; t_k <= t_q; t_k++)
                            attn_weights[t_k] *= inv_sum;
                    } else {
                        float inv_n = 1.0f / (t_q + 1);
                        for (int t_k = 0; t_k <= t_q; t_k++)
                            attn_weights[t_k] = inv_n;
                    }

                    const float *v_ptrs[4096];
                    for (int t_k = 0; t_k <= t_q; t_k++) {
                        v_ptrs[t_k] = V_ball +
                            ((b * T + t_k) * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                    }

                    float out_ball[GQA_HEAD_DIM];
                    wubu_poincare_linear_comb(v_ptrs, attn_weights, t_q + 1,
                                               GQA_HEAD_DIM, R, out_ball);
                    wubu_log_map(out_ball, GQA_HEAD_DIM, R, out_vec);
                }
            }
        }
    }

    // Save attn_out_pre_gate
    if (save->attn_out_pre_gate) memcpy(save->attn_out_pre_gate, attn_out, N * q_dim * sizeof(float));

    free(Q_ball);
    free(K_ball);
    free(V_ball);
    free(K_ball_combined);
    free(V_ball_combined);

    // Step 6: Gate (sigmoid)
    float *gate_sig = (float *)malloc(N * q_dim * sizeof(float));
    if (!gate_sig) {
        free(Q_full); free(gate); free(K); free(V);
        free(Q_norm); free(K_norm); free(attn_out);
        return;
    }
    wubu_sigmoid(N * q_dim, gate, gate_sig);
    if (save->gate_sig) memcpy(save->gate_sig, gate_sig, N * q_dim * sizeof(float));

    for (int i = 0; i < N * q_dim; i++) {
        attn_out[i] *= gate_sig[i];
    }

    // Step 7: Output projection
    for (int s = 0; s < N; s++) {
        const float *inp = attn_out + s * q_dim;
        float *out = output + s * D_MODEL;
        for (int j = 0; j < D_MODEL; j++) {
            double sum = 0.0;
            for (int i = 0; i < q_dim; i++)
                sum += (double)inp[i] * (double)weights->attn_output_weight[i * D_MODEL + j];
            out[j] = (float)sum;
        }
    }

    free(Q_full);
    free(gate);
    free(K);
    free(V);
    free(Q_norm);
    free(K_norm);
    free(attn_out);
    free(gate_sig);
}

// ============================================================
// Hyperbolic KV Cache: init / resize / free / append
// ============================================================
void poincare_kv_cache_init(poincare_kv_cache_t *cache, int init_capacity) {
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
    cache->max_T = init_capacity > 0 ? init_capacity : 1;
    cache->current_T = 0;
    cache->K_ball_cached = (float *)calloc((int64_t)cache->max_T * kv_dim, sizeof(float));
    cache->V_ball_cached = (float *)calloc((int64_t)cache->max_T * kv_dim, sizeof(float));
    if (!cache->K_ball_cached || !cache->V_ball_cached) {
        fprintf(stderr, "poincare_kv_cache_init: allocation failed\n");
        free(cache->K_ball_cached); cache->K_ball_cached = NULL;
        free(cache->V_ball_cached); cache->V_ball_cached = NULL;
        cache->max_T = 0;
        cache->current_T = 0;
    }
}

void poincare_kv_cache_resize(poincare_kv_cache_t *cache, int new_capacity) {
    if (new_capacity <= cache->max_T) return;
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
    int64_t old_bytes = (int64_t)cache->max_T * kv_dim * sizeof(float);
    int64_t new_bytes = (int64_t)new_capacity * kv_dim * sizeof(float);

    float *new_K = (float *)realloc(cache->K_ball_cached, (size_t)new_bytes);
    float *new_V = (float *)realloc(cache->V_ball_cached, (size_t)new_bytes);
    if (!new_K || !new_V) {
        fprintf(stderr, "poincare_kv_cache_resize: realloc failed\n");
        if (new_K) cache->K_ball_cached = new_K;
        if (new_V) cache->V_ball_cached = new_V;
        return;
    }
    cache->K_ball_cached = new_K;
    cache->V_ball_cached = new_V;
    // Zero the newly allocated portion
    memset(cache->K_ball_cached + old_bytes / sizeof(float), 0,
           (size_t)(new_bytes - old_bytes));
    memset(cache->V_ball_cached + old_bytes / sizeof(float), 0,
           (size_t)(new_bytes - old_bytes));
    cache->max_T = new_capacity;
}

void poincare_kv_cache_free(poincare_kv_cache_t *cache) {
    free(cache->K_ball_cached);
    free(cache->V_ball_cached);
    cache->K_ball_cached = NULL;
    cache->V_ball_cached = NULL;
    cache->max_T = 0;
    cache->current_T = 0;
}
