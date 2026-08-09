#include "wubu_poincare_gqa.h"
#include "wubu_mobius.h"
#include "gguf_reader.h"
#include "safetensors_reader.h"
#include "wubu_rotate.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Dequant a [rows, cols] BF16/F16 matrix into F32 [cols, rows] (transposed),
 * matching the forward's [IN, OUT] column-major layout. */
static void gqa_dequant_t(const uint8_t *raw, int rows, int cols, int dtype, float *out) {
    const uint16_t *b = (const uint16_t *)raw;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            float v = (dtype == ST_DTYPE_F16) ? st_f16_to_f32(b[r*cols + c])
                                              : st_bf16_to_f32(b[r*cols + c]);
            out[(size_t)c * rows + r] = v;
        }
}

/* Materialize quantized GQA proj matrices to F32 (once). The engine's
 * forward uses the raw quantized blob (attn_*_weight_q) directly; this
 * helper exists for tooling that needs plain F32 buffers. Dequant layout
 * matches quantized_matmul_batched's [D_MODEL, proj_dim] convention. */
void wubu_gqa_ensure_f32(gqa_layer_weights *w, int d_model) {
    if (!w) return;
    int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
    int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;

    /* KB6 (doc 013 QuaRot): fixed Hadamard fuse on weight columns
     * before quantization. One-time O(M·K) cost at materialization;
     * amortized over all subsequent decode steps. Gated by
     * WUBU_HADAMARD_FUSE=1. No accuracy impact (H is orthonormal). */
    int do_rotate = getenv("WUBU_HADAMARD_FUSE") != NULL;

    if (!w->attn_q_weight && w->attn_q_weight_q) {
        int64_t n = (int64_t)d_model * q_dim * 2;
        w->attn_q_weight = (float *)malloc((size_t)n * sizeof(float));
        if (w->attn_q_weight) gguf_dequantize(w->attn_q_weight_q, w->attn_q_weight_type, n, w->attn_q_weight);
        if (do_rotate && w->attn_q_weight) {
            wubu_rotate_fuse_right(w->attn_q_weight, q_dim * 2, d_model);
            wubu_rotate_fuse_right(w->attn_q_weight + d_model, q_dim * 2, d_model);
        }
    }
    if (!w->attn_k_weight && w->attn_k_weight_q) {
        int64_t n = (int64_t)d_model * kv_dim;
        w->attn_k_weight = (float *)malloc((size_t)n * sizeof(float));
        if (w->attn_k_weight) gguf_dequantize(w->attn_k_weight_q, w->attn_k_weight_type, n, w->attn_k_weight);
        if (do_rotate && w->attn_k_weight) wubu_rotate_fuse_right(w->attn_k_weight, kv_dim, d_model);
    }
    if (!w->attn_v_weight && w->attn_v_weight_q) {
        int64_t n = (int64_t)d_model * kv_dim;
        w->attn_v_weight = (float *)malloc((size_t)n * sizeof(float));
        if (w->attn_v_weight) gguf_dequantize(w->attn_v_weight_q, w->attn_v_weight_type, n, w->attn_v_weight);
        if (do_rotate && w->attn_v_weight) wubu_rotate_fuse_right(w->attn_v_weight, kv_dim, d_model);
    }
    if (!w->attn_output_weight && w->attn_output_weight_q) {
        int64_t n = (int64_t)q_dim * d_model;
        w->attn_output_weight = (float *)malloc((size_t)n * sizeof(float));
        if (w->attn_output_weight) gguf_dequantize(w->attn_output_weight_q, w->attn_output_weight_type, n, w->attn_output_weight);
        if (do_rotate && w->attn_output_weight) wubu_rotate_fuse_right(w->attn_output_weight, d_model, q_dim);
    }
}

/* Inverse of wubu_gqa_ensure_f32: free materialized F32 proj matrices.
 * Only frees buffers that were materialized from quantized blobs; layers
 * loaded as resident F32 keep their buffer (the _q pointer is NULL then). */
void wubu_gqa_release_f32(gqa_layer_weights *w) {
    if (!w) return;
    if (w->attn_q_weight_q)     { free(w->attn_q_weight);     w->attn_q_weight = NULL; }
    if (w->attn_k_weight_q)     { free(w->attn_k_weight);     w->attn_k_weight = NULL; }
    if (w->attn_v_weight_q)     { free(w->attn_v_weight);     w->attn_v_weight = NULL; }
    if (w->attn_output_weight_q) { free(w->attn_output_weight); w->attn_output_weight = NULL; }
}

// ============================================================
// Poincaré GQA Forward Pass
//
// Architecture matches wubu_gqa_forward() except Step 5
// (attention) uses hyperbolic distance in the Poincaré ball.
// ============================================================
void wubu_poincare_gqa_forward(const float *x, int B, int T,
                                const gqa_layer_weights *w,
                                float R,
                                float *output) {
    const int N = B * T;
    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;   // 16*256 = 4096
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;  // 2*256 = 512

    // ========== Allocate buffers ==========
    float *Q_full   = (float *)malloc(N * q_dim * sizeof(float));
    float *gate     = (float *)malloc(N * q_dim * sizeof(float));
    float *K        = (float *)malloc(N * kv_dim * sizeof(float));
    float *V        = (float *)malloc(N * kv_dim * sizeof(float));
    float *Q_norm   = (float *)malloc(N * q_dim * sizeof(float));
    float *K_norm   = (float *)malloc(N * kv_dim * sizeof(float));
    float *attn_out = (float *)malloc(N * q_dim * sizeof(float));

    if (!Q_full || !gate || !K || !V || !Q_norm || !K_norm || !attn_out) {
        fprintf(stderr, "Poincare GQA: allocation failed (step 1)\n");
        free(Q_full); free(gate); free(K); free(V);
        free(Q_norm); free(K_norm); free(attn_out);
        return;
    }

    // ========== Step 1: Q + gate fused projection ==========
    // w->attn_q_weight shape: [D_MODEL, q_dim*2] = [2048, 8192]
    // First q_dim columns = Q, next q_dim columns = gate
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        // Q projection (first half)
        for (int j = 0; j < q_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)w->attn_q_weight[i * (q_dim * 2) + j];
            Q_full[s * q_dim + j] = (float)sum;
        }
        // Gate projection (second half)
        for (int j = 0; j < q_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)w->attn_q_weight[i * (q_dim * 2) + (j + q_dim)];
            gate[s * q_dim + j] = (float)sum;
        }
    }

    // ========== Step 2: K and V projections ==========
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        // K projection: [D_MODEL, kv_dim] = [2048, 512]
        for (int j = 0; j < kv_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)w->attn_k_weight[i * kv_dim + j];
            K[s * kv_dim + j] = (float)sum;
        }
        // V projection: [D_MODEL, kv_dim] = [2048, 512]
        for (int j = 0; j < kv_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_s[i] * (double)w->attn_v_weight[i * kv_dim + j];
            V[s * kv_dim + j] = (float)sum;
        }
    }

    // ========== Step 3: Q/K RMSNorm ==========
    // Q is stored in first q_dim entries of Q_full (gate is separate)
    float *Q_only = (float *)malloc(N * q_dim * sizeof(float));
    if (!Q_only) {
        fprintf(stderr, "Poincare GQA: allocation failed (Q_only)\n");
        free(Q_full); free(gate); free(K); free(V);
        free(Q_norm); free(K_norm); free(attn_out);
        return;
    }
    memcpy(Q_only, Q_full, N * q_dim * sizeof(float));
    wubu_rms_norm(B, T * GQA_Q_HEADS, GQA_HEAD_DIM,
                  Q_only, w->attn_q_norm_weight, 1e-6f, Q_norm);
    wubu_rms_norm(B, T * GQA_KV_HEADS, GQA_HEAD_DIM,
                  K, w->attn_k_norm_weight, 1e-6f, K_norm);
    free(Q_only);

    // ========== Step 5: Poincaré Distance Attention ==========
    //
    // Instead of softmax(Q·K^T / sqrt(d)), we:
    //   1. Map Q/K to Poincaré ball via exp_map
    //   2. Score = softmax(-d(q_i, k_j) / tau)
    //      where d(·,·) = R * artanh(||(-x)⊕y|| / R)
    //   3. Map V to Poincaré ball via exp_map
    //   4. Output_ball = Möbius combination of V_ball weighted by scores
    //   5. Output = log_map(output_ball, R)

    // Allocate ball-space buffers
    // Q_ball, K_ball, V_ball: each (batch, timestep, head, head_dim)
    float *Q_ball = (float *)malloc(N * GQA_Q_HEADS * GQA_HEAD_DIM * sizeof(float));
    float *K_ball = (float *)malloc(N * GQA_KV_HEADS * GQA_HEAD_DIM * sizeof(float));
    float *V_ball = (float *)malloc(N * GQA_KV_HEADS * GQA_HEAD_DIM * sizeof(float));

    if (!Q_ball || !K_ball || !V_ball) {
        fprintf(stderr, "Poincare GQA: allocation failed (ball buffers)\n");
        free(Q_ball); free(K_ball); free(V_ball);
        free(Q_full); free(gate); free(K); free(V);
        free(Q_norm); free(K_norm); free(attn_out);
        return;
    }

    // Map Q_norm (after RMSNorm) to Poincaré ball
    // Layout: Q_norm[(b*T + t) * GQA_Q_HEADS + h, :]
    for (int i = 0; i < N * GQA_Q_HEADS; i++) {
        wubu_exp_map(Q_norm + i * GQA_HEAD_DIM, GQA_HEAD_DIM, R,
                     Q_ball + i * GQA_HEAD_DIM);
    }

    // Map K_norm to Poincaré ball
    // Layout: K_norm[(b*T + t) * GQA_KV_HEADS + h_kv, :]
    for (int i = 0; i < N * GQA_KV_HEADS; i++) {
        wubu_exp_map(K_norm + i * GQA_HEAD_DIM, GQA_HEAD_DIM, R,
                     K_ball + i * GQA_HEAD_DIM);
    }

    // Map V to Poincaré ball
    for (int i = 0; i < N * GQA_KV_HEADS; i++) {
        wubu_exp_map(V + i * GQA_HEAD_DIM, GQA_HEAD_DIM, R,
                     V_ball + i * GQA_HEAD_DIM);
    }

    // Hyperbolic attention temperature
    const float tau = 1.0f;

    // Precise loops for attention — use OMP on h_q dimension only
    // to keep the logic simple
    for (int b = 0; b < B; b++) {
        for (int t_q = 0; t_q < T; t_q++) {
            #pragma omp parallel for if(T > 32 || (B > 1 && T > 4))
            for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
                // Which KV head serves this Q head (GQA: 16 Q heads, 2 KV heads)
                int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS); // h_q / 8

                // Pointers to Q_ball and output for this (b, t_q, h_q)
                const float *q_ball = Q_ball +
                    ((b * T + t_q) * GQA_Q_HEADS + h_q) * GQA_HEAD_DIM;
                float *out_vec = attn_out +
                    ((b * T + t_q) * GQA_Q_HEADS + h_q) * GQA_HEAD_DIM;

                // Attention weights for this query (max T = up to 4096)
                float attn_weights[4096];
                float max_score = -1e30f;

                // Compute hyperbolic attention scores over all previous timesteps
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    const float *k_ball = K_ball +
                        ((b * T + t_k) * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;

                    // Poincaré geodesic distance: d(q,k) = R * artanh(||(-q)⊕k||/R)
                    float dist = wubu_poincare_dist(q_ball, k_ball,
                                                     GQA_HEAD_DIM, R);

                    // Score = -distance / tau (closer = higher score)
                    float score = -dist / tau;
                    attn_weights[t_k] = score;
                    if (score > max_score) max_score = score;
                }

                // Softmax over attention weights
                float sum_exp = 0.0f;
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    attn_weights[t_k] = expf(attn_weights[t_k] - max_score);
                    sum_exp += attn_weights[t_k];
                }
                // Normalize
                if (sum_exp > 1e-30f) {
                    float inv_sum = 1.0f / sum_exp;
                    for (int t_k = 0; t_k <= t_q; t_k++)
                        attn_weights[t_k] *= inv_sum;
                } else {
                    // Uniform fallback
                    float inv_n = 1.0f / (t_q + 1);
                    for (int t_k = 0; t_k <= t_q; t_k++)
                        attn_weights[t_k] = inv_n;
                }

                // Möbius combination: weighted hyperbolic average of V_ball
                // Build array of pointers to V_ball vectors
                const float *v_ptrs[4096];
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    v_ptrs[t_k] = V_ball +
                        ((b * T + t_k) * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                }

                // Tangent-space linear combination (Poincaré I):
                //   out_ball = exp_map(sum w_i * log_map(v_ball_i))
                float out_ball[GQA_HEAD_DIM];
                wubu_poincare_linear_comb(v_ptrs, attn_weights, t_q + 1,
                                           GQA_HEAD_DIM, R, out_ball);

                // Map back from ball to Euclidean tangent space
                wubu_log_map(out_ball, GQA_HEAD_DIM, R, out_vec);
            }
        }
    }

    // Free ball-space buffers
    free(Q_ball);
    free(K_ball);
    free(V_ball);

    // ========== Step 6: Gate (sigmoid) ==========
    float *gate_sig = (float *)malloc(N * q_dim * sizeof(float));
    if (!gate_sig) {
        fprintf(stderr, "Poincare GQA: allocation failed (gate_sig)\n");
        free(Q_full); free(gate); free(K); free(V);
        free(Q_norm); free(K_norm); free(attn_out);
        return;
    }
    wubu_sigmoid(N * q_dim, gate, gate_sig);

    // Gate the attention output: attn_out *= sigmoid(gate)
    for (int i = 0; i < N * q_dim; i++) {
        attn_out[i] *= gate_sig[i];
    }

    // ========== Step 7: Output projection ==========
    // attn_output_weight shape: [q_dim, D_MODEL] = [4096, 2048]
    // output[s,j] = sum_i attn_out[s,i] * W[i,j]
    for (int s = 0; s < N; s++) {
        const float *inp = attn_out + s * q_dim;
        float *out = output + s * D_MODEL;
        for (int j = 0; j < D_MODEL; j++) {
            double sum = 0.0;
            for (int i = 0; i < q_dim; i++)
                sum += (double)inp[i] * (double)w->attn_output_weight[i * D_MODEL + j];
            out[j] = (float)sum;
        }
    }

    // ========== Cleanup ==========
    free(Q_full);
    free(gate);
    free(K);
    free(V);
    free(Q_norm);
    free(K_norm);
    free(attn_out);
    free(gate_sig);
}
