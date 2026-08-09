/**
 * wubu_moe_backward.c — MoE backward pass (CPU)
 *
 * Full backprop through MoE: shared expert + top-k routed experts.
 * Handles NULL expert weight pointers gracefully (skips that section).
 */
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#if !defined(_WIN32)
#include <sys/resource.h>
#include <sys/prctl.h>
#endif

static inline float silu_f(float x) {
    if (x < -80.0f) return 0.0f;
    return x / (1.0f + expf(-x));
}

static inline float silu_deriv(float x, float silu_x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return silu_x + sig * (1.0f - silu_x);
}

// Recompute router: logits -> softmax -> top-k with normalized weights
static void moe_router_backward_prep(const float *x, int B, int T,
                                     const float *gate_inp,
                                     float *softmax_out,
                                     int *topk_indices,
                                     float *topk_weights)
{
    int N = B * T;
    float *logits = (float *)malloc(N * N_EXPERTS * sizeof(float));
    if (!logits) return;

    wubu_moe_router(x, B, T, gate_inp, logits);

    for (int s = 0; s < N; s++) {
        float *logit_s = logits + s * N_EXPERTS;
        float *score_s = softmax_out + s * N_EXPERTS;
        float max_s = logit_s[0];
        for (int e = 1; e < N_EXPERTS; e++)
            if (logit_s[e] > max_s) max_s = logit_s[e];
        float sum_exp = 0.0f;
        for (int e = 0; e < N_EXPERTS; e++)
            sum_exp += expf(logit_s[e] - max_s);
        float inv_sum = 1.0f / (sum_exp + 1e-30f);
        for (int e = 0; e < N_EXPERTS; e++)
            score_s[e] = expf(logit_s[e] - max_s) * inv_sum;

        int *ind_s = topk_indices + s * N_ACTIVE_EXPTS;
        float *wt_s = topk_weights + s * N_ACTIVE_EXPTS;
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            int best = -1;
            float best_v = -1e30f;
            for (int e = 0; e < N_EXPERTS; e++) {
                bool used = false;
                for (int pk = 0; pk < k; pk++)
                    if (ind_s[pk] == e) { used = true; break; }
                if (!used && score_s[e] > best_v) { best_v = score_s[e]; best = e; }
            }
            ind_s[k] = best;
            wt_s[k] = best_v;
        }
        float sum_w = 0.0f;
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) sum_w += wt_s[k];
        if (sum_w > 1e-30f) {
            float inv = 1.0f / sum_w;
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) wt_s[k] *= inv;
        }
    }
    free(logits);
}

// Expert backward for one token. All weight pointers must be non-NULL.
static void moe_expert_backward(
    const float *x_s, const float *gate_w, const float *up_w, const float *down_w,
    const float *d_expert_out, float *temp,
    float *d_gate_w, float *d_up_w, float *d_down_w, float *d_x,
    int d_model, int d_ff)
{
    float *gate_out = temp;
    float *up_out = temp + d_ff;
    float *act = temp + 2 * d_ff;

    for (int j = 0; j < d_ff; j++) {
        double sum = 0.0;
        for (int p = 0; p < d_model; p++)
            sum += (double)x_s[p] * (double)gate_w[p * d_ff + j];
        gate_out[j] = (float)sum;
    }
    for (int j = 0; j < d_ff; j++) {
        double sum = 0.0;
        for (int p = 0; p < d_model; p++)
            sum += (double)x_s[p] * (double)up_w[p * d_ff + j];
        up_out[j] = (float)sum;
    }
    for (int j = 0; j < d_ff; j++)
        act[j] = silu_f(gate_out[j]) * up_out[j];

    float d_act[d_ff];
    for (int p = 0; p < d_ff; p++) {
        double sum = 0.0;
        for (int j = 0; j < d_model; j++)
            sum += (double)d_expert_out[j] * (double)down_w[p * d_model + j];
        d_act[p] = (float)sum;
    }
    if (d_down_w) {
        for (int p = 0; p < d_ff; p++)
            for (int j = 0; j < d_model; j++)
                d_down_w[p * d_model + j] += d_expert_out[j] * act[p];
    }
    for (int j = 0; j < d_ff; j++) {
        float g = gate_out[j];
        float sg = silu_f(g);
        float dsg = silu_deriv(g, sg);
        float d_up = d_act[j] * sg;
        float d_gate_v = d_act[j] * dsg * up_out[j];
        if (d_gate_w) {
            for (int p = 0; p < d_model; p++)
                d_gate_w[p * d_ff + j] += d_gate_v * x_s[p];
        }
        if (d_up_w) {
            for (int p = 0; p < d_model; p++)
                d_up_w[p * d_ff + j] += d_up * x_s[p];
        }
        for (int p = 0; p < d_model; p++) {
            d_x[p] += d_gate_v * gate_w[p * d_ff + j]
                    + d_up * up_w[p * d_ff + j];
        }
    }
}

// Shared expert backward
static void moe_shared_backward(
    const float *x_s, const float *d_out_s,
    const moe_weights_t *w,
    float *d_gate_shexp, float *d_up_shexp, float *d_down_shexp,
    float *d_x_s,
    float *temp_shared, float *d_shared_act_buf,
    int d_model, int d_ff_shared)
{
    if (!w->ffn_gate_shexp || !w->ffn_up_shexp || !w->ffn_down_shexp) return;

    float *s_gate = temp_shared;
    float *s_up = temp_shared + d_ff_shared;
    float *s_act = temp_shared + 2 * d_ff_shared;

    for (int j = 0; j < d_ff_shared; j++) {
        double sum = 0.0;
        for (int k = 0; k < d_model; k++)
            sum += (double)x_s[k] * (double)w->ffn_gate_shexp[k * d_ff_shared + j];
        s_gate[j] = (float)sum;
    }
    for (int j = 0; j < d_ff_shared; j++) {
        double sum = 0.0;
        for (int k = 0; k < d_model; k++)
            sum += (double)x_s[k] * (double)w->ffn_up_shexp[k * d_ff_shared + j];
        s_up[j] = (float)sum;
    }
    for (int j = 0; j < d_ff_shared; j++)
        s_act[j] = silu_f(s_gate[j]) * s_up[j];

    for (int k = 0; k < d_ff_shared; k++) {
        double sum = 0.0;
        for (int j = 0; j < d_model; j++)
            sum += (double)d_out_s[j] * (double)w->ffn_down_shexp[k * d_model + j];
        d_shared_act_buf[k] = (float)sum;
    }
    if (d_down_shexp) {
        for (int k = 0; k < d_ff_shared; k++)
            for (int j = 0; j < d_model; j++)
                d_down_shexp[k * d_model + j] += d_out_s[j] * s_act[k];
    }
    for (int j = 0; j < d_ff_shared; j++) {
        float g = s_gate[j];
        float sg = silu_f(g);
        float dsg = silu_deriv(g, sg);
        float d_up = d_shared_act_buf[j] * sg;
        float d_gate = d_shared_act_buf[j] * dsg * s_up[j];
        if (d_gate_shexp) {
            for (int k = 0; k < d_model; k++)
                d_gate_shexp[k * d_ff_shared + j] += d_gate * x_s[k];
        }
        if (d_up_shexp) {
            for (int k = 0; k < d_model; k++)
                d_up_shexp[k * d_ff_shared + j] += d_up * x_s[k];
        }
        for (int k = 0; k < d_model; k++) {
            d_x_s[k] += d_gate * w->ffn_gate_shexp[k * d_ff_shared + j]
                      + d_up * w->ffn_up_shexp[k * d_ff_shared + j];
        }
    }
}

// ========================================================================
// Public API matching wubu_moe.h signature
// ========================================================================
void wubu_moe_backward(const float *d_output, int B, int T,
                       const float *normed2,
                       const moe_weights_t *w,
                       float *d_normed2,
                       float *d_gate_inp,
                       float *d_gate_exps,
                       float *d_up_exps,
                       float *d_down_exps,
                       float *d_gate_shexp,
                       float *d_up_shexp,
                       float *d_down_shexp)
{
    const int n_experts = N_EXPERTS;
    const int n_active_experts = N_ACTIVE_EXPTS;
    const int d_model = D_MODEL;
    const int d_ff = D_FF;
    float *x = (float *)normed2;
    float *d_x = d_normed2;
    int *selected_experts = NULL;

    if (!w || !w->loaded) {
        // No weights loaded: identity backward
        int N = B * T;
        memcpy(d_x, d_output, N * d_model * sizeof(float));
        return;
    }

    bool has_expert_weights = (w->ffn_gate_exps && w->ffn_up_exps && w->ffn_down_exps);
    bool has_shared = (w->ffn_gate_shexp && w->ffn_up_shexp && w->ffn_down_shexp);
    bool has_router = (w->ffn_gate_inp != NULL);

    int N = B * T;

    float *softmax_vals = (float *)malloc(N * n_experts * sizeof(float));
    int *topk_indices = (int *)malloc((size_t)N * n_active_experts * sizeof(int));
    selected_experts = topk_indices; // recomputed here
    float *topk_weights = (float *)malloc((size_t)N * n_active_experts * sizeof(float));
    float *expert_temp = (float *)malloc(d_ff * 3 * sizeof(float));
    int d_ff_shared = SHARED_D_FF;
    float *shared_temp = (float *)malloc(d_ff_shared * 3 * sizeof(float));
    float *d_shared_act_buf = (float *)malloc(d_ff_shared * sizeof(float));

    if (!softmax_vals || !topk_indices || !topk_weights || !expert_temp ||
        !shared_temp || !d_shared_act_buf) {
        memcpy(d_x, d_output, N * d_model * sizeof(float));
        goto cleanup;
    }

    // Recompute router if gate_inp available
    if (has_router) {
        moe_router_backward_prep(x, B, T, w->ffn_gate_inp,
                                 softmax_vals, topk_indices, topk_weights);
    }

    memset(d_x, 0, N * d_model * sizeof(float));

    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * d_model;
        const float *d_out_s = d_output + s * d_model;
        const float *scores_s = has_router ? softmax_vals + s * n_experts : NULL;
        const int *ind_s = selected_experts + s * n_active_experts;
        const float *wt_s = has_router ? topk_weights + s * n_active_experts : NULL;
        float *d_x_s = d_x + s * d_model;

        // ===== SHARED EXPERT BACKWARD =====
        if (has_shared) {
            moe_shared_backward(x_s, d_out_s, w,
                                d_gate_shexp, d_up_shexp, d_down_shexp,
                                d_x_s,
                                shared_temp, d_shared_act_buf,
                                d_model, d_ff_shared);
        }

        // ===== ROUTED EXPERT BACKWARD =====
        if (has_expert_weights && has_router) {
            // Compute d_wgt for each active expert
            float d_wgt[64]; // max N_ACTIVE_EXPTS
            for (int k = 0; k < n_active_experts; k++) {
                int e = ind_s[k];
                if (e < 0 || wt_s[k] < 1e-30f) { d_wgt[k] = 0.0f; continue; }
                const float *gate_w = w->ffn_gate_exps + (int64_t)e * d_model * d_ff;
                const float *up_w   = w->ffn_up_exps   + (int64_t)e * d_model * d_ff;
                const float *down_w = w->ffn_down_exps  + (int64_t)e * d_ff * d_model;
                float *e_gate = expert_temp;
                float *e_up = expert_temp + d_ff;
                float *e_act = expert_temp + 2 * d_ff;
                for (int j = 0; j < d_ff; j++) {
                    double sum = 0.0;
                    for (int p = 0; p < d_model; p++)
                        sum += (double)x_s[p] * (double)gate_w[p * d_ff + j];
                    e_gate[j] = (float)sum;
                }
                for (int j = 0; j < d_ff; j++) {
                    double sum = 0.0;
                    for (int p = 0; p < d_model; p++)
                        sum += (double)x_s[p] * (double)up_w[p * d_ff + j];
                    e_up[j] = (float)sum;
                }
                for (int j = 0; j < d_ff; j++)
                    e_act[j] = silu_f(e_gate[j]) * e_up[j];
                double dw = 0.0;
                for (int j = 0; j < d_model; j++) {
                    double e_out_j = 0.0;
                    for (int p = 0; p < d_ff; p++)
                        e_out_j += (double)e_act[p] * (double)down_w[p * d_model + j];
                    dw += (double)d_out_s[j] * e_out_j;
                }
                d_wgt[k] = (float)dw;
            }

            // Router gradient
            float S_top = 1e-30f;
            for (int k = 0; k < n_active_experts; k++)
                if (ind_s[k] >= 0) S_top += scores_s[ind_s[k]];
            float sum_dw_s = 0.0f;
            for (int k = 0; k < n_active_experts; k++)
                if (ind_s[k] >= 0) sum_dw_s += d_wgt[k] * scores_s[ind_s[k]];

            float d_softmax[256]; // max N_EXPERTS
            memset(d_softmax, 0, sizeof(d_softmax));
            for (int k = 0; k < n_active_experts; k++) {
                int e = ind_s[k];
                if (e < 0) continue;
                d_softmax[e] = d_wgt[k] / S_top - wt_s[k] * sum_dw_s / (S_top * S_top);
            }
            float s_dot_ds = 0.0f;
            for (int e = 0; e < n_experts; e++)
                s_dot_ds += scores_s[e] * d_softmax[e];

            float d_score[256];
            for (int e = 0; e < n_experts; e++)
                d_score[e] = scores_s[e] * (d_softmax[e] - s_dot_ds);

            if (d_gate_inp && w->ffn_gate_inp) {
                for (int e = 0; e < n_experts; e++)
                    for (int p = 0; p < d_model; p++)
                        d_gate_inp[p * n_experts + e] += d_score[e] * x_s[p];
            }
            for (int p = 0; p < d_model; p++) {
                double sum = 0.0;
                for (int e = 0; e < n_experts; e++)
                    sum += (double)d_score[e] * (double)w->ffn_gate_inp[p * n_experts + e];
                d_x_s[p] += (float)sum;
            }

            // Expert backward (weight grads into caller buffers)
            float d_expert_scratch[2048]; // max D_MODEL
            for (int k = 0; k < n_active_experts; k++) {
                int e = ind_s[k];
                float wgt = wt_s[k];
                if (e < 0 || wgt < 1e-30f) continue;
                const float *gate_w = w->ffn_gate_exps + (int64_t)e * d_model * d_ff;
                const float *up_w   = w->ffn_up_exps   + (int64_t)e * d_model * d_ff;
                const float *down_w = w->ffn_down_exps  + (int64_t)e * d_ff * d_model;
                for (int j = 0; j < d_model; j++)
                    d_expert_scratch[j] = d_out_s[j] * wgt;
                float *d_gw = d_gate_exps ? d_gate_exps + (int64_t)e * d_model * d_ff : NULL;
                float *d_uw = d_up_exps   ? d_up_exps   + (int64_t)e * d_model * d_ff : NULL;
                float *d_dw = d_down_exps ? d_down_exps + (int64_t)e * d_ff * d_model : NULL;
                moe_expert_backward(x_s, gate_w, up_w, down_w,
                                   d_expert_scratch, expert_temp,
                                   d_gw, d_uw, d_dw, d_x_s,
                                   d_model, d_ff);
            }
        }

        // If no router/expert weights: identity gradient for routed components
        if (!has_expert_weights && !has_shared) {
            for (int k = 0; k < d_model; k++)
                d_x_s[k] += d_out_s[k];
        }
    }

cleanup:
    free(softmax_vals);
    free(topk_indices);
    free(topk_weights);
    free(expert_temp);
    free(shared_temp);
    free(d_shared_act_buf);
}
