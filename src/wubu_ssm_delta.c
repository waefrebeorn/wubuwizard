/*
 * wubu_ssm_delta.c — Gated DeltaNet 3:1 hybrid linear attention.
 *
 * Contains the core recurrence: sequential (exact) and the BPTT backward.
 * Extracted from wubu_ssm.c to isolate the DeltaNet algorithm.
 */

#include "wubu_ssm.h"
#include "wubu_ssm_utils.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

/* SSM_D_STATE = 128 = 16 × 8 (nice for AVX2 which processes 8 floats at a time) */
#define SSM_STATE_STRIDE 128

#ifdef __AVX2__

/* ============================================================
 * AVX2-optimized SSM selective scan helpers
 * ============================================================ */

/* State decay: h[i][j] *= gg for all i,j in [0,SSM_D_STATE) */
static inline void avx2_state_decay(float *h, float gg) {
    const int n = SSM_STATE_STRIDE * SSM_STATE_STRIDE;  /* 16384 */
    const __m256 v_gg = _mm256_set1_ps(gg);
    for (int i = 0; i < n; i += 8) {
        _mm256_storeu_ps(h + i, _mm256_mul_ps(_mm256_loadu_ps(h + i), v_gg));
    }
}

/* h @ k: hk[i] = sum_j h[i][j] * k[j] */
static inline void avx2_hk(const float *h, const float *k, float *hk) {
    const int d = SSM_STATE_STRIDE;
    for (int i = 0; i < d; i++) {
        const float *h_row = h + i * d;
        __m256 sum = _mm256_setzero_ps();
        for (int j = 0; j < d; j += 8) {
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(h_row + j),
                                  _mm256_loadu_ps(k + j), sum);
        }
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        hk[i] = _mm_cvtss_f32(lo);
    }
}

/* State update: h[i][j] += k[i] * diff[j] * bg */
static inline void avx2_state_update(float *h, const float *k,
                                      const float *diff, float bg) {
    const int d = SSM_STATE_STRIDE;
    for (int i = 0; i < d; i++) {
        float k_bg = k[i] * bg;
        float *h_row = h + i * d;
        for (int j = 0; j < d; j++) {
            h_row[j] += k_bg * diff[j];
        }
    }
}

/* h @ q: out[i] = sum_j h[i][j] * q[j] */
static inline void avx2_hq(const float *h, const float *q, float *out) {
    const int d = SSM_STATE_STRIDE;
    for (int i = 0; i < d; i++) {
        const float *h_row = h + i * d;
        __m256 sum = _mm256_setzero_ps();
        for (int j = 0; j < d; j += 8) {
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(h_row + j),
                                  _mm256_loadu_ps(q + j), sum);
        }
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        out[i] = _mm_cvtss_f32(lo);
    }
}
#endif /* __AVX2__ */

/* ============================================================
 * Sequential SSM recurrence (exact match to original code,
 * extracted for chunked verification)
 * ============================================================ */
void wubu_ssm_sequential_recurrence(int B, int T,
                                     const float *q_norm,
                                     const float *k_norm,
                                     const float *v_conv,
                                     const float *beta_flat,
                                     const float *gate_flat,
                                     float *ssm_state,
                                     float *delta_out)
{
    const int d  = SSM_D_STATE;
    const int hk = SSM_K_HEADS;
    const int hv = SSM_V_HEADS;
    const int rf = hv / hk;
    const float q_scale = 1.0f / sqrtf((float)d);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int s = b * T + t;
            const float *beta_s = beta_flat + s * hv;
            const float *gate_s = gate_flat + s * hv;

            for (int vh = 0; vh < hv; vh++) {
                int kh = vh / rf;
                float bg = beta_s[vh];
                float gg = tgt_safe_expf(gate_s[vh]);

                const float *q_vh = q_norm + (s * hk + kh) * d;
                const float *k_vh = k_norm + (s * hk + kh) * d;
                const float *v_vh = v_conv + (s * hv + vh) * d;
                float *h = ssm_state + (vh * d * d);

                float q_scaled[d];
                for (int i = 0; i < d; i++)
                    q_scaled[i] = q_vh[i] * q_scale;

                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                        h[i * d + j] *= gg;

                float hk_v[d];
                memset(hk_v, 0, sizeof(hk_v));
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                        hk_v[i] += h[i * d + j] * k_vh[j];

                float diff[d];
                for (int i = 0; i < d; i++)
                    diff[i] = v_vh[i] - hk_v[i];

                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                        h[i * d + j] += k_vh[j] * diff[i] * bg;

                float *out = delta_out + (s * hv + vh) * d;
                memset(out, 0, d * sizeof(float));
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                        out[i] += h[i * d + j] * q_scaled[j];
            }
        }
    }
}

/* ============================================================
 * Backward Pass — SSM Delta Net Recurrence (Step 9) — BPTT
 * ============================================================ */
void wubu_ssm_backward_recurrence(
    int B, int T,
    const float *saved_states,
    const float *q_norm, const float *k_norm,
    const float *v_conv,
    const float *beta_flat, const float *gate_flat,
    const float *d_output,
    float *d_q_norm, float *d_k_norm,
    float *d_v_conv,
    float *d_beta_flat, float *d_gate_flat,
    float *d_state_init)
{
    const int d = SSM_D_STATE;
    const int n_vh = SSM_V_HEADS;
    const int n_kh = SSM_K_HEADS;
    const int repeat = n_vh / n_kh;
    const int N = B * T;
    const int state_sz = n_vh * d * d;

    float *d_h_next = (float *)calloc(state_sz, sizeof(float));
    if (!d_h_next) {
        fprintf(stderr, "backward_recurrence: d_h_next alloc failed\n");
        return;
    }

    for (int t = T - 1; t >= 0; t--) {
        for (int b = 0; b < B; b++) {
            int s = b * T + t;
            const float *h_old = saved_states + t * state_sz;
            const float *h_new = saved_states + (t + 1) * state_sz;

            float *d_h_new = (float *)calloc(state_sz, sizeof(float));
            if (!d_h_new) { free(d_h_next); return; }

            /* dL/dh_new += d_output @ q^T */
            for (int vh = 0; vh < n_vh; vh++) {
                int kh = vh / repeat;
                const float *q_vh = q_norm + (s * n_kh + kh) * d;
                const float *do_vh = d_output + (s * n_vh + vh) * d;
                for (int i = 0; i < d; i++) {
                    float *dh = d_h_new + vh * d * d + i * d;
                    float do_i = do_vh[i];
                    for (int j = 0; j < d; j++) {
                        dh[j] += do_i * q_vh[j];
                    }
                }
            }

            /* Add BPTT term from future timesteps */
            for (int i = 0; i < state_sz; i++) {
                d_h_new[i] += d_h_next[i];
            }

            /* dL/dh_old from d_h_new through the recurrence */
            for (int vh = 0; vh < n_vh; vh++) {
                int kh = vh / repeat;
                const float *k_vh = k_norm + (s * n_kh + kh) * d;
                float bg = beta_flat[s * DT_RANK + kh];
                float gg = expf(gate_flat[s * DT_RANK + kh]);

                for (int j = 0; j < d; j++) {
                    double S = 0.0;
                    for (int m = 0; m < d; m++) {
                        S += (double)d_h_new[vh * d * d + m * d + j] * (double)k_vh[m];
                    }
                    float factor = gg * bg * (float)S;
                    for (int i = 0; i < d; i++) {
                        float grad = d_h_new[vh * d * d + i * d + j] * gg;
                        grad -= k_vh[i] * factor;
                        if (t > 0) {
                            d_h_next[vh * d * d + i * d + j] = grad;
                        } else if (d_state_init) {
                            d_state_init[vh * d * d + i * d + j] += grad;
                        }
                    }
                }
            }

            /* Gradients w.r.t. k, q, v, gg, bg */
            for (int vh = 0; vh < n_vh; vh++) {
                int kh = vh / repeat;
                const float *h_new_vh = h_new + vh * d * d;
                const float *do_vh = d_output + (s * n_vh + vh) * d;
                float *dq_vh = d_q_norm + (s * n_kh + kh) * d;
                for (int j = 0; j < d; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < d; i++) {
                        sum += (double)do_vh[i] * (double)h_new_vh[i * d + j];
                    }
                    dq_vh[j] += (float)sum;
                }
            }

            for (int vh = 0; vh < n_vh; vh++) {
                int kh = vh / repeat;
                const float *k_vh = k_norm + (s * n_kh + kh) * d;
                const float *v_vh = v_conv + (s * n_vh + vh) * d;
                const float *h_old_vh = h_old + vh * d * d;
                float bg = beta_flat[s * DT_RANK + kh];
                float gg = expf(gate_flat[s * DT_RANK + kh]);
                float *dk_vh = d_k_norm + (s * n_kh + kh) * d;
                float *dv_vh = d_v_conv + (s * n_vh + vh) * d;

                float diff[SSM_D_STATE];
                for (int n = 0; n < d; n++) {
                    double hk_val = 0.0;
                    for (int p = 0; p < d; p++)
                        hk_val += (double)h_old_vh[p * d + n] * (double)k_vh[p];
                    diff[n] = v_vh[n] - gg * (float)hk_val;
                }

                /* dL/dk_vh[i] */
                for (int i = 0; i < d; i++) {
                    double grad_k = 0.0;
                    for (int n = 0; n < d; n++) {
                        grad_k += (double)d_h_new[vh * d * d + i * d + n] * (double)diff[n] * (double)bg;
                    }
                    for (int m = 0; m < d; m++) {
                        for (int n = 0; n < d; n++) {
                            grad_k -= (double)gg * (double)bg
                                    * (double)d_h_new[vh * d * d + m * d + n]
                                    * (double)k_vh[m] * (double)h_old_vh[i * d + n];
                        }
                    }
                    dk_vh[i] += (float)grad_k;
                }

                /* dL/dv_vh[j] */
                for (int j = 0; j < d; j++) {
                    double grad_v = 0.0;
                    for (int i = 0; i < d; i++) {
                        grad_v += (double)d_h_new[vh * d * d + i * d + j]
                                * (double)k_vh[i] * (double)bg;
                    }
                    dv_vh[j] += (float)grad_v;
                }

                /* dL/dbg */
                {
                    double grad_bg = 0.0;
                    for (int i = 0; i < d; i++) {
                        for (int j = 0; j < d; j++) {
                            grad_bg += (double)d_h_new[vh * d * d + i * d + j]
                                     * (double)k_vh[i] * (double)diff[j];
                        }
                    }
                    d_beta_flat[s * DT_RANK + kh] += (float)grad_bg;
                }

                /* dL/dgg */
                {
                    double grad_gg = 0.0;
                    for (int i = 0; i < d; i++) {
                        for (int j = 0; j < d; j++) {
                            grad_gg += (double)d_h_new[vh * d * d + i * d + j]
                                     * (double)(h_old_vh[i * d + j] - k_vh[i] * bg * diff[j]);
                        }
                    }
                    d_gate_flat[s * DT_RANK + kh] += (float)(grad_gg * gg);
                }
            }

            free(d_h_new);
        }
    }

    free(d_h_next);
}
