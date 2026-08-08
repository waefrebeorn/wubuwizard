#include "wubu_ssm.h"
#include "wubu_mobius.h"
#include "gguf_reader.h"
#include "thread_pool.h"
#include "wubu_model.h"  // for kv_cache_read_head / kv_cache_write_head
#include <omp.h>
#include <immintrin.h>  // AVX2/FMA intrinsics for GQA attention
// GQA_MAX_CTX from wubu_model.h — max KV cache positions (also used for attn stack buf)
#ifndef GQA_MAX_CTX
#define GQA_MAX_CTX 4096
#endif
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#ifdef GPU_SUPPORT
#include <cuda_runtime.h>
#endif

// ============================================================
// SSM Workspace: pre-allocated buffers reused across all layers.
// All pointers 64-byte aligned via posix_memalign.
// ============================================================
ssm_workspace_t *wubu_ssm_workspace_alloc(int B, int T) {
    const int N = B * T;
    const int C = CONV_DIM;
    ssm_workspace_t *ws = (ssm_workspace_t *)malloc(sizeof(ssm_workspace_t));
    if (!ws) return NULL;
    memset(ws, 0, sizeof(ssm_workspace_t));
    ws->B = B; ws->T = T; ws->N = N;

    #define WS_ALLOC(ptr, count) do { \
        size_t _sz = (size_t)(count) * sizeof(float); \
        if (_sz > 0 && posix_memalign((void**)&(ptr), 64, _sz)) { \
            fprintf(stderr, "SSM workspace: " #ptr " alloc failed (%zu bytes)\n", _sz); \
            wubu_ssm_workspace_free(ws); return NULL; \
        } \
    } while(0)

    WS_ALLOC(ws->qkv_all,        (int64_t)N * (KEY_DIM * 2 + VALUE_DIM));
    WS_ALLOC(ws->z_all,          (int64_t)N * VALUE_DIM);
    WS_ALLOC(ws->beta_raw,       (int64_t)N * DT_RANK);
    WS_ALLOC(ws->alpha_raw,      (int64_t)N * DT_RANK);
    WS_ALLOC(ws->conv_input,     (int64_t)B * (T + CONV_KERNEL - 1) * (int64_t)C);
    WS_ALLOC(ws->conv_output,    (int64_t)N * C);
    WS_ALLOC(ws->q_conv,         (int64_t)N * KEY_DIM);
    WS_ALLOC(ws->k_conv,         (int64_t)N * KEY_DIM);
    WS_ALLOC(ws->v_conv,         (int64_t)N * VALUE_DIM);
    WS_ALLOC(ws->q_norm,         (int64_t)N * KEY_DIM);
    WS_ALLOC(ws->k_norm,         (int64_t)N * KEY_DIM);
    WS_ALLOC(ws->delta_out,      (int64_t)N * VALUE_DIM);
    WS_ALLOC(ws->z_silu,         (int64_t)N * VALUE_DIM);
    WS_ALLOC(ws->beta_flat,      (int64_t)N * DT_RANK);
    WS_ALLOC(ws->gate_flat,      (int64_t)N * DT_RANK);
    WS_ALLOC(ws->alpha_biased,   (int64_t)N * DT_RANK);
    WS_ALLOC(ws->alpha_softplus, (int64_t)N * DT_RANK);

    #undef WS_ALLOC
    return ws;
}

void wubu_ssm_workspace_free(ssm_workspace_t *ws) {
    if (!ws) return;
    free(ws->qkv_all);        free(ws->z_all);
    free(ws->beta_raw);       free(ws->alpha_raw);
    free(ws->conv_input);     free(ws->conv_output);
    free(ws->q_conv);         free(ws->k_conv);
    free(ws->v_conv);         free(ws->q_norm);
    free(ws->k_norm);         free(ws->delta_out);
    free(ws->z_silu);         free(ws->beta_flat);
    free(ws->gate_flat);      free(ws->alpha_biased);
    free(ws->alpha_softplus);
    free(ws);
}

// GPU SSM recurrence (declared in gpu_ssm_recurrence.cu, extern C linkage)
#ifdef GPU_SUPPORT
void wubu_gpu_ssm_recurrence(
    float *ssm_state,
    const float *q, const float *k, const float *v,
    const float *beta, const float *gate,
    float *delta_out,
    void *stream);
#endif
#include <stdio.h>

// QK_K = 256 for all K-quant types (must match quantized_matmul.c)
#ifndef QK_K
#define QK_K 256
#endif

// SSM_D_STATE = 128 = 16 × 8 (nice for AVX2 which processes 8 floats at a time)
#define SSM_STATE_STRIDE 128

// ============================================================
// AVX2-optimized SSM selective scan helpers
// ============================================================

// State decay: h[i][j] *= gg for all i,j in [0,SSM_D_STATE)
// h: [SSM_D_STATE, SSM_D_STATE] row-major = 16384 floats
// gg: scalar multiplier (exp(gate))
static inline void avx2_state_decay(float *h, float gg) {
    const int n = SSM_STATE_STRIDE * SSM_STATE_STRIDE;  // 16384
    const __m256 v_gg = _mm256_set1_ps(gg);
    for (int i = 0; i < n; i += 8) {
        _mm256_storeu_ps(h + i, _mm256_mul_ps(_mm256_loadu_ps(h + i), v_gg));
    }
}

// h @ k: hk[i] = sum_j h[i][j] * k[j]
// h:  [SSM_D_STATE, SSM_D_STATE]
// k:  [SSM_D_STATE]
// hk: [SSM_D_STATE] output (must be zeroed before call)
static inline void avx2_hk(const float *h, const float *k, float *hk) {
    const int d = SSM_STATE_STRIDE;
    for (int i = 0; i < d; i++) {
        const float *h_row = h + i * d;
        __m256 sum = _mm256_setzero_ps();
        for (int j = 0; j < d; j += 8) {
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(h_row + j),
                                  _mm256_loadu_ps(k + j), sum);
        }
        // Horizontal reduction
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        hk[i] = _mm_cvtss_f32(lo);
    }
}

// State update: h[i][j] += k[i] * diff[j] * bg
// Equivalent to outer product: h += k ⊗ (diff * bg)
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

// h @ q: out[i] = sum_j h[i][j] * q[j]
// Same pattern as h @ k
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

// ============================================================
// Utility: Activation Functions
// ============================================================

// SSM L2 norm epsilon — read from GGUF config (should be 1e-6 for Qwen3.6)
float g_ssm_l2_eps = 1e-6f;

// Centralized quantized matmul dispatch for SSM/GQA projections.
// If quantized weight (W_q) is available and type is not F32, uses quantized_matmul.
// Otherwise falls back to the provided F32 weight with a column loop.
// x: [n_rows] input, W_f32: [n_rows*n_cols] F32 or NULL, W_q: quantized or NULL
// out: [n_cols] output
static void proj_matmul(const float *x, int64_t n_rows, int64_t n_cols,
                         const float *W_f32, const uint8_t *W_q, int weight_type,
                         float *out) {
    if (W_q && weight_type != GGML_TYPE_F32 && n_cols > 0) {
        quantized_matmul(x, W_q, weight_type, n_rows, n_cols, 0, out);
    } else {
        #pragma omp parallel for if(n_cols > 4)
        for (int64_t j = 0; j < n_cols; j++) {
            double sum = 0.0;
            for (int64_t i = 0; i < n_rows; i++)
                sum += (double)x[i] * (double)W_f32[j * n_rows + i];
            out[j] = (float)sum;
        }
    }
}

void wubu_softplus(int n, const float *x, float *out) {
    #pragma omp parallel for if(n > 100000)
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v > 80.0f) out[i] = v;          // linear region
        else if (v < -80.0f) out[i] = 0.0f; // zero region
        else out[i] = logf(1.0f + expf(v));
    }
}

void wubu_silu(int n, const float *x, float *out) {
    #pragma omp parallel for if(n > 100000)
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v < -80.0f) out[i] = 0.0f;
        else out[i] = v / (1.0f + expf(-v));
    }
}

void wubu_sigmoid(int n, const float *x, float *out) {
    #pragma omp parallel for if(n > 100000)
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v < -80.0f) out[i] = 0.0f;
        else if (v > 80.0f) out[i] = 1.0f;
        else out[i] = 1.0f / (1.0f + expf(-v));
    }
}

// ============================================================
// Utility: Normalization
// ============================================================

void wubu_l2_norm(int B, int T, int n_heads, int d,
                  const float *x, float eps, float *out) {
    // x: [B, T, n_heads, d]
    // out: [B, T, n_heads, d]
    int seq_len = B * T;
    #pragma omp parallel for collapse(2) if(seq_len * n_heads > 100)
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_heads; h++) {
            const float *inp = x + (s * n_heads + h) * d;
            float *oup = out + (s * n_heads + h) * d;
            float sum_sq = 0.0f;
#ifdef __AVX2__
            __m256 acc = _mm256_setzero_ps();
            int i;
            for (i = 0; i <= d - 8; i += 8) {
                __m256 v = _mm256_loadu_ps(inp + i);
                acc = _mm256_fmadd_ps(v, v, acc);
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            sum_sq = _mm_cvtss_f32(lo);
            for (; i < d; i++) sum_sq += inp[i] * inp[i];
#else
            for (int i = 0; i < d; i++) sum_sq += inp[i] * inp[i];
#endif
            float scale = 1.0f / sqrtf(sum_sq + eps);
#ifdef __AVX2__
            __m256 v_scale = _mm256_set1_ps(scale);
            for (int i = 0; i <= d - 8; i += 8)
                _mm256_storeu_ps(oup + i, _mm256_mul_ps(_mm256_loadu_ps(inp + i), v_scale));
            for (int i = (d / 8) * 8; i < d; i++) oup[i] = inp[i] * scale;
#else
            for (int i = 0; i < d; i++) oup[i] = inp[i] * scale;
#endif
        }
    }
}

void wubu_rms_norm(int B, int T, int d,
                   const float *x, const float *weight,
                   float eps, float *out) {
    // x: [B, T, d]
    // weight: [d]
    // out: [B, T, d]
    int seq_len = B * T;
    #pragma omp parallel for if(seq_len > 10)
    for (int s = 0; s < seq_len; s++) {
        const float *inp = x + s * d;
        float *oup = out + s * d;
        float sum_sq = 0.0f;
#ifdef __AVX2__
        __m256 acc = _mm256_setzero_ps();
        int i;
        for (i = 0; i <= d - 8; i += 8) {
            __m256 v = _mm256_loadu_ps(inp + i);
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sum_sq = _mm_cvtss_f32(lo);
        for (; i < d; i++) sum_sq += inp[i] * inp[i];
#else
        for (int i = 0; i < d; i++) sum_sq += inp[i] * inp[i];
#endif
        float rms = sqrtf(sum_sq / d + eps);
        float scale = 1.0f / rms;
#ifdef __AVX2__
        __m256 v_scale = _mm256_set1_ps(scale);
        for (int i = 0; i <= d - 8; i += 8)
            _mm256_storeu_ps(oup + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(inp + i), v_scale), _mm256_loadu_ps(weight + i)));
        for (int i = (d / 8) * 8; i < d; i++) oup[i] = inp[i] * scale * weight[i];
#else
        for (int i = 0; i < d; i++) oup[i] = inp[i] * scale * weight[i];
#endif
    }
}

// ============================================================
// Utility: Matrix multiply (simple, non-optimized)
// ============================================================

static void matmul_nt(int M, int N, int K,
                      const float *A, const float *B,
                      float *C) {
    // C[M,N] = A[M,K] @ B[N,K]^T  (B is stored NxK but we use it as KxN internally)
    // Non-atomic: each thread writes to C[m, 0:N] for its assigned m range
    long long ops = (long long)M * N * K;
    #pragma omp parallel for if(ops > 500000)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];  // B[N,K] stored row-major
            }
            C[m * N + n] = sum;
        }
    }
}

// ============================================================
// Utility: 1D Convolution (depthwise, causal)
// ============================================================

void wubu_conv1d(int B, int T, int C, int k,
                 const float *input, const float *kernel,
                 float *output) {
    if (B <= 0 || T <= 0 || C <= 0 || k <= 0) return;
    // input: [B, T+k-1, C] — already padded with k-1 zeros at start
    // kernel: [k, C]
    // output: [B, T, C]
    #pragma omp parallel for collapse(2) if(B * T * C * k > 100000)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // AVX2 conv1d disabled: kernel[ki + c*k] is per-channel.
            // Broadcasting one channel's kernel to all 8 vector channels
            // produces wrong output. Sequential path always used.
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int ki = 0; ki < k; ki++) {
                    int t_in = t + ki;
                    sum += input[(b * (T + k - 1) + t_in) * C + c] *
                           kernel[ki + c * k];
                }
                output[(b * T + t) * C + c] = sum;
            }
        }
    }
}

// TGT (Toroidal Gradient Transformation) safe wrapping using π odometer
// Keeps values in [-π, π] range, preventing float32 overflow
#define TGT_PI      3.14159265358979323846f
#define TGT_BOUNDARY (2.0f * TGT_PI)

static inline float tgt_wrap(float x) {
    return fmodf(x + TGT_PI, TGT_BOUNDARY) - TGT_PI;
}

static inline float tgt_safe_expf(float x) {
    // Clamp to avoid float32 overflow: expf(89) ≈ 5e38 ≈ overflow
    if (x > 80.0f) x = 80.0f;
    if (x < -80.0f) return 0.0f;
    return expf(x);
}

// ============================================================
// SSM Layer Forward Pass (Euclidean)
// ============================================================

void wubu_ssm_forward(const float *x, int B, int T,
                      const ssm_layer_weights *w,
                      float *ssm_state,
                      float *conv_state,
                      float *output,
                      const float *gpu_qkv, const float *gpu_z,
                      ssm_workspace_t *ws) {
    // x: [B, T, D_MODEL]
    // output: [B, T, D_MODEL]
    
    const int N = B * T;  // total tokens
    const int C = CONV_DIM;  // 8192
    
    // Use pre-allocated workspace when available (avoids 17 malloc/free per layer)
    // Fall back to per-call allocation for backward compatibility
    int own_alloc = 0;
    float *qkv_all, *z_all, *beta_raw, *alpha_raw;
    float *conv_input, *conv_output;
    float *q_conv, *k_conv, *v_conv;
    float *q_norm, *k_norm;
    float *delta_out, *z_silu;
    
    if (ws && ws->N >= N && ws->B >= B && ws->T >= T) {
        qkv_all   = ws->qkv_all;
        z_all     = ws->z_all;
        beta_raw  = ws->beta_raw;
        alpha_raw = ws->alpha_raw;
        conv_input  = ws->conv_input;
        conv_output = ws->conv_output;
        q_conv    = ws->q_conv;
        k_conv    = ws->k_conv;
        v_conv    = ws->v_conv;
        q_norm    = ws->q_norm;
        k_norm    = ws->k_norm;
        delta_out = ws->delta_out;
        z_silu    = ws->z_silu;
    } else {
        own_alloc = 1;
        qkv_all = (float *)malloc(N * (KEY_DIM * 2 + VALUE_DIM) * sizeof(float));
        z_all = (float *)malloc(N * VALUE_DIM * sizeof(float));
        beta_raw = (float *)malloc(N * DT_RANK * sizeof(float));
        alpha_raw = (float *)malloc(N * DT_RANK * sizeof(float));
        conv_input = (float *)malloc(B * (T + CONV_KERNEL - 1) * C * sizeof(float));
        conv_output = (float *)malloc(N * C * sizeof(float));
        q_conv = (float *)malloc(N * KEY_DIM * sizeof(float));
        k_conv = (float *)malloc(N * KEY_DIM * sizeof(float));
        v_conv = (float *)malloc(N * VALUE_DIM * sizeof(float));
        q_norm = (float *)malloc(N * KEY_DIM * sizeof(float));
        k_norm = (float *)malloc(N * KEY_DIM * sizeof(float));
        delta_out = (float *)malloc(N * VALUE_DIM * sizeof(float));
        z_silu = (float *)malloc(N * VALUE_DIM * sizeof(float));
        
        if (!qkv_all || !z_all || !beta_raw || !alpha_raw || !conv_input ||
            !conv_output || !q_conv || !k_conv || !v_conv || !q_norm || !k_norm ||
            !delta_out || !z_silu) {
            fprintf(stderr, "SSM forward: allocation failed\n");
            free(qkv_all); free(z_all); free(beta_raw); free(alpha_raw);
            free(conv_input); free(conv_output);
            free(q_conv); free(k_conv); free(v_conv);
            free(q_norm); free(k_norm);
            free(delta_out); free(z_silu);
            return;
        }
    }
    
    const char *dd = getenv("DUMP_SSM_DEBUG");
    // Step 1+2: QKV + gate projection
    // If gpu_qkv/gpu_z are provided, skip the CPU quantized matmuls
    if (gpu_qkv && gpu_z) {
        memcpy(qkv_all, gpu_qkv, (size_t)N * C * sizeof(float));
        memcpy(z_all, gpu_z, (size_t)N * VALUE_DIM * sizeof(float));
        if (dd) printf("  [SSM] Using GPU projections\n");
    } else if (N > 1) {
            // Batched prefill: quantize all tokens, weight read once from RAM
            quantized_matmul_batched(x,
                w->attn_qkv_weight_q, w->attn_qkv_weight_type,
                D_MODEL, C, 0, N, qkv_all);
            quantized_matmul_batched(x,
                w->attn_gate_weight_q, w->attn_gate_weight_type,
                D_MODEL, VALUE_DIM, 0, N, z_all);
        } else {
            // Decode (N=1): quantize once, reuse for both projections
            const int n_q8_blocks = (D_MODEL + QK_K - 1) / QK_K;
            const int q8_buf_size = n_q8_blocks * 292;
            uint8_t *ssm_q8_buf = (uint8_t *)malloc(q8_buf_size);
            if (!ssm_q8_buf) { fprintf(stderr, "SSM forward: q8 alloc failed\n"); goto cleanup; }
            quantize_row_q8_K(x, (block_q8_K *)ssm_q8_buf, D_MODEL);
            quantized_matmul_from_q8(ssm_q8_buf,
                w->attn_qkv_weight_q, w->attn_qkv_weight_type,
                D_MODEL, C, 0, qkv_all);
            quantized_matmul_from_q8(ssm_q8_buf,
                w->attn_gate_weight_q, w->attn_gate_weight_type,
                D_MODEL, VALUE_DIM, 0, z_all);
            free(ssm_q8_buf);
        }
    if (dd) {
        FILE *f = fopen("/tmp/dbg_qkv_out.bin", "wb");
        if (f) { fwrite(qkv_all, sizeof(float), N * C, f); fclose(f); }
    }
    
    // Step 3: beta/alpha projections
    #pragma omp parallel for
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *beta_s = beta_raw + s * DT_RANK;
        float *alpha_s = alpha_raw + s * DT_RANK;
        for (int j = 0; j < DT_RANK; j++) {
            float sum_b = 0.0f, sum_a = 0.0f;
            for (int i = 0; i < D_MODEL; i++) {
                sum_b += x_s[i] * w->ssm_beta_weight[i + j * D_MODEL];
                sum_a += x_s[i] * w->ssm_alpha_weight[i + j * D_MODEL];
            }
            beta_s[j] = sum_b;
            alpha_s[j] = sum_a;
        }
    }
    
    // Step 4: Compute beta and gate (decay)
    // beta = sigmoid(beta_raw)
    // alpha_biased = alpha + ssm_dt_bias -> softplus -> * ssm_a
    // Beta/gate/alpha buffers: use workspace when available
    float *beta_flat, *gate_flat, *alpha_biased, *alpha_softplus;
    if (ws && ws->N >= N && ws->B >= B && ws->T >= T) {
        beta_flat = ws->beta_flat;
        gate_flat = ws->gate_flat;
        alpha_biased = ws->alpha_biased;
        alpha_softplus = ws->alpha_softplus;
    } else {
        beta_flat = (float *)malloc(N * DT_RANK * sizeof(float));
        gate_flat = (float *)malloc(N * DT_RANK * sizeof(float));
        if (!beta_flat || !gate_flat) {
            fprintf(stderr, "SSM forward: beta/gate alloc failed\n");
            goto cleanup;
        }
        
        alpha_biased = (float *)malloc(N * DT_RANK * sizeof(float));
        alpha_softplus = (float *)malloc(N * DT_RANK * sizeof(float));
        if (!alpha_biased || !alpha_softplus) goto cleanup;
    }
    
    wubu_sigmoid(N * DT_RANK, beta_raw, beta_flat);
    
    for (int s = 0; s < N; s++) {
        for (int j = 0; j < DT_RANK; j++) {
            alpha_biased[s * DT_RANK + j] = alpha_raw[s * DT_RANK + j] + w->ssm_dt_bias[j];
        }
    }
    wubu_softplus(N * DT_RANK, alpha_biased, alpha_softplus);
    
    for (int s = 0; s < N; s++) {
        for (int j = 0; j < DT_RANK; j++) {
            gate_flat[s * DT_RANK + j] = alpha_softplus[s * DT_RANK + j] * w->ssm_a[j];
        }
    }
    
    // Step 5: Convolution
    // Build conv_input: [B, T+CONV_KERNEL-1, C]
    // First CONV_KERNEL-1 elements are from conv_state, rest from qkv_all
    for (int b = 0; b < B; b++) {
        // Copy conv_state
        memcpy(conv_input + b * (T + CONV_KERNEL - 1) * C,
               conv_state + b * (CONV_KERNEL - 1) * C,
               (CONV_KERNEL - 1) * C * sizeof(float));
        // Copy qkv_all
        memcpy(conv_input + (b * (T + CONV_KERNEL - 1) + (CONV_KERNEL - 1)) * C,
               qkv_all + b * T * C,
               T * C * sizeof(float));
    }
    
    // Run convolution
    wubu_conv1d(B, T, C, CONV_KERNEL, conv_input, w->ssm_conv1d_weight, conv_output);
    
    // SiLU activation on conv output
    wubu_silu(N * C, conv_output, conv_output);
    if (dd) {
        FILE *f = fopen("/tmp/dbg_conv_out.bin", "wb");
        if (f) { fwrite(conv_output, sizeof(float), N * C, f); fclose(f); }
    }
    
    // Update conv_state: last CONV_KERNEL-1 elements of conv_input
    for (int b = 0; b < B; b++) {
        float *ci = conv_input + (b * (T + CONV_KERNEL - 1) + T) * C;  // last k-1 elements
        memcpy(conv_state + b * (CONV_KERNEL - 1) * C, ci,
               (CONV_KERNEL - 1) * C * sizeof(float));
    }
    
    // Step 6: Split conv output into Q, K, V
    for (int s = 0; s < N; s++) {
        const float *cv = conv_output + s * C;
        memcpy(q_conv + s * KEY_DIM, cv, KEY_DIM * sizeof(float));
        memcpy(k_conv + s * KEY_DIM, cv + KEY_DIM, KEY_DIM * sizeof(float));
        memcpy(v_conv + s * VALUE_DIM, cv + 2 * KEY_DIM, VALUE_DIM * sizeof(float));
    }
    
    // Step 7: L2 normalize Q and K
    // q_conv: [N, SSM_K_HEADS, SSM_D_STATE]
    wubu_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, q_conv, g_ssm_l2_eps, q_norm);
    wubu_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, k_conv, g_ssm_l2_eps, k_norm);
    
    if (dd) {
        FILE *f = fopen("/tmp/dbg_l2_q.bin", "wb"); if (f) { fwrite(q_norm, sizeof(float), N * KEY_DIM, f); fclose(f); }
        FILE *g = fopen("/tmp/dbg_l2_k.bin", "wb"); if (g) { fwrite(k_norm, sizeof(float), N * KEY_DIM, g); fclose(g); }
        FILE *h = fopen("/tmp/dbg_beta.bin", "wb"); if (h) { fwrite(beta_flat, sizeof(float), N * DT_RANK, h); fclose(h); }
        FILE *i = fopen("/tmp/dbg_gate.bin", "wb"); if (i) { fwrite(gate_flat, sizeof(float), N * DT_RANK, i); fclose(i); }
    }
    
    // Step 8: Repeat Q/K heads: 16 -> 32 (to match V heads)
    
    // Step 9: Gated Delta Net recurrence per head
    // ssm_state: [SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
    // We process B batches and T timesteps, updating the state in-place
    int repeat_factor = SSM_V_HEADS / SSM_K_HEADS;  // 2
    
    // GPU recurrence path: skip CPU recurrence, set delta_out via GPU kernel
#ifdef GPU_SUPPORT
    if (w->gpu_ssm_state && N == 1 && !getenv("FORCE_CPU_SSM")) {
        float *d_state = (float*)w->gpu_ssm_state;
        float *d_q     = (float*)w->gpu_q_buf;
        float *d_k     = (float*)w->gpu_k_buf;
        float *d_v     = (float*)w->gpu_v_buf;
        float *d_beta  = (float*)w->gpu_beta_buf;
        float *d_gate  = (float*)w->gpu_gate_buf;
        float *d_delta = (float*)w->gpu_delta_buf;
        cudaStream_t st = (cudaStream_t)w->gpu_stream;
        
        float host_q[SSM_V_HEADS * SSM_D_STATE];
        float host_k[SSM_V_HEADS * SSM_D_STATE];
        float host_v[SSM_V_HEADS * SSM_D_STATE];
        float host_beta[SSM_V_HEADS];
        float host_gate[SSM_V_HEADS];
        for (int vh = 0; vh < SSM_V_HEADS; vh++) {
            int kh = vh % SSM_K_HEADS;
            for (int i = 0; i < SSM_D_STATE; i++) {
                host_q[vh * SSM_D_STATE + i] = q_norm[kh * SSM_D_STATE + i];
                host_k[vh * SSM_D_STATE + i] = k_norm[kh * SSM_D_STATE + i];
            }
            memcpy(host_v + vh * SSM_D_STATE,
                   v_conv + vh * SSM_D_STATE, SSM_D_STATE * sizeof(float));
            host_beta[vh] = beta_flat[vh];
            host_gate[vh] = gate_flat[vh];
        }
        cudaMemcpyAsync(d_q, host_q, sizeof(host_q), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(d_k, host_k, sizeof(host_k), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(d_v, host_v, sizeof(host_v), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(d_beta, host_beta, sizeof(host_beta), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(d_gate, host_gate, sizeof(host_gate), cudaMemcpyHostToDevice, st);
        wubu_gpu_ssm_recurrence(d_state, d_q, d_k, d_v, d_beta, d_gate, d_delta, st);
        cudaMemcpyAsync(delta_out, d_delta,
            (size_t)SSM_V_HEADS * SSM_D_STATE * sizeof(float),
            cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        if (dd) printf("  [SSM] GPU recurrence active\n");
        goto gpu_rec_done;
    }
#endif  // GPU_SUPPORT
    
    // Chunked SSM path for prefill (T >= CS tokens, CS compiled into chunked function)
    // Falls through to sequential for decode (T=1) or small batches.
    // Override threshold via SSM_CHUNK_MIN env var (default 1M — sequential always used for correctness).
    // NOTE: Chunked SSM (CS=2) is designed for TRAINING/GPU where the A=(I+L)^{-T}
    // attention matrix mixes tokens within a chunk. This produces DIFFERENT outputs
    // from sequential (cos-sim ~0.96 at T=4 with constant input) because future K
    // within the chunk affects past Q outputs — correct for training, wrong for inference.
    // Sequential path is always correct for inference. Chunked is only appropriate
    // when exact bit-level match with sequential is NOT required (e.g. GPU training).
    int ssm_chunk_min = getenv("SSM_CHUNK_MIN") ? atoi(getenv("SSM_CHUNK_MIN")) : 1000000;
    if (T >= ssm_chunk_min && !getenv("FORCE_CPU_SSM_SEQ")) {
        wubu_ssm_chunked_recurrence(B, T, q_norm, k_norm, v_conv,
                                     beta_flat, gate_flat,
                                     ssm_state, delta_out);
        goto gpu_rec_done;
    }
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int s = b * T + t;
            float *beta_s = beta_flat + s * DT_RANK;  // [DT_RANK=32]
            float *gate_s = gate_flat + s * DT_RANK;   // [DT_RANK=32]
            
            // For each V-head (32 heads):
            // For each V-head (32 heads) — fully parallel, each writes non-overlapping state
            #pragma omp parallel for
            for (int vh = 0; vh < SSM_V_HEADS; vh++) {
                if (dd && (vh == 0 || vh == 2)) {
                    int tmp_kh = vh % SSM_K_HEADS;
                    float tmp_bg = beta_s[vh];
                    float tmp_gg = tgt_safe_expf(gate_s[vh]);
                    const float *tmp_q = q_norm + (s * SSM_K_HEADS + tmp_kh) * SSM_D_STATE;
                    const float *tmp_k = k_norm + (s * SSM_K_HEADS + tmp_kh) * SSM_D_STATE;
                    const float *tmp_v = v_conv + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                    printf("C_DEBUG s=%d vh=%d kh=%d bg=%.6f gg=%.6f\\\\n", s, vh, tmp_kh, tmp_bg, tmp_gg);
                    for (int i = 0; i < 5; i++) printf("C_DEBUG q[%d]=%.8f k[%d]=%.8f v[%d]=%.8f\\\\n",
                        i, tmp_q[i], i, tmp_k[i], i, tmp_v[i]);
                    float *h_debug = ssm_state + (vh * SSM_D_STATE * SSM_D_STATE);
                    for (int ri = 0; ri < 3; ri++) for (int rj = 0; rj < 3; rj++)
                        printf("C_DEBUG state_before[%d][%d]=%.8f\\\\n", ri, rj, h_debug[ri * SSM_D_STATE + rj]);
                }
                
                // Full gated delta net recurrence (matches llama.cpp EXACT)
                int kh = vh % SSM_K_HEADS;
                float beta_val = beta_s[vh];
                float gate_exp = tgt_safe_expf(gate_s[vh]);

                const float *q_vh = q_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                const float *k_vh = k_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                const float *v_vh = v_conv + (s * SSM_V_HEADS + vh) * SSM_D_STATE;

                float *h = ssm_state + (vh * SSM_D_STATE * SSM_D_STATE);

                // Step 8a: State decay — multiply all elements by exp(gate) [llama: ggml_vec_scale]
                for (int jj = 0; jj < SSM_D_STATE * SSM_D_STATE; jj++) {
                    h[jj] *= gate_exp;
                }

                // Step 8b: Compute h @ k -> hk
                float hk_tmp[SSM_D_STATE];
                memset(hk_tmp, 0, sizeof(hk_tmp));
                for (int i = 0; i < SSM_D_STATE; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < SSM_D_STATE; j++) {
                        sum += h[i * SSM_D_STATE + j] * k_vh[j];
                    }
                    hk_tmp[i] = sum;
                }

                // Step 8c: delta = (v - hk) * beta
                float delta[SSM_D_STATE];
                for (int i = 0; i < SSM_D_STATE; i++) {
                    delta[i] = (v_vh[i] - hk_tmp[i]) * beta_val;
                }

                // Step 8d: State update — outer product: h[i][j] += v[i] * k[j] * beta
                // NOTE: This uses delta[i] (row) * k[j] (col) which matches the model's
                // trained convention (transposed outer product). Using k[i] * delta[j] (standard
                // GDN formula) BREAKS single-token output (cos-sim drops 0.97→-0.40).
                // The model was trained with the transposed convention.
                for (int i = 0; i < SSM_D_STATE; i++) {
                    float di = delta[i];
                    float *h_row = h + i * SSM_D_STATE;
                    for (int j = 0; j < SSM_D_STATE; j++) {
                        h_row[j] += k_vh[j] * di;
                    }
                }

                // Step 8e: Output = h @ q * scale
                // [llama: attn_data[j] = sum_i s_out[j*S_v+i] * q[i] * scale]
                float *out = delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                const float q_scale = 1.0f / sqrtf((float)SSM_D_STATE);
                for (int i = 0; i < SSM_D_STATE; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < SSM_D_STATE; j++) {
                        sum += h[i * SSM_D_STATE + j] * q_vh[j];
                    }
                    out[i] = sum * q_scale;
                }
            }
        }
    }
    gpu_rec_done:
    
    if (dd) {
        FILE *f = fopen("/tmp/dbg_state_after_t0.bin", "wb");
        if (f) { fwrite(ssm_state, sizeof(float), SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, f); fclose(f); }
    }
    // Dump Q, K, V for head 0 before recurrence (token 0) for debug
    if (dd) {
        FILE *f = fopen("/tmp/dbg_q0_q.bin", "wb"); if (f) { fwrite(q_norm, sizeof(float), N * KEY_DIM, f); fclose(f); }
        FILE *g = fopen("/tmp/dbg_q0_k.bin", "wb"); if (g) { fwrite(k_norm, sizeof(float), N * KEY_DIM, g); fclose(g); }
        FILE *h = fopen("/tmp/dbg_q0_v.bin", "wb"); if (h) { fwrite(v_conv, sizeof(float), N * VALUE_DIM, h); fclose(h); }
    }
    
    // Step 10: Gated normalization
    if (dd) {
        FILE *f = fopen("/tmp/dbg_delta_out.bin", "wb");
        if (f) { fwrite(delta_out, sizeof(float), N * VALUE_DIM, f); fclose(f); }
    }
    
    // delta_out: [N, SSM_V_HEADS, SSM_D_STATE] = [N, 32, 128]
    // ssm_norm: [SSM_D_STATE] = [128]
    // z_silu: silu(z_all[VALUE_DIM])
    wubu_silu(N * VALUE_DIM, z_all, z_silu);
    
    // RMSNorm along SSM_D_STATE per head
    for (int s = 0; s < N; s++) {
        for (int vh = 0; vh < SSM_V_HEADS; vh++) {
            float *out_vh = delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
            float *z_vh = z_silu + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
            
            // RMSNorm
            float sum_sq = 0.0f;
            for (int i = 0; i < SSM_D_STATE; i++) sum_sq += out_vh[i] * out_vh[i];
            float rms = sqrtf(sum_sq / SSM_D_STATE + 1e-6f);
            float scale = 1.0f / rms;
            
            // Apply norm weight and multiply by silu(z)
            for (int i = 0; i < SSM_D_STATE; i++) {
                out_vh[i] = (out_vh[i] * scale * w->ssm_norm_weight[i]) * z_vh[i];
            }
        }
    }

    // Step 11: Output projection via quantized or F32 matmul
        // Batched for prefill (N>1): single pass through Q6_K weight for all tokens
        if (N > 1 && w->ssm_out_weight_q && w->ssm_out_weight_type != GGML_TYPE_F32) {
            quantized_matmul_batched(delta_out,
                w->ssm_out_weight_q, w->ssm_out_weight_type,
                VALUE_DIM, D_MODEL, 0, N, output);
        } else {
            for (int s = 0; s < N; s++) {
                proj_matmul(delta_out + s * VALUE_DIM, VALUE_DIM, D_MODEL,
                            w->ssm_out_weight, w->ssm_out_weight_q, w->ssm_out_weight_type,
                            output + s * D_MODEL);
            }
        }

    cleanup:
    if (own_alloc) {
        free(qkv_all); free(z_all);
        free(beta_raw); free(alpha_raw);
        free(conv_input); free(conv_output);
        free(q_conv); free(k_conv); free(v_conv);
        free(q_norm); free(k_norm);
        free(delta_out); free(z_silu);
        free(beta_flat); free(gate_flat);
        free(alpha_biased); free(alpha_softplus);
    }
}

// SSM forward with intermediate saving (for backward)
void wubu_ssm_forward_save(const float *x, int B, int T,
                           const ssm_layer_weights *w,
                           float *ssm_state,
                           float *conv_state,
                           float *output,
                           ssm_fwd_save_t *save)
{
    // Forward: run same computation, then save or free intermediates
    const int N = B * T;
    const int C = CONV_DIM;
    
    float *qkv_all = (float *)malloc(N * (KEY_DIM * 2 + VALUE_DIM) * sizeof(float));
    float *z_all = (float *)malloc(N * VALUE_DIM * sizeof(float));
    float *beta_raw = (float *)malloc(N * DT_RANK * sizeof(float));
    float *alpha_raw = (float *)malloc(N * DT_RANK * sizeof(float));
    float *conv_input = (float *)malloc(B * (T + CONV_KERNEL - 1) * C * sizeof(float));
    float *conv_output = (float *)malloc(N * C * sizeof(float));
    float *q_conv = (float *)malloc(N * KEY_DIM * sizeof(float));
    float *k_conv = (float *)malloc(N * KEY_DIM * sizeof(float));
    float *v_conv = (float *)malloc(N * VALUE_DIM * sizeof(float));
    float *q_norm = (float *)malloc(N * KEY_DIM * sizeof(float));
    float *k_norm = (float *)malloc(N * KEY_DIM * sizeof(float));
    float *delta_out = (float *)malloc(N * VALUE_DIM * sizeof(float));
    float *z_silu = (float *)malloc(N * VALUE_DIM * sizeof(float));
    float *beta_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    float *gate_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    float *alpha_biased = (float *)malloc(N * DT_RANK * sizeof(float));
    float *alpha_softplus = (float *)malloc(N * DT_RANK * sizeof(float));
    
    if (!qkv_all || !z_all || !beta_raw || !alpha_raw || !conv_input ||
        !conv_output || !q_conv || !k_conv || !v_conv || !q_norm || !k_norm ||
        !delta_out || !z_silu || !beta_flat || !gate_flat || !alpha_biased || !alpha_softplus) {
        fprintf(stderr, "SSM save: alloc failed\n");
        goto cleanup_save;
    }
    
    // === Steps 1-11: Same as wubu_ssm_forward ===
    // (Steps 1-10 are identical, just compute)
    
    // Step 1+2: Fused QKV + gate projection via single Q8_K quantization
    const int n_q8_blocks = (D_MODEL + QK_K - 1) / QK_K;
    const int q8_buf_size = n_q8_blocks * 292;
    uint8_t *ssm_q8_buf = (uint8_t *)malloc(q8_buf_size);
    if (!ssm_q8_buf) { fprintf(stderr, "SSM save: q8 alloc failed\n"); goto cleanup_save; }
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        quantize_row_q8_K(x_s, (block_q8_K *)ssm_q8_buf, D_MODEL);
        quantized_matmul_from_q8(ssm_q8_buf,
            w->attn_qkv_weight_q, w->attn_qkv_weight_type,
            D_MODEL, C, 0, qkv_all + s * C);
        quantized_matmul_from_q8(ssm_q8_buf,
            w->attn_gate_weight_q, w->attn_gate_weight_type,
            D_MODEL, VALUE_DIM, 0, z_all + s * VALUE_DIM);
    }
    free(ssm_q8_buf);
    
    // Step 3: beta/alpha projections
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *beta_s = beta_raw + s * DT_RANK;
        float *alpha_s = alpha_raw + s * DT_RANK;
        for (int j = 0; j < DT_RANK; j++) {
            double sum_b = 0.0, sum_a = 0.0;
            for (int i = 0; i < D_MODEL; i++) {
                sum_b += (double)x_s[i] * (double)w->ssm_beta_weight[i * DT_RANK + j];
                sum_a += (double)x_s[i] * (double)w->ssm_alpha_weight[i * DT_RANK + j];
            }
            beta_s[j] = (float)sum_b;
            alpha_s[j] = (float)sum_a;
        }
    }
    
    // Step 4: sigmoid(beta), softplus(alpha + bias) * ssm_a
    wubu_sigmoid(N * DT_RANK, beta_raw, beta_flat);
    for (int s = 0; s < N; s++)
        for (int j = 0; j < DT_RANK; j++)
            alpha_biased[s * DT_RANK + j] = alpha_raw[s * DT_RANK + j] + w->ssm_dt_bias[j];
    wubu_softplus(N * DT_RANK, alpha_biased, alpha_softplus);
    for (int s = 0; s < N; s++)
        for (int j = 0; j < DT_RANK; j++)
            gate_flat[s * DT_RANK + j] = alpha_softplus[s * DT_RANK + j] * w->ssm_a[j];
    
    // Step 5: Convolution + SiLU
    for (int b = 0; b < B; b++) {
        memcpy(conv_input + b * (T + CONV_KERNEL - 1) * C,
               conv_state + b * (CONV_KERNEL - 1) * C,
               (CONV_KERNEL - 1) * C * sizeof(float));
        memcpy(conv_input + (b * (T + CONV_KERNEL - 1) + (CONV_KERNEL - 1)) * C,
               qkv_all + b * T * C, T * C * sizeof(float));
    }
    wubu_conv1d(B, T, C, CONV_KERNEL, conv_input, w->ssm_conv1d_weight, conv_output);
    wubu_silu(N * C, conv_output, conv_output);
    
    // Update conv_state
    for (int b = 0; b < B; b++) {
        float *ci = conv_input + (b * (T + CONV_KERNEL - 1) + T) * C;
        memcpy(conv_state + b * (CONV_KERNEL - 1) * C, ci,
               (CONV_KERNEL - 1) * C * sizeof(float));
    }
    
    // Step 6: Split into Q, K, V
    for (int s = 0; s < N; s++) {
        const float *cv = conv_output + s * C;
        memcpy(q_conv + s * KEY_DIM, cv, KEY_DIM * sizeof(float));
        memcpy(k_conv + s * KEY_DIM, cv + KEY_DIM, KEY_DIM * sizeof(float));
        memcpy(v_conv + s * VALUE_DIM, cv + 2 * KEY_DIM, VALUE_DIM * sizeof(float));
    }
    
    // Step 7: L2 normalize Q and K
    wubu_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, q_conv, g_ssm_l2_eps, q_norm);
    wubu_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, k_conv, g_ssm_l2_eps, k_norm);
    
    // Step 8+9: Delta net recurrence with per-timestep state saving
    int state_sz = SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int s = b * T + t;
            float *beta_s = beta_flat + s * DT_RANK;
            float *gate_s = gate_flat + s * DT_RANK;
            
            // Save state before this timestep
            if (save && save->states_t) {
                memcpy(save->states_t + (b * (T+1) + t) * state_sz,
                       ssm_state, state_sz * sizeof(float));
            }
            
            #pragma omp parallel for
            for (int vh = 0; vh < SSM_V_HEADS; vh++) {
                int kh = vh / (SSM_V_HEADS / SSM_K_HEADS);
                float bg = beta_s[vh];
                float gg = tgt_safe_expf(gate_s[vh]);  // TGT: safe exp (clamped)
                const float *q_vh = q_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                const float *k_vh = k_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                const float *v_vh = v_conv + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                float *h = ssm_state + (vh * SSM_D_STATE * SSM_D_STATE);
                
                for (int i = 0; i < SSM_D_STATE; i++)
                    for (int j = 0; j < SSM_D_STATE; j++)
                        h[i * SSM_D_STATE + j] *= gg;
                
                float hk[SSM_D_STATE];
                memset(hk, 0, sizeof(hk));
                for (int i = 0; i < SSM_D_STATE; i++)
                    for (int j = 0; j < SSM_D_STATE; j++)
                        hk[i] += h[i * SSM_D_STATE + j] * k_vh[j];
                
                for (int i = 0; i < SSM_D_STATE; i++) {
                    float diff = v_vh[i] - hk[i];
                    for (int j = 0; j < SSM_D_STATE; j++)
                        h[i * SSM_D_STATE + j] += k_vh[j] * diff * bg;
                }
                
                float *out = delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                memset(out, 0, SSM_D_STATE * sizeof(float));
                for (int i = 0; i < SSM_D_STATE; i++)
                    for (int j = 0; j < SSM_D_STATE; j++)
                        out[i] += h[i * SSM_D_STATE + j] * q_vh[j];
            }
        }
    }
    // Save final state
    if (save && save->states_t)
        memcpy(save->states_t + T * state_sz, ssm_state, state_sz * sizeof(float));
    
    // Step 10: Gated normalization
    wubu_silu(N * VALUE_DIM, z_all, z_silu);
    for (int s = 0; s < N; s++) {
        for (int vh = 0; vh < SSM_V_HEADS; vh++) {
            float *out_vh = delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
            float *z_vh = z_silu + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
            float sum_sq = 0.0f;
            for (int i = 0; i < SSM_D_STATE; i++) sum_sq += out_vh[i] * out_vh[i];
            float rms = sqrtf(sum_sq / SSM_D_STATE + 1e-6f);
            float scale = 1.0f / rms;
            for (int i = 0; i < SSM_D_STATE; i++)
                out_vh[i] = (out_vh[i] * scale * w->ssm_norm_weight[i]) * z_vh[i];
        }
    }
    
    // Step 11: Output projection via quantized or F32 matmul
    for (int s = 0; s < N; s++) {
        proj_matmul(delta_out + s * VALUE_DIM, VALUE_DIM, D_MODEL,
                    w->ssm_out_weight, w->ssm_out_weight_q, w->ssm_out_weight_type,
                    output + s * D_MODEL);
    }
    
    // === Save intermediates for backward ===
    if (save) {
        if (save->qkv_all) memcpy(save->qkv_all, qkv_all, N * C * sizeof(float));
        if (save->z_all) memcpy(save->z_all, z_all, N * VALUE_DIM * sizeof(float));
        if (save->beta_raw) memcpy(save->beta_raw, beta_raw, N * DT_RANK * sizeof(float));
        if (save->alpha_raw) memcpy(save->alpha_raw, alpha_raw, N * DT_RANK * sizeof(float));
        if (save->conv_post_silu) memcpy(save->conv_post_silu, conv_output, N * C * sizeof(float));
        if (save->q_conv) memcpy(save->q_conv, q_conv, N * KEY_DIM * sizeof(float));
        if (save->k_conv) memcpy(save->k_conv, k_conv, N * KEY_DIM * sizeof(float));
        if (save->v_conv) memcpy(save->v_conv, v_conv, N * VALUE_DIM * sizeof(float));
        if (save->q_norm) memcpy(save->q_norm, q_norm, N * KEY_DIM * sizeof(float));
        if (save->k_norm) memcpy(save->k_norm, k_norm, N * KEY_DIM * sizeof(float));
        if (save->delta_out) memcpy(save->delta_out, delta_out, N * VALUE_DIM * sizeof(float));
        if (save->z_silu) memcpy(save->z_silu, z_silu, N * VALUE_DIM * sizeof(float));
        if (save->beta_flat) memcpy(save->beta_flat, beta_flat, N * DT_RANK * sizeof(float));
        if (save->gate_flat) memcpy(save->gate_flat, gate_flat, N * DT_RANK * sizeof(float));
        if (save->conv_state_copy) memcpy(save->conv_state_copy, conv_state, B * (CONV_KERNEL-1) * C * sizeof(float));
    }
    
cleanup_save:
    free(qkv_all); free(z_all);
    free(beta_raw); free(alpha_raw);
    free(conv_input); free(conv_output);
    free(q_conv); free(k_conv); free(v_conv);
    free(q_norm); free(k_norm);
    free(delta_out); free(z_silu);
    free(beta_flat); free(gate_flat);
    free(alpha_biased); free(alpha_softplus);
}

// ============================================================
// Poincaré SSM Layer Forward Pass
// ============================================================

void wubu_poincare_ssm_forward(const float *x, int B, int T,
                               const ssm_layer_weights *w,
                               float *ssm_state,
                               float *conv_state,
                               float R,
                               float *output) {
    // Same as wubu_ssm_forward() except the recurrence step (Step 9)
    // uses Möbius operations instead of Euclidean linear algebra.
    //
    // Euclidean:  h[t] = h[t-1] * exp(gate) + (k_vh ⊗ (v - h[t-1] @ k_vh)) * beta
    // Poincaré:  h[t] = mobius_add(scalar_mul(exp(gate), h[t-1]),
    //                               exp_map(k_vh ⊗ (log_map(v, R) - log_map(h[t-1] @ k_vh)) * beta, R))
    //
    // Steps 1-8, 10-11 are IDENTICAL to Euclidean. Only Step 9 differs.
    // We reuse all preceding steps by copying the Euclidean forward code,
    // and modifying the recurrence section.
    
    // ... [Steps 1-8 are identical to Euclidean above] ...
    // For now, we call the Euclidean forward to get conv output and projections,
    // then rewrite the recurrence separately.
    
    // HACK: We copy most of the Euclidean code here. A cleaner design would
    // extract Steps 1-8 as shared helpers, but for Phase 2.2 this keeps things explicit.
    
    const int N = B * T;
    const int C = KEY_DIM * 2 + VALUE_DIM;  // = 8192
    
    // Allocate (same as Euclidean)
    float *qkv_all = (float *)malloc(N * C * sizeof(float));
    float *z_all = (float *)malloc(N * VALUE_DIM * sizeof(float));
    float *beta_raw = (float *)malloc(N * DT_RANK * sizeof(float));
    float *alpha_raw = (float *)malloc(N * DT_RANK * sizeof(float));
    float *conv_input = (float *)malloc(B * (T + CONV_KERNEL - 1) * C * sizeof(float));
    float *conv_output = (float *)malloc(N * C * sizeof(float));
    float *q_conv = (float *)malloc(N * KEY_DIM * sizeof(float));
    float *k_conv = (float *)malloc(N * KEY_DIM * sizeof(float));
    float *v_conv = (float *)malloc(N * VALUE_DIM * sizeof(float));
    float *q_norm = (float *)malloc(N * KEY_DIM * sizeof(float));
    float *k_norm = (float *)malloc(N * KEY_DIM * sizeof(float));
    float *delta_out = (float *)malloc(N * VALUE_DIM * sizeof(float));
    float *z_silu = (float *)malloc(N * VALUE_DIM * sizeof(float));
    
    if (!qkv_all || !z_all || !beta_raw || !alpha_raw || !conv_input ||
        !conv_output || !q_conv || !k_conv || !v_conv || !q_norm || !k_norm ||
        !delta_out || !z_silu) {
        fprintf(stderr, "Poincaré SSM forward: allocation failed\n");
        free(qkv_all); free(z_all); free(beta_raw); free(alpha_raw);
        free(conv_input); free(conv_output);
        free(q_conv); free(k_conv); free(v_conv);
        free(q_norm); free(k_norm);
        free(delta_out); free(z_silu);
        return;
    }
    
    // ========== Steps 1-8: IDENTICAL to Euclidean ==========
    
    // Step 1+2: Fused QKV + gate projection via single Q8_K quantization
    // Both projections use the same input x[s], so quantize once and reuse
    const int n_q8_blocks = (D_MODEL + QK_K - 1) / QK_K;
    const int q8_buf_size = n_q8_blocks * 292;  // Q8K_BLOCK_SIZE
    uint8_t *ssm_q8_buf = (uint8_t *)malloc(q8_buf_size);
    block_q8_K *ssm_q8 = (block_q8_K *)ssm_q8_buf;
    if (!ssm_q8_buf) { fprintf(stderr, "SSM forward: q8 alloc failed\n"); goto cleanup_p; }
    
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        // Quantize once
        quantize_row_q8_K(x_s, ssm_q8, D_MODEL);
        
        // Reuse for both projections
        quantized_matmul_from_q8(ssm_q8_buf,
            w->attn_qkv_weight_q, w->attn_qkv_weight_type,
            D_MODEL, C, 0, qkv_all + s * C);
        quantized_matmul_from_q8(ssm_q8_buf,
            w->attn_gate_weight_q, w->attn_gate_weight_type,
            D_MODEL, VALUE_DIM, 0, z_all + s * VALUE_DIM);
    }
    free(ssm_q8_buf);
    
    // Step 3: beta/alpha projections
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *beta_s = beta_raw + s * DT_RANK;
        float *alpha_s = alpha_raw + s * DT_RANK;
        for (int j = 0; j < DT_RANK; j++) {
            float sum_b = 0.0f, sum_a = 0.0f;
            for (int i = 0; i < D_MODEL; i++) {
                sum_b += x_s[i] * w->ssm_beta_weight[i + j * D_MODEL];
                sum_a += x_s[i] * w->ssm_alpha_weight[i + j * D_MODEL];
            }
            beta_s[j] = sum_b;
            alpha_s[j] = sum_a;
        }
    }
    
    // Step 4: Compute beta and gate (identical to Euclidean)
    float *beta_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    float *gate_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    if (!beta_flat || !gate_flat) {
        fprintf(stderr, "Poincaré SSM forward: beta/gate alloc failed\n");
        goto cleanup_p;
    }
    
    wubu_sigmoid(N * DT_RANK, beta_raw, beta_flat);
    
    float *alpha_biased = (float *)malloc(N * DT_RANK * sizeof(float));
    float *alpha_softplus = (float *)malloc(N * DT_RANK * sizeof(float));
    if (!alpha_biased || !alpha_softplus) goto cleanup_p;
    
    for (int s = 0; s < N; s++) {
        for (int j = 0; j < DT_RANK; j++) {
            alpha_biased[s * DT_RANK + j] = alpha_raw[s * DT_RANK + j] + w->ssm_dt_bias[j];
        }
    }
    wubu_softplus(N * DT_RANK, alpha_biased, alpha_softplus);
    
    for (int s = 0; s < N; s++) {
        for (int j = 0; j < DT_RANK; j++) {
            gate_flat[s * DT_RANK + j] = alpha_softplus[s * DT_RANK + j] * w->ssm_a[j];
        }
    }
    
    // Steps 5-8: Conv, split, L2 norm (identical to Euclidean)
    for (int b = 0; b < B; b++) {
        memcpy(conv_input + b * (T + CONV_KERNEL - 1) * C,
               conv_state + b * (CONV_KERNEL - 1) * C,
               (CONV_KERNEL - 1) * C * sizeof(float));
        memcpy(conv_input + (b * (T + CONV_KERNEL - 1) + (CONV_KERNEL - 1)) * C,
               qkv_all + b * T * C,
               T * C * sizeof(float));
    }
    wubu_conv1d(B, T, C, CONV_KERNEL, conv_input, w->ssm_conv1d_weight, conv_output);
    wubu_silu(N * C, conv_output, conv_output);
    
    for (int b = 0; b < B; b++) {
        float *ci = conv_input + (b * (T + CONV_KERNEL - 1) + T) * C;
        memcpy(conv_state + b * (CONV_KERNEL - 1) * C, ci,
               (CONV_KERNEL - 1) * C * sizeof(float));
    }
    
    for (int s = 0; s < N; s++) {
        const float *cv = conv_output + s * C;
        memcpy(q_conv + s * KEY_DIM, cv, KEY_DIM * sizeof(float));
        memcpy(k_conv + s * KEY_DIM, cv + KEY_DIM, KEY_DIM * sizeof(float));
        memcpy(v_conv + s * VALUE_DIM, cv + 2 * KEY_DIM, VALUE_DIM * sizeof(float));
    }
    
    wubu_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, q_conv, g_ssm_l2_eps, q_norm);
    wubu_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, k_conv, g_ssm_l2_eps, k_norm);
    
    // ========== Step 9: POINCARÉ RECURRENCE (differs from Euclidean) ==========
    
    int repeat_factor = SSM_V_HEADS / SSM_K_HEADS;  // 2
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int s = b * T + t;
            float *beta_s = beta_flat + s * DT_RANK;
            float *gate_s = gate_flat + s * DT_RANK;
            
            for (int vh = 0; vh < SSM_V_HEADS; vh++) {
                int kh = vh / repeat_factor;
                
                float bg = beta_s[vh];
                float gg = tgt_safe_expf(gate_s[vh]);  // TGT: safe exp (clamped)  // scalar for Möbius multiplication
                
                const float *q_vh = q_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                const float *k_vh = k_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                const float *v_vh = v_conv + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                
                float *h = ssm_state + (vh * SSM_D_STATE * SSM_D_STATE);
                
                // === Poincaré recurrence ===
                //
                // Instead of:
                //   h = h * gg  (Euclidean scalar multiply)
                //   hk = h @ k  (Euclidean matvec)
                //   diff = v - hk  (Euclidean subtract)
                //   h += k_vh ⊗ diff * bg  (Euclidean outer product update)
                //
                // We do:
                //   1. Decay state: h_decayed = scalar_mul(gg, h) in Poincaré ball
                //   2. Map to tangent: hk_tan = log_map(h, R) @ k  (inner in tangent space)
                //   3. Map V to tangent: v_tan = log_map(v, R)
                //   4. Diff in tangent: diff_tan = v_tan - hk_tan
                //   5. Update in tangent: update_tan = k ⊗ diff_tan * bg (outer product)
                //   6. Map back: update_ball = exp_map(update_tan, R)
                //   7. h[t] = mobius_add(h_decayed, update_ball)
                
                // Temporary buffers
                float temp_h[SSM_D_STATE];
                float temp_k[SSM_D_STATE];
                float temp_v[SSM_D_STATE];
                float hk_tan[SSM_D_STATE];
                float v_tan[SSM_D_STATE];
                float diff_tan[SSM_D_STATE];
                float update_ball[SSM_D_STATE];
                
                // Step 9a: Decay state in Poincaré ball
                // h_decayed = gg ⊗ h  (Möbius scalar multiplication)
                // h is a matrix [SSM_D_STATE x SSM_D_STATE]; we decay each row
                // For simplicity, treat h rows as independent ball vectors
                for (int i = 0; i < SSM_D_STATE; i++) {
                    const float *h_row = h + i * SSM_D_STATE;
                    wubu_mobius_scalar_mul(gg, h_row, SSM_D_STATE, R, temp_h);
                    memcpy(h + i * SSM_D_STATE, temp_h, SSM_D_STATE * sizeof(float));
                }
                
                // Step 9b: Predict v from h using tangent-space inner product
                // Map each row of h to tangent space, compute inner with k
                memset(hk_tan, 0, SSM_D_STATE * sizeof(float));
                for (int i = 0; i < SSM_D_STATE; i++) {
                    const float *h_row = h + i * SSM_D_STATE;
                    wubu_log_map(h_row, SSM_D_STATE, R, temp_h);
                    // hk_tan[i] = log_map(h[i,:], R) · k  (tangent inner product)
                    hk_tan[i] = wubu_dot(temp_h, k_vh, SSM_D_STATE);
                }
                
                // Step 9c: Map V to tangent space
                wubu_log_map(v_vh, SSM_D_STATE, R, v_tan);
                
                // Step 9d: Diff in tangent space
                for (int i = 0; i < SSM_D_STATE; i++) {
                    diff_tan[i] = v_tan[i] - hk_tan[i];
                }
                
                // Step 9e: Outer product update in tangent space, then map to ball
                memset(update_ball, 0, SSM_D_STATE * sizeof(float));
                for (int i = 0; i < SSM_D_STATE; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < SSM_D_STATE; j++) {
                        sum += k_vh[i] * diff_tan[j] * bg;
                    }
                    update_ball[i] = sum;
                }
                
                // Step 9f: Apply update — Möbius add instead of Euclidean add
                // update_ball is in tangent space, need to map it to ball
                float upd_ball[SSM_D_STATE];
                wubu_exp_map(update_ball, SSM_D_STATE, R, upd_ball);
                
                // Step 9g: h[t] = h[t-1] ⊕ update_ball  (Möbius addition)
                for (int i = 0; i < SSM_D_STATE; i++) {
                    const float *h_row = h + i * SSM_D_STATE;
                    wubu_mobius_add(h_row, upd_ball, SSM_D_STATE, R, temp_h);
                    memcpy(h + i * SSM_D_STATE, temp_h, SSM_D_STATE * sizeof(float));
                }
                
                // Step 9h: output = h @ q  (still Euclidean — Q is in tangent space)
                float *out = delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                memset(out, 0, SSM_D_STATE * sizeof(float));
                for (int i = 0; i < SSM_D_STATE; i++) {
                    const float *h_row = h + i * SSM_D_STATE;
                    for (int j = 0; j < SSM_D_STATE; j++) {
                        out[i] += h_row[j] * q_vh[j];
                    }
                }
            }
        }
    }
    
    // ========== Steps 10-11: IDENTICAL to Euclidean ==========
    
    wubu_silu(N * VALUE_DIM, z_all, z_silu);
    
    for (int s = 0; s < N; s++) {
        for (int vh = 0; vh < SSM_V_HEADS; vh++) {
            float *out_vh = delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
            float *z_vh = z_silu + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
            float sum_sq = 0.0f;
            for (int i = 0; i < SSM_D_STATE; i++) sum_sq += out_vh[i] * out_vh[i];
            float rms = sqrtf(sum_sq / SSM_D_STATE + 1e-6f);
            float scale = 1.0f / rms;
            for (int i = 0; i < SSM_D_STATE; i++) {
                out_vh[i] = (out_vh[i] * scale * w->ssm_norm_weight[i]) * z_vh[i];
            }
        }
    }
    
    for (int s = 0; s < N; s++) {
        proj_matmul(delta_out + s * VALUE_DIM, VALUE_DIM, D_MODEL,
                    w->ssm_out_weight, w->ssm_out_weight_q, w->ssm_out_weight_type,
                    output + s * D_MODEL);
    }
    
cleanup_p:
    free(qkv_all); free(z_all); free(beta_raw); free(alpha_raw);
    free(conv_input); free(conv_output);
    free(q_conv); free(k_conv); free(v_conv);
    free(q_norm); free(k_norm);
    free(delta_out); free(z_silu);
    free(beta_flat); free(gate_flat);
    free(alpha_biased); free(alpha_softplus);
}

// ============================================================
// GQA Layer Forward Pass
// ============================================================

void wubu_gqa_forward(const float *x, int B, int T,
                      const gqa_layer_weights *w,
                      float *output,
                      const void *k_cache, const void *v_cache, int cache_len,
                      void *k_out, void *v_out) {
    const int N = B * T;
    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;  // 4096
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;  // 512
    
    // Allocate
    // Q_full: [N, q_dim*2] = [N, 8192] — first q_dim=4096 Q, next 4096 gate
    float *Q_full = (float *)malloc(N * q_dim * 2 * sizeof(float));
    float *gate = (float *)malloc(N * q_dim * sizeof(float));
    float *K = (float *)malloc(N * kv_dim * sizeof(float));
    float *V = (float *)malloc(N * kv_dim * sizeof(float));
    float *Q_norm = (float *)malloc(N * q_dim * sizeof(float));
    float *K_norm = (float *)malloc(N * kv_dim * sizeof(float));
    float *attn_out = (float *)malloc(N * q_dim * sizeof(float));
    
    if (!Q_full || !gate || !K || !V || !Q_norm || !K_norm || !attn_out) {
        fprintf(stderr, "GQA forward: allocation failed\n");
        free(Q_full); free(gate); free(K); free(V);
        free(Q_norm); free(K_norm); free(attn_out);
        return;
    }
    
    // Step 1: Q + gate fused projection — batched for all N tokens
    // Weight read ONCE from RAM via quantized_matmul_batched
    quantized_matmul_batched(x,
        w->attn_q_weight_q, w->attn_q_weight_type,
        D_MODEL, q_dim * 2, 0, N, Q_full);
    
    // Extract gate from Q_full — INTERLEAVED per-head layout:
    // Q_full layout: [Q_h0(256) | gate_h0(256) | Q_h1(256) | gate_h1(256) | ...]
    for (int s = 0; s < N; s++) {
        int q_offset = s * q_dim * 2;
        for (int h = 0; h < GQA_Q_HEADS; h++) {
            for (int j = 0; j < GQA_HEAD_DIM; j++) {
                int qf_idx = q_offset + h * (2 * GQA_HEAD_DIM) + GQA_HEAD_DIM + j;
                int g_idx  = s * q_dim + h * GQA_HEAD_DIM + j;
                gate[g_idx] = Q_full[qf_idx];
            }
        }
    }
    
    // K projection — batched
    quantized_matmul_batched(x,
        w->attn_k_weight_q, w->attn_k_weight_type,
        D_MODEL, kv_dim, 0, N, K);
    
    // V projection — batched
    quantized_matmul_batched(x,
        w->attn_v_weight_q, w->attn_v_weight_type,
        D_MODEL, kv_dim, 0, N, V);
    
    // NaN guard: replace any NaN/Inf in Q_full, K, V with 0
    // Only check when debug is enabled; normally matmul produces no NaN
    const char *gqa_dd = getenv("DUMP_GQA_DEBUG");
    if (gqa_dd) {
        for (int i = 0; i < N * q_dim * 2; i++)
            if (isnan(Q_full[i]) || isinf(Q_full[i])) Q_full[i] = 0.0f;
        for (int i = 0; i < N * q_dim; i++)
            if (isnan(gate[i]) || isinf(gate[i])) gate[i] = 0.0f;
        for (int i = 0; i < N * kv_dim; i++) {
            if (isnan(K[i]) || isinf(K[i])) K[i] = 0.0f;
            if (isnan(V[i]) || isinf(V[i])) V[i] = 0.0f;
        }
    }
    
    // Step 4: Q/K RMSNorm
    // Q_full has [N, q_dim*2] = [N, 8192] with INTERLEAVED per-head layout:
    //   [Q_h0(256) | gate_h0(256) | Q_h1(256) | gate_h1(256) | ...]
    // Extract Q only (skip the gate values) for normalization
    float *Q_only = (float *)malloc(N * q_dim * sizeof(float));
    if (!Q_only) { free(Q_full); free(gate); free(K); free(V); free(Q_norm); free(K_norm); free(attn_out); return; }
    for (int s = 0; s < N; s++) {
        int q_offset = s * q_dim * 2;
        for (int h = 0; h < GQA_Q_HEADS; h++) {
            for (int j = 0; j < GQA_HEAD_DIM; j++) {
                Q_only[s * q_dim + h * GQA_HEAD_DIM + j] = Q_full[q_offset + h * (2 * GQA_HEAD_DIM) + j];
            }
        }
    }
    wubu_rms_norm(B, T * GQA_Q_HEADS, GQA_HEAD_DIM, Q_only, w->attn_q_norm_weight, 1e-6f, Q_norm);
    free(Q_only);
    
    // K RMSNorm — K has [N, kv_dim] = [N, KV_HEADS * HEAD_DIM]
    // Data layout: K[(b*T+t)*kv_dim + h*HEAD_DIM + i]
    // RMSNorm sees [B, T*KV_HEADS, HEAD_DIM] same layout
    wubu_rms_norm(B, T * GQA_KV_HEADS, GQA_HEAD_DIM, K, w->attn_k_norm_weight, 1e-6f, K_norm);
    
    // Step 4: IMRoPE (Interleaved MultiRoPE) with pre-computed theta table
    // Qwen3.6: rope.dimension_sections=[11,11,10,0], rope.dimension_count=64, rope.freq_base=10000000.0
    // For text-only generation: all position IDs equal, reduces to standard RoPE
    // Apply to first N_ROT=64 dims of each head for both Q and K
    // N_ROT = 64 = 32 pairs. Standard RoPE: (x_2i, x_{2i+1}) * rotation_matrix(theta_i)
    // theta_i(pos) = pos * freq_base^{-2i/N_ROT}
    //
    // RoPE extrapolation (Qwen2.5-1M §3.1):
    //   ROPE_SCALE_FACTOR=0.25 extends 64K→256K (4x)
    //   theta_i(pos) = (pos * scale) * freq_base^{-2i/N_ROT}
    //
    // Optimization: pre-compute theta_i table (static, once) to eliminate powf() calls
    {
        const int n_rot = 64;  // rope.dimension_count
        const float freq_base = 10000000.0f;
        const int q_rot_pairs = n_rot / 2;  // 32
        const char *rope_scale_env = getenv("ROPE_SCALE_FACTOR");
        const float scale_factor = rope_scale_env ? atof(rope_scale_env) : 1.0f;
        
        // Static pre-computed theta_i table (freq_base^{-2i/n_rot})
        // Computed once, reused across all forward calls
        static float rope_theta[32];
        static int rope_theta_ready = 0;
        if (!rope_theta_ready) {
            for (int i = 0; i < q_rot_pairs; i++)
                rope_theta[i] = powf(freq_base, -2.0f * i / (float)n_rot);
            rope_theta_ready = 1;
        }
        
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float pos = (float)(b * T + t);
                float pos_scale = pos * scale_factor;
                
                // Apply to all 16 Q heads — fully parallel
                #pragma omp parallel for
                for (int h = 0; h < GQA_Q_HEADS; h++) {
                    float *q_h = Q_norm + ((b * T + t) * GQA_Q_HEADS + h) * GQA_HEAD_DIM;
                    for (int i = 0; i < q_rot_pairs; i++) {
                        float theta = pos_scale * rope_theta[i];
                        float cos_t = cosf(theta);
                        float sin_t = sinf(theta);
                        float x0 = q_h[2*i];
                        float x1 = q_h[2*i+1];
                        q_h[2*i]   = x0 * cos_t - x1 * sin_t;
                        q_h[2*i+1] = x0 * sin_t + x1 * cos_t;
                    }
                }
                
                // Apply to all 2 KV heads
                for (int h = 0; h < GQA_KV_HEADS; h++) {
                    float *k_h = K_norm + ((b * T + t) * GQA_KV_HEADS + h) * GQA_HEAD_DIM;
                    for (int i = 0; i < q_rot_pairs; i++) {
                        float theta = pos_scale * rope_theta[i];
                        float cos_t = cosf(theta);
                        float sin_t = sinf(theta);
                        float x0 = k_h[2*i];
                        float x1 = k_h[2*i+1];
                        k_h[2*i]   = x0 * cos_t - x1 * sin_t;
                        k_h[2*i+1] = x0 * sin_t + x1 * cos_t;
                    }
                }
            }
        }
    }
    
    // Step 5: GQA Attention with KV cache
    // Read directly from cache + new tokens — no O(T) copy
    int total_kv = cache_len + N;  // total positions to attend over
    // attn_weights on heap (too large for stack at 256k context)
    
    // NSA-style sparse attention for long contexts (DSA pattern from DeepSeek-V3.2)
    // S(i) = {i} ∪ local_window(i, w) ∪ global_positions(i, g)
    // Reduces O(L²) to O(L·(w+g)) per layer
    int use_sparse = getenv("USE_SPARSE_ATTN") != NULL;
    int sparse_w = getenv("SPARSE_W") ? atoi(getenv("SPARSE_W")) : 512;   // local window size
    int sparse_g = getenv("SPARSE_G") ? atoi(getenv("SPARSE_G")) : 128;   // global positions count
    int sparse_min_len = getenv("SPARSE_MIN") ? atoi(getenv("SPARSE_MIN")) : 512;  // min ctx for sparse (was 4096, lowered May 27: GQA is not bottleneck, but helps at >2K)
    
    // Pre-allocate sparse index buffer (stack for small, heap for extreme)
    int max_sparse = sparse_w + sparse_g + 1;  // window + global + self
    int sparse_stack[2048];  // up to 2048 ints = 8KB stack (max realistic: 512+128+1=641)
    int *sparse_buf = NULL;
    if (use_sparse && total_kv >= sparse_min_len) {
        if (max_sparse <= 2048) {
            sparse_buf = sparse_stack;
        } else {
            sparse_buf = (int *)malloc((size_t)max_sparse * sizeof(int));
            if (!sparse_buf) use_sparse = 0;
        }
    } else {
        use_sparse = 0;
    }
    
    // 16 Q heads, 2 KV heads. Each KV head serves 8 Q heads.
    float scale = 1.0f / sqrtf(GQA_HEAD_DIM);
    
    // AVX2 horizontal sum helper (inlined)
#ifdef __AVX2__
    #define HSUM256(v) ({ \
        __m128 vlow  = _mm256_castps256_ps128(v); \
        __m128 vhigh = _mm256_extractf128_ps(v, 1); \
        __m128 s     = _mm_add_ps(vlow, vhigh); \
        __m128 shuf  = _mm_movehdup_ps(s); \
        __m128 sum   = _mm_add_ps(s, shuf); \
        shuf         = _mm_movehl_ps(shuf, sum); \
        sum          = _mm_add_ss(sum, shuf); \
        _mm_cvtss_f32(sum); \
    })
#endif
    
    // Tiled GQA attention: for each KV position, read K cache ONCE per KV head,
    // then compute dot products with all Q heads sharing that KV head.
    // This reduces K cache reads by 8× at 256k context.
    // Each tile processes KV head 0 (Q heads 0-7) or KV head 1 (Q heads 8-15).
    
    // Pre-allocate attention weight buffer once (removes per-position malloc/free at 512k)
    int max_attend = use_sparse ? max_sparse : (total_kv < sparse_min_len ? total_kv : max_sparse);
    float *all_attn_w = NULL;
    if (max_attend > 0) {
        all_attn_w = (float *)malloc((size_t)GQA_Q_HEADS * max_attend * sizeof(float));
        if (!all_attn_w) {
            fprintf(stderr, "GQA: attn_w pre-alloc failed (%zu)\n", (size_t)GQA_Q_HEADS * max_attend * 4);
            goto gqa_alloc_fail;
        }
    }
    
    for (int b = 0; b < B; b++) {
        for (int t_q = 0; t_q < T; t_q++) {
            int global_t = cache_len + b * T + t_q;
            int attend_len = global_t + 1;
            
            // Build sparse attendance set for this query position
            int sparse_count = attend_len;  // default: full attention
            if (use_sparse) {
                // S(i) = local_window(i, w) ∪ global_positions(i, g)
                // Local window: last sparse_w positions
                int win_start = global_t - sparse_w + 1;
                if (win_start < 0) win_start = 0;
                int win_count = global_t - win_start + 1;
                // Global positions: uniformly spaced over [0, win_start)
                int hist_len = win_start;  // positions before window
                int g_step = (hist_len > sparse_g) ? (hist_len / sparse_g) : 1;
                if (g_step < 1) g_step = 1;
                int g_count = hist_len / g_step;
                if (g_count > sparse_g) g_count = sparse_g;
                if (sparse_w + g_count + 1 > max_sparse) { /* should not happen */ }
                sparse_count = 0;
                // Global positions (evenly spaced from history)
                for (int i = 0; i < g_count; i++) {
                    sparse_buf[sparse_count++] = i * g_step;
                }
                // Local window positions
                for (int i = win_start; i <= global_t; i++) {
                    sparse_buf[sparse_count++] = i;
                }
            }
            
            // Pre-allocate per-Q-head attention weights (sparse or full)
            // Buffer was pre-allocated above the position loop — just use it
            // (all_attn_w has capacity for max_attend positions)
            
            #pragma omp parallel for if(attend_len > 64)
            for (int _tk = 0; _tk < sparse_count; _tk++) {
                int t_k = use_sparse ? sparse_buf[_tk] : _tk;
                float k_buf0[GQA_HEAD_DIM], k_buf1[GQA_HEAD_DIM];
                const float *k0, *k1;
                
                // Read K cache for both KV heads (or from new tokens)
                if (t_k < cache_len) {
                    int64_t off0 = (int64_t)t_k * GQA_KV_HEADS + 0;
                    int64_t off1 = (int64_t)t_k * GQA_KV_HEADS + 1;
                    kv_cache_read_head(k_cache, off0 * GQA_HEAD_DIM, k_buf0, GQA_HEAD_DIM);
                    kv_cache_read_head(k_cache, off1 * GQA_HEAD_DIM, k_buf1, GQA_HEAD_DIM);
                    k0 = k_buf0; k1 = k_buf1;
                } else {
                    int new_idx = t_k - cache_len;
                    k0 = K_norm + (new_idx * GQA_KV_HEADS + 0) * GQA_HEAD_DIM;
                    k1 = K_norm + (new_idx * GQA_KV_HEADS + 1) * GQA_HEAD_DIM;
                }
                
                // Compute Q·K for all 16 Q heads at this position
                for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
                    const float *k_vec = (h_q < 8) ? k0 : k1;
                    const float *q_vec = Q_norm + ((b * T + t_q) * GQA_Q_HEADS + h_q) * GQA_HEAD_DIM;
                    float score;
                    
#ifdef __AVX2__
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    for (int i = 0; i < GQA_HEAD_DIM; i += 32) {
                        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(q_vec + i),     _mm256_loadu_ps(k_vec + i),     acc0);
                        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(q_vec + i + 8), _mm256_loadu_ps(k_vec + i + 8), acc1);
                        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(q_vec + i + 16),_mm256_loadu_ps(k_vec + i + 16),acc2);
                        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(q_vec + i + 24),_mm256_loadu_ps(k_vec + i + 24),acc3);
                    }
                    __m256 tot = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                    score = HSUM256(tot) * scale;
#else
                    score = 0.0f;
                    for (int i = 0; i < GQA_HEAD_DIM; i++)
                        score += q_vec[i] * k_vec[i];
                    score *= scale;
#endif
                    all_attn_w[(size_t)h_q * sparse_count + _tk] = score;
                }
            }
            
            // Now softmax and V weighted sum per Q head
            #pragma omp parallel for
            for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
                int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);
                float *out_vec = attn_out + ((b * T + t_q) * GQA_Q_HEADS + h_q) * GQA_HEAD_DIM;
                memset(out_vec, 0, GQA_HEAD_DIM * sizeof(float));
                
                // Find max score for this head
                float max_score = -1e30f;
                for (int _tk = 0; _tk < sparse_count; _tk++) {
                    float s = all_attn_w[(size_t)h_q * sparse_count + _tk];
                    if (s > max_score) max_score = s;
                }
                
                // Softmax
                float sum_exp = 0.0f;
                for (int _tk = 0; _tk < sparse_count; _tk++) {
                    float s = expf(all_attn_w[(size_t)h_q * sparse_count + _tk] - max_score);
                    all_attn_w[(size_t)h_q * sparse_count + _tk] = s;
                    sum_exp += s;
                }
                float inv_sum = 1.0f / sum_exp;
                for (int _tk = 0; _tk < sparse_count; _tk++) {
                    all_attn_w[(size_t)h_q * sparse_count + _tk] *= inv_sum;
                }
                
                // Weighted sum of V
                for (int _tk = 0; _tk < sparse_count; _tk++) {
                    int t_k = use_sparse ? sparse_buf[_tk] : _tk;
                    float v_buf[GQA_HEAD_DIM];
                    const float *v_vec;
                    if (t_k < cache_len) {
                        int64_t off = (int64_t)t_k * GQA_KV_HEADS + h_kv;
                        kv_cache_read_head(v_cache, off * GQA_HEAD_DIM, v_buf, GQA_HEAD_DIM);
                        v_vec = v_buf;
                    } else {
                        int new_idx = t_k - cache_len;
                        v_vec = V + (new_idx * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                    }
                    float a = all_attn_w[(size_t)h_q * sparse_count + _tk];
#ifdef __AVX2__
                    __m256 a_v = _mm256_set1_ps(a);
                    for (int i = 0; i < GQA_HEAD_DIM; i += 8) {
                        __m256 v = _mm256_loadu_ps(v_vec + i);
                        __m256 o = _mm256_loadu_ps(out_vec + i);
                        o = _mm256_fmadd_ps(a_v, v, o);
                        _mm256_storeu_ps(out_vec + i, o);
                    }
#else
                    for (int i = 0; i < GQA_HEAD_DIM; i++) {
                        out_vec[i] += a * v_vec[i];
                    }
#endif
                }
                
                // NaN debug (only check first head, rarely triggers)
                // (q_vec is defined in the tiled Q·K loop above, skip NaN debug for now)
                
                // Free per-head attention weights (all stored in all_attn_w)
            }
        }
    }
    // Free pre-allocated attention weight buffer
    free(all_attn_w);
    all_attn_w = NULL;
    
    // Step 6: Gate (sigmoid)
    float *gate_sig = (float *)malloc(N * q_dim * sizeof(float));
    if (!gate_sig) { free(gate_sig); free(Q_full); free(gate); free(K); free(V); free(Q_norm); free(K_norm); free(attn_out); return; }
    wubu_sigmoid(N * q_dim, gate, gate_sig);
    
    for (int i = 0; i < N * q_dim; i++) {
        attn_out[i] *= gate_sig[i];
    }
    
    // Step 7: Output projection via quantized or F32 matmul
    for (int s = 0; s < N; s++) {
        proj_matmul(attn_out + s * q_dim, q_dim, D_MODEL,
                    w->attn_output_weight, w->attn_output_weight_q, w->attn_output_weight_type,
                    output + s * D_MODEL);
    }
    
    // Copy K_norm and V to output buffers for KV cache (stored as F16 if enabled)
    if (k_out && v_out) {
        kv_cache_write_head(k_out, 0, K_norm, N * kv_dim);
        kv_cache_write_head(v_out, 0, V, N * kv_dim);
    }
    
    free(Q_full);
    free(gate);
    free(K);
    free(V);
    free(Q_norm);
    free(K_norm);
    free(attn_out);
    free(gate_sig);
    if (sparse_buf && sparse_buf != sparse_stack) free(sparse_buf);
    return;

gqa_alloc_fail:
    free(all_attn_w);
    if (sparse_buf && sparse_buf != sparse_stack) free(sparse_buf);
    free(Q_full);
    free(gate);
    free(K);
    free(V);
    free(Q_norm);
    free(K_norm);
    free(attn_out);
}

// GQA forward with intermediate saving (for backward)
void wubu_gqa_forward_save(const float *x, int B, int T,
                           const gqa_layer_weights *w,
                           float *output,
                           gqa_fwd_save_t *save)
{
    const int N = B * T;
    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
    
    float *Q_full = (float *)malloc(N * q_dim * 2 * sizeof(float));
    float *gate = (float *)malloc(N * q_dim * sizeof(float));
    float *K = (float *)malloc(N * kv_dim * sizeof(float));
    float *V = (float *)malloc(N * kv_dim * sizeof(float));
    float *Q_norm = (float *)malloc(N * q_dim * sizeof(float));
    float *K_norm = (float *)malloc(N * kv_dim * sizeof(float));
    float *attn_out = (float *)malloc(N * q_dim * sizeof(float));
    
    if (!Q_full || !gate || !K || !V || !Q_norm || !K_norm || !attn_out) {
        fprintf(stderr, "GQA save: alloc failed\n");
        goto cleanup_gqa_save;
    }
    
    // === Steps 1-7: Same as wubu_gqa_forward ===
    
    // Step 1: Q + gate fused projection — batched
    quantized_matmul_batched(x,
        w->attn_q_weight_q, w->attn_q_weight_type,
        D_MODEL, q_dim * 2, 0, N, Q_full);
    
    // Extract gate from Q_full
    for (int s = 0; s < N; s++) {
        int q_offset = s * q_dim * 2;
        for (int j = 0; j < q_dim; j++)
            gate[s * q_dim + j] = Q_full[q_offset + q_dim + j];
    }
    
    // Steps 2-3: K and V projections — batched
    quantized_matmul_batched(x,
        w->attn_k_weight_q, w->attn_k_weight_type,
        D_MODEL, kv_dim, 0, N, K);
    quantized_matmul_batched(x,
        w->attn_v_weight_q, w->attn_v_weight_type,
        D_MODEL, kv_dim, 0, N, V);
    
    // Step 3: Q/K RMSNorm
    float *Q_only = (float *)malloc(N * q_dim * sizeof(float));
    for (int s = 0; s < N; s++)
        memcpy(Q_only + s * q_dim, Q_full + s * q_dim * 2, q_dim * sizeof(float));
    wubu_rms_norm(B, T * GQA_Q_HEADS, GQA_HEAD_DIM, Q_only, w->attn_q_norm_weight, 1e-6f, Q_norm);
    free(Q_only);
    wubu_rms_norm(B, T * GQA_KV_HEADS, GQA_HEAD_DIM, K, w->attn_k_norm_weight, 1e-6f, K_norm);
    
    // Step 4: (RoPE skipped)
    
    // Step 5: GQA Attention
    float scale = 1.0f / sqrtf(GQA_HEAD_DIM);
    for (int b = 0; b < B; b++) {
        for (int t_q = 0; t_q < T; t_q++) {
            for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
                int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);
                const float *q_vec = Q_norm + ((b * T + t_q) * GQA_Q_HEADS + h_q) * GQA_HEAD_DIM;
                float *out_vec = attn_out + ((b * T + t_q) * GQA_Q_HEADS + h_q) * GQA_HEAD_DIM;
                memset(out_vec, 0, GQA_HEAD_DIM * sizeof(float));
                
                float attn_weights[4096];
                float max_score = -1e30f;
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    const float *k_vec = K_norm + ((b * T + t_k) * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                    float score = 0.0f;
                    for (int i = 0; i < GQA_HEAD_DIM; i++)
                        score += q_vec[i] * k_vec[i];
                    score *= scale;
                    // TGT: wrap attention score to prevent overflow — REMOVED: breaks softmax (inverts weights)
                    //score = tgt_wrap(score);
                    attn_weights[t_k] = score;
                    if (score > max_score) max_score = score;
                }
                
                float sum_exp = 0.0f;
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    attn_weights[t_k] = expf(attn_weights[t_k] - max_score);
                    sum_exp += attn_weights[t_k];
                }
                for (int t_k = 0; t_k <= t_q; t_k++)
                    attn_weights[t_k] /= sum_exp;
                
                for (int t_k = 0; t_k <= t_q; t_k++) {
                    const float *v_vec = V + ((b * T + t_k) * GQA_KV_HEADS + h_kv) * GQA_HEAD_DIM;
                    float a = attn_weights[t_k];
                    for (int i = 0; i < GQA_HEAD_DIM; i++)
                        out_vec[i] += a * v_vec[i];
                }
            }
        }
    }
    
    // Step 6: Gate (sigmoid)
    float *gate_sig = (float *)malloc(N * q_dim * sizeof(float));
    wubu_sigmoid(N * q_dim, gate, gate_sig);
    
    // Save attn_out BEFORE gate multiply
    if (save && save->attn_out_pre_gate)
        memcpy(save->attn_out_pre_gate, attn_out, N * q_dim * sizeof(float));
    
    for (int i = 0; i < N * q_dim; i++)
        attn_out[i] *= gate_sig[i];
    
    // Step 7: Output projection via quantized or F32 matmul
    for (int s = 0; s < N; s++) {
        proj_matmul(attn_out + s * q_dim, q_dim, D_MODEL,
                    w->attn_output_weight, w->attn_output_weight_q, w->attn_output_weight_type,
                    output + s * D_MODEL);
    }
    
    // Save intermediates
    if (save) {
        if (save->Q_raw) memcpy(save->Q_raw, Q_full, N * q_dim * sizeof(float));  // first half
        if (save->Q_norm) memcpy(save->Q_norm, Q_norm, N * q_dim * sizeof(float));
        if (save->K_raw) memcpy(save->K_raw, K, N * kv_dim * sizeof(float));
        if (save->K_norm) memcpy(save->K_norm, K_norm, N * kv_dim * sizeof(float));
        if (save->V) memcpy(save->V, V, N * kv_dim * sizeof(float));
        if (save->gate) memcpy(save->gate, gate, N * q_dim * sizeof(float));
        if (save->gate_sig) memcpy(save->gate_sig, gate_sig, N * q_dim * sizeof(float));
    }
    
cleanup_gqa_save:
    free(Q_full); free(gate); free(K); free(V);
    free(Q_norm); free(K_norm); free(attn_out); free(gate_sig);
}

// Poincaré GQA forward is now in src/wubu_poincare_gqa.c
// (The old tangent-space-approximation stub was removed — the proper
//  implementation uses full Poincaré distance + Möbius combination.)

// ============================================================
// Layer Type Helpers
// ============================================================

int wubu_is_ssm_layer(int layer_idx) {
    // Every 4th layer (index 3, 7, 11, ...) is GQA (full attention)
    // Rest are SSM layers
    return (layer_idx + 1) % 4 != 0;
}

// ============================================================
// Backward Pass — SSM Output Projection (Step 11)
// ============================================================
// Forward: output[s,j] = sum_i delta_out[s,i] * W[i,j]
//          where W = [VALUE_DIM, D_MODEL]
// Backward:
//   d_delta_out += d_output @ W^T   [N,V] = [N,D] @ [D,V]^T
//   dW += delta_out^T @ d_output   [V,D] = [V,N] @ [N,D]
void wubu_ssm_backward_output_proj(
    const float *delta_out,      // [N, VALUE_DIM] forward input (for dW)
    const float *d_output,       // [N, D_MODEL] gradient from upstream
    const float *ssm_out_weight, // [VALUE_DIM, D_MODEL] forward weight (F32, may be NULL)
    const uint8_t *ssm_out_weight_q, int ssm_out_weight_type, // quantized fallback
    float *d_delta_out,          // [N, VALUE_DIM] gradient to propagate
    float *d_ssm_out_weight,     // [VALUE_DIM, D_MODEL] weight grad accum (or NULL)
    int N)
{
    // Dequant fallback: if F32 weight is NULL but quantized available, dequant on-the-fly
    float *dequant_buf = NULL;
    if (!ssm_out_weight && ssm_out_weight_q) {
        int64_t total = (int64_t)VALUE_DIM * D_MODEL;
        dequant_buf = (float *)malloc(total * sizeof(float));
        if (!dequant_buf) {
            fprintf(stderr, "SSM backward output proj: dequant alloc %lld failed\\n",
                    (long long)(total * (int64_t)sizeof(float)));
            memset(d_delta_out, 0, N * VALUE_DIM * sizeof(float));
            return;
        }
        gguf_dequantize(ssm_out_weight_q, ssm_out_weight_type, total, dequant_buf);
        ssm_out_weight = dequant_buf;
    }
    // d_delta_out = d_output @ W^T
    for (int s = 0; s < N; s++) {
        for (int i = 0; i < VALUE_DIM; i++) {
            double sum = 0.0;
            for (int j = 0; j < D_MODEL; j++)
                sum += (double)d_output[s * D_MODEL + j] * (double)ssm_out_weight[i * D_MODEL + j];
            d_delta_out[s * VALUE_DIM + i] += (float)sum;
        }
    }
    // dW = delta_out^T @ d_output  (only if weight grad is requested)
    if (d_ssm_out_weight) {
        for (int i = 0; i < VALUE_DIM; i++) {
            for (int j = 0; j < D_MODEL; j++) {
                double sum = 0.0;
                for (int s = 0; s < N; s++)
                    sum += (double)delta_out[s * VALUE_DIM + i] * (double)d_output[s * D_MODEL + j];
                d_ssm_out_weight[i * D_MODEL + j] += (float)sum;
            }
        }
    }
    if (dequant_buf) free(dequant_buf);
}

// ============================================================
// Backward Pass — Gated Normalization (Step 10)
// ============================================================
// Forward: out[i] = x[i] * scale * w[i] * z[i]
//   where scale = 1/sqrt(mean(x²)+eps), w = norm_weight, z = silu(z_raw)
// Backward: see derivation below
void wubu_ssm_backward_gated_norm(
    const float *x,          // [N, VALUE_DIM] pre-norm delta_out
    const float *z_silu,     // [N, VALUE_DIM] silu(z_raw) from forward
    const float *d_out,      // [N, VALUE_DIM] upstream grad (dL/dout)
    const float *norm_w,     // [SSM_D_STATE] norm weight (broadcast over V_HEADS)
    float *d_x,              // [N, VALUE_DIM] grad to propagate
    float *d_z_silu,         // [N, VALUE_DIM] grad for z_silu
    int B, int T)
{
    const int N = B * T;
    const int d = SSM_D_STATE;  // 128
    const int n_heads = SSM_V_HEADS;  // 32
    
    for (int s = 0; s < N; s++) {
        for (int h = 0; h < n_heads; h++) {
            const float *x_h = x + (s * n_heads + h) * d;
            const float *z_h = z_silu + (s * n_heads + h) * d;
            const float *do_h = d_out + (s * n_heads + h) * d;
            float *dx_h = d_x + (s * n_heads + h) * d;
            float *dz_h = d_z_silu + (s * n_heads + h) * d;
            
            // Compute mean(x²) and rms
            double sum_sq = 0.0;
            for (int i = 0; i < d; i++) sum_sq += (double)x_h[i] * (double)x_h[i];
            float rms = sqrtf((float)(sum_sq / d) + 1e-6f);
            float s = 1.0f / rms;  // scale
            float s3 = s * s * s;  // ds/dm = -s³/2
            
            // Compute inner = sum_j d_out[j] * x[j] * w[j] * z[j]
            double inner = 0.0;
            for (int j = 0; j < d; j++)
                inner += (double)do_h[j] * (double)x_h[j] * (double)norm_w[j] * (double)z_h[j];
            
            // dL/dx[i] = do[i] * w[i] * s * z[i] - (s³/d) * x[i] * inner
            for (int i = 0; i < d; i++) {
                float grad = do_h[i] * norm_w[i] * s * z_h[i];
                grad -= (s3 / d) * x_h[i] * (float)inner;
                dx_h[i] += grad;
            }
            
            // dL/dz_silu[i] = do[i] * x[i] * w[i] * s
            for (int i = 0; i < d; i++) {
                dz_h[i] += do_h[i] * x_h[i] * norm_w[i] * s;
            }
        }
    }
}

// ============================================================
// Backward Pass — Gated Norm Weight Gradient
// ============================================================
// dL/dw[i] = sum_{s,h} dL/dy[s,h,i] * x[s,h,i] * s[s,h] * z[s,h,i]
void wubu_ssm_backward_gated_norm_weight(
    const float *x, const float *z_silu,
    const float *d_out,
    float *d_norm_weight, int B, int T)
{
    if (!d_norm_weight) return;
    const int N = B * T;
    const int d = SSM_D_STATE;
    const int n_vh = SSM_V_HEADS;
    for (int s = 0; s < N; s++) {
        for (int h = 0; h < n_vh; h++) {
            const float *x_h = x + (s * n_vh + h) * d;
            const float *z_h = z_silu + (s * n_vh + h) * d;
            const float *do_h = d_out + (s * n_vh + h) * d;
            double sum_sq = 0.0;
            for (int i = 0; i < d; i++) sum_sq += (double)x_h[i] * (double)x_h[i];
            float s_val = 1.0f / sqrtf((float)(sum_sq / d) + 1e-6f);
            for (int i = 0; i < d; i++)
                d_norm_weight[i] += do_h[i] * x_h[i] * s_val * z_h[i];
        }
    }
}

// ============================================================
// Backward Pass — SiLU activation
// ============================================================
// silu(x) = x * sigmoid(x)
// silu'(x) = silu(x) + sigmoid(x) * (1 - silu(x))
void wubu_silu_backward(int n, const float *x, const float *y,
                        const float *dy, float *dx) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float sig = 1.0f / (1.0f + expf(-v));
        float silu = y[i];
        float silu_grad = silu + sig * (1.0f - silu);
        dx[i] += dy[i] * silu_grad;
    }
}

// ============================================================
// Backward Pass — L2 Normalization
// ============================================================
// Forward: out[i] = x[i] / sqrt(sum(x²) + eps)
// Backward: see derivation in header comment
void wubu_l2_norm_backward(int B, int T, int n_heads, int d,
                           const float *x, float eps,
                           const float *d_out, float *d_x) {
    const int N = B * T;
    for (int s = 0; s < N; s++) {
        for (int h = 0; h < n_heads; h++) {
            const float *inp = x + (s * n_heads + h) * d;
            const float *do_h = d_out + (s * n_heads + h) * d;
            float *dx = d_x + (s * n_heads + h) * d;
            
            double sum_sq = 0.0;
            for (int i = 0; i < d; i++) sum_sq += (double)inp[i] * (double)inp[i];
            float norm = sqrtf(sum_sq + eps);
            float n3 = norm * norm * norm;
            
            // d_i = (do_i / norm) - (x_i / n³) * sum_j (do_j * x_j)
            double dot = 0.0;
            for (int j = 0; j < d; j++) dot += (double)do_h[j] * (double)inp[j];
            
            for (int i = 0; i < d; i++) {
                dx[i] += (float)((double)do_h[i] / norm - (double)inp[i] * dot / n3);
            }
        }
    }
}

// ============================================================
// Backward Pass — SSM Delta Net Recurrence (Step 9)
// ============================================================
// Forward (per head):
//   h_new = h_old * gg + k * (v - (h_old * gg) @ k) * bg
//   output = h_new @ q
// where gg = exp(gate), bg = beta
//
// Backward processes timesteps in reverse (BPTT).
// Requires per-timestep saved states from forward pass.
void wubu_ssm_backward_recurrence(
    int B, int T,
    const float *saved_states,      // [T+1, SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
                                    // saved_states[t] = state after timestep t
                                    // saved_states[0] = initial state (before any timestep)
    const float *q_norm,            // [N, SSM_K_HEADS, SSM_D_STATE]
    const float *k_norm,            // [N, SSM_K_HEADS, SSM_D_STATE]
    const float *v_conv,            // [N, SSM_V_HEADS, SSM_D_STATE]
    const float *beta_flat,         // [N, DT_RANK]
    const float *gate_flat,         // [N, DT_RANK]
    const float *d_output,          // [N, VALUE_DIM] grad from gated norm (dL/d(delta_out))
    float *d_q_norm,                // [N, SSM_K_HEADS, SSM_D_STATE] output grad
    float *d_k_norm,                // [N, SSM_K_HEADS, SSM_D_STATE] output grad
    float *d_v_conv,                // [N, SSM_V_HEADS, SSM_D_STATE] output grad
    float *d_beta_flat,             // [N, DT_RANK] output grad
    float *d_gate_flat,             // [N, DT_RANK] output grad
    float *d_state_init)            // [SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE] BPTT to initial state
{
    const int d = SSM_D_STATE;     // 128
    const int n_vh = SSM_V_HEADS;  // 32
    const int n_kh = SSM_K_HEADS;  // 16
    const int repeat = n_vh / n_kh;
    const int N = B * T;
    const int state_sz = n_vh * d * d;
    
    // BPTT: process timesteps in reverse
    // d_h_next accumulates the gradient w.r.t. h_new from future timesteps
    float *d_h_next = (float *)calloc(state_sz, sizeof(float));
    if (!d_h_next) {
        fprintf(stderr, "backward_recurrence: d_h_next alloc failed\n");
        return;
    }
    
    for (int t = T - 1; t >= 0; t--) {
        for (int b = 0; b < B; b++) {
            int s = b * T + t;  // flat token index
            
            // Get saved states: h_old = state before this timestep, h_new = state after
            const float *h_old = saved_states + t * state_sz;
            const float *h_new = saved_states + (t + 1) * state_sz;
            
            // d_h_new = d_output[t] @ q^T + d_h_next (BPTT from t+1)
            // d_output[t]: [VALUE_DIM] = [n_vh * d]
            // For each vh: output[i] = sum_j h_new[i,j] * q[j]
            // dL/dh_new[i,j] += d_output[t][i] * q[j]
            
            float *d_h_new = (float *)calloc(state_sz, sizeof(float));
            if (!d_h_new) { free(d_h_next); return; }
            
            // dL/dh_new += d_output @ q^T
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
            
            // Add BPTT term from future timesteps
            for (int i = 0; i < state_sz; i++) {
                d_h_new[i] += d_h_next[i];
            }
            
            // Now compute dL/dh_old from d_h_new through the recurrence
            // h_new[i,j] = h_old[i,j] * gg + k[i] * (v[j] - gg * sum_p h_old[p,j] * k[p]) * bg
            // dL/dh_old[i,j] = d_h_new[i,j] * gg
            //                   - gg * k[i] * bg * sum_m d_h_new[m,j] * k[m]
            
            for (int vh = 0; vh < n_vh; vh++) {
                int kh = vh / repeat;
                const float *k_vh = k_norm + (s * n_kh + kh) * d;
                float bg = beta_flat[s * DT_RANK + kh];
                float gg = expf(gate_flat[s * DT_RANK + kh]);
                
                // For each column j of h_new, compute:
                // dL/dh_old[i,j] = d_h_new[i,j] * gg - gg * k_vh[i] * bg * sum_m d_h_new[m,j] * k_vh[m]
                
                for (int j = 0; j < d; j++) {
                    // Compute S[j] = sum_m d_h_new[m,j] * k_vh[m]
                    double S = 0.0;
                    for (int m = 0; m < d; m++) {
                        S += (double)d_h_new[vh * d * d + m * d + j] * (double)k_vh[m];
                    }
                    float factor = gg * bg * (float)S;
                    
                    for (int i = 0; i < d; i++) {
                        float grad = d_h_new[vh * d * d + i * d + j] * gg;
                        grad -= k_vh[i] * factor;
                        
                        if (t > 0) {
                            // Accumulate to d_h_prev (for BPTT to t-1)
                            d_h_next[vh * d * d + i * d + j] = grad;
                        } else if (d_state_init) {
                            // First timestep: accumulate to d_state_init
                            d_state_init[vh * d * d + i * d + j] += grad;
                        }
                    }
                }
            }
            
            // Now compute gradients w.r.t. k, q, v, gg, bg
            // output[i] = sum_j h_new[i,j] * q[j]
            // dL/dq[j] += sum_i d_output[t][i] * h_new[i,j]
            for (int vh = 0; vh < n_vh; vh++) {
                int kh = vh / repeat;
                const float *h_new_vh = h_new + vh * d * d;
                const float *do_vh = d_output + (s * n_vh + vh) * d;
                float *dq_vh = d_q_norm + (s * n_kh + kh) * d;
                
                // dL/dq[j] = sum_i d_output[i] * h_new[i,j]
                for (int j = 0; j < d; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < d; i++) {
                        sum += (double)do_vh[i] * (double)h_new_vh[i * d + j];
                    }
                    dq_vh[j] += (float)sum;
                }
            }
            
            // Gradient through recurrence w.r.t. k, v, gg, bg
            for (int vh = 0; vh < n_vh; vh++) {
                int kh = vh / repeat;
                const float *k_vh = k_norm + (s * n_kh + kh) * d;
                const float *v_vh = v_conv + (s * n_vh + vh) * d;
                const float *h_old_vh = h_old + vh * d * d;
                float bg = beta_flat[s * DT_RANK + kh];
                float gg = expf(gate_flat[s * DT_RANK + kh]);
                float *dk_vh = d_k_norm + (s * n_kh + kh) * d;
                float *dv_vh = d_v_conv + (s * n_vh + vh) * d;
                
                // h_new[i,j] = h_old[i,j]*gg + k_vh[i]*(v_vh[j] - gg * sum_p h_old[p,j]*k_vh[p])*bg
                // output[i] = sum_j h_new[i,j] * q_vh[j]
                // dL/dk_vh[i] = ...
                //   k appears in: h_new[:,j] contribution [+], h_old @ k term [from diff computation]
                
                // h_new[i,j] = h_old[i,j]*gg + k_vh[i]*v_vh[j]*bg - k_vh[i]*gg*bg*sum_p h_old[p,j]*k_vh[p]
                // 
                // dL/dk_vh[i] = sum_m,n d_h_new[m,n] * d(h_new[m,n])/d(k_vh[i])
                // d(h_new[m,n])/d(k_vh[i]) = 
                //   term1: v_vh[n]*bg if m==i, else 0
                //   term2: -gg*bg * [h_old[i,n]*k_vh[i] + sum_p h_old[p,n]*k_vh[p]]  ... 
                //   Actually: -k_vh[m] * gg * bg * (h_old[i,n] * delta_mi + k_vh[i] * h_old[m,n]?)
                //   
                // Let me redo: the term is -k_vh[m]*gg*bg*sum_p h_old[p,n]*k_vh[p]
                // d/d(k_vh[i]): -delta_mi*gg*bg*sum_p h_old[p,n]*k_vh[p] - k_vh[m]*gg*bg*h_old[i,n]
                // So: dL/dk_vh[i] = 
                //   sum_n d_h_new[i,n] * v_vh[n] * bg     [from v*bg term, m=i path]
                //   - sum_n d_h_new[i,n] * gg*bg * sum_p h_old[p,n]*k_vh[p]  [from - sum_p term, m=i path]
                //   - sum_m,n d_h_new[m,n] * k_vh[m]*gg*bg*h_old[i,n]  [from -k*h_old term, m!=i path]
                // = gg*bg * [
                //   sum_n d_h_new[i,n] * (v_vh[n] - sum_p h_old[p,n]*k_vh[p])  
                //   - sum_m,n d_h_new[m,n] * k_vh[m] * h_old[i,n]
                // ]
                
                // First compute diff[n] = v_vh[n] - sum_p h_old[p,n]*k_vh[p]
                float diff[SSM_D_STATE];
                for (int n = 0; n < d; n++) {
                    double hk = 0.0;
                    for (int p = 0; p < d; p++)
                        hk += (double)h_old_vh[p * d + n] * (double)k_vh[p];
                    diff[n] = v_vh[n] - gg * (float)hk;
                }
                
                // dL/dk_vh[i]
                for (int i = 0; i < d; i++) {
                    double grad_k = 0.0;
                    
                    // sum_n d_h_new[i,n] * diff[n] * bg
                    for (int n = 0; n < d; n++) {
                        grad_k += (double)d_h_new[vh * d * d + i * d + n] * (double)diff[n] * (double)bg;
                    }
                    
                    // - gg * bg * sum_m,n d_h_new[m,n] * k_vh[m] * h_old[i,n]
                    for (int m = 0; m < d; m++) {
                        for (int n = 0; n < d; n++) {
                            grad_k -= (double)gg * (double)bg 
                                    * (double)d_h_new[vh * d * d + m * d + n] 
                                    * (double)k_vh[m] * (double)h_old_vh[i * d + n];
                        }
                    }
                    
                    dk_vh[i] += (float)grad_k;
                }
                
                // dL/dv_vh[j] = sum_i d_h_new[i,j] * k_vh[i] * bg
                for (int j = 0; j < d; j++) {
                    double grad_v = 0.0;
                    for (int i = 0; i < d; i++) {
                        grad_v += (double)d_h_new[vh * d * d + i * d + j] 
                                * (double)k_vh[i] * (double)bg;
                    }
                    dv_vh[j] += (float)grad_v;
                }
                
                // dL/dbg = sum_i,j d_h_new[i,j] * k_vh[i] * (v_vh[j] - gg*sum_p h_old[p,j]*k_vh[p])
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
                
                // dL/dgg where gg = exp(gate)
                // h_new[i,j] = h_old[i,j]*gg + k_vh[i]*v_vh[j]*bg - k_vh[i]*gg*bg*sum_p h_old[p,j]*k_vh[p]
                // 
                // d(h_new[i,j])/dgg = h_old[i,j] - k_vh[i]*bg*sum_p h_old[p,j]*k_vh[p]
                //
                // dL/dgg = sum_i,j d_h_new[i,j] * (h_old[i,j] - k_vh[i]*bg*hk_pred[j])
                // where hk_pred[j] = sum_p h_old[p,j]*k_vh[p]
                {
                    double grad_gg = 0.0;
                    for (int i = 0; i < d; i++) {
                        for (int j = 0; j < d; j++) {
                            grad_gg += (double)d_h_new[vh * d * d + i * d + j]
                                     * (double)(h_old_vh[i * d + j] - k_vh[i] * bg * diff[j]);
                        }
                    }
                    // gg = exp(gate), so dL/dgate = dL/dgg * gg
                    d_gate_flat[s * DT_RANK + kh] += (float)(grad_gg * gg);
                }
            }
            
            free(d_h_new);
        }
    }
    
    free(d_h_next);
}

// ============================================================
// Backward Pass — Generic MatMul backward helper
// ============================================================
// Forward: output[s,j] = sum_i input[s,i] * W[i,j]   [N, Din] @ [Din, Dout] -> [N, Dout]
// Backward:
//   d_input[s,i] += sum_j d_output[s,j] * W[i,j]
//   dW[i,j] += sum_s input[s,i] * d_output[s,j]
static void backward_matmul_nt(int N, int Din, int Dout,
                               const float *input, const float *d_output,
                               const float *W, float *d_input, float *dW) {
    // d_input = d_output @ W^T  [N,Din] += [N,Dout] @ [Din,Dout]^T
    for (int s = 0; s < N; s++) {
        for (int i = 0; i < Din; i++) {
            double sum = 0.0;
            for (int j = 0; j < Dout; j++)
                sum += (double)d_output[s * Dout + j] * (double)W[i * Dout + j];
            d_input[s * Din + i] += (float)sum;
        }
    }
    // dW = input^T @ d_output (only if requested)
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
// Backward Pass — Conv1D backward
// ============================================================
// Forward: output[t,c] = sum_{ki=0}^{k-1} input[t+ki,c] * kernel[ki,c]
// Backward:
//   d_input[t+ki,c] += d_output[t,c] * kernel[ki,c]
//   d_kernel[ki,c] += sum_t d_output[t,c] * input[t+ki,c]
static void backward_conv1d(int B, int T, int C, int k,
                            const float *input, // [B, T+k-1, C]
                            const float *d_output, // [B, T, C]
                            const float *kernel, // [k, C]
                            float *d_input, // [B, T+k-1, C]
                            float *d_kernel) { // [k, C]
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                float do_val = d_output[(b * T + t) * C + c];
                for (int ki = 0; ki < k; ki++) {
                    int t_in = t + ki;
                    // d_input[t+ki,c] += d_output[t,c] * kernel[ki,c]
                    d_input[(b * (T + k - 1) + t_in) * C + c] += do_val * kernel[ki * C + c];
                }
            }
        }
    }
    // d_kernel
    for (int ki = 0; ki < k; ki++) {
        for (int c = 0; c < C; c++) {
            double sum = 0.0;
            for (int b = 0; b < B; b++) {
                for (int t = 0; t < T; t++) {
                    int t_in = t + ki;
                    sum += (double)d_output[(b * T + t) * C + c] 
                         * (double)input[(b * (T + k - 1) + t_in) * C + c];
                }
            }
            if (d_kernel) d_kernel[ki * C + c] += (float)sum;
        }
    }
}

// ============================================================
// Full SSM Layer Backward Pass
// ============================================================
// Chains all backward steps from 11 to 0.
// Requires all intermediate buffers from the forward pass.
void wubu_ssm_backward(
    int B, int T,
    const float *x,              // [B, T, D_MODEL] input to forward
    const float *output,         // [B, T, D_MODEL] output from forward
    const float *d_output,       // [B, T, D_MODEL] upstream gradient
    
    // Forward intermediate buffers (must be preserved from forward call)
    const float *qkv_all,        // [N, CONV_DIM] step 1 output
    const float *z_all,          // [N, VALUE_DIM] step 2 output
    const float *beta_raw,       // [N, DT_RANK] step 3 output
    const float *alpha_raw,      // [N, DT_RANK] step 3 output
    const float *conv_output,    // [N, CONV_DIM] after step 5 silu
    const float *q_conv,         // [N, KEY_DIM] step 6 split
    const float *k_conv,         // [N, KEY_DIM] step 6 split
    const float *v_conv,         // [N, VALUE_DIM] step 6 split
    const float *q_norm,         // [N, KEY_DIM] after step 7 L2 norm
    const float *k_norm,         // [N, KEY_DIM] after step 7 L2 norm
    const float *delta_out,      // [N, VALUE_DIM] after step 9 (pre-gated-norm)
    const float *z_silu,         // [N, VALUE_DIM] silu(z_all) from forward
    const float *ssm_states,     // [T+1, SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
    const float *beta_flat,      // [N, DT_RANK] sigmoid(beta_raw)
    const float *gate_flat,      // [N, DT_RANK] alpha_softplus * ssm_a
    const float *conv_state,     // [B, CONV_KERNEL-1, CONV_DIM] for conv_input reconstruction
    
    // Model weights (for the forward pass - needed for backward)
    const ssm_layer_weights *w,
    
    // Output gradients (accumulated)
    float *d_x,                  // [B, T, D_MODEL] gradient to propagate
    float *d_qkv_weight,         // [D_MODEL, CONV_DIM] gradient for attn_qkv_weight
    float *d_gate_weight,        // [D_MODEL, VALUE_DIM] gradient for attn_gate_weight
    float *d_beta_weight,        // [D_MODEL, DT_RANK] gradient for ssm_beta_weight
    float *d_alpha_weight,       // [D_MODEL, DT_RANK] gradient for ssm_alpha_weight
    float *d_conv1d_weight,      // [CONV_KERNEL, CONV_DIM] gradient for ssm_conv1d_weight
    float *d_ssm_out_weight,     // [VALUE_DIM, D_MODEL] gradient for ssm_out_weight
    float *d_ssm_norm_weight,    // [SSM_D_STATE] gradient for norm weight
    
    // State gradient (for BPTT)
    float *d_ssm_state_init)     // [SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
{
    const int N = B * T;
    const int C = CONV_DIM;
    const int d = SSM_D_STATE;
    const int n_vh = SSM_V_HEADS;
    const int n_kh = SSM_K_HEADS;
    const int state_sz = n_vh * d * d;
    
    // Allocate all buffers first, initialized to NULL for safe cleanup
    float *d_delta_out = (float *)calloc(N * VALUE_DIM, sizeof(float));
    float *d_z_silu = (float *)calloc(N * VALUE_DIM, sizeof(float));
    float *d_z_all = (float *)calloc(N * VALUE_DIM, sizeof(float));
    float *d_q_norm = (float *)calloc(N * KEY_DIM, sizeof(float));
    float *d_k_norm = (float *)calloc(N * KEY_DIM, sizeof(float));
    float *d_v_conv = (float *)calloc(N * VALUE_DIM, sizeof(float));
    float *d_q_conv = (float *)calloc(N * KEY_DIM, sizeof(float));
    float *d_k_conv = (float *)calloc(N * KEY_DIM, sizeof(float));
    float *d_conv_out = (float *)calloc(N * C, sizeof(float));
    float *d_conv_input = (float *)calloc(B * (T + CONV_KERNEL - 1) * C, sizeof(float));
    float *d_beta_flat = (float *)calloc(N * DT_RANK, sizeof(float));
    float *d_gate_flat = (float *)calloc(N * DT_RANK, sizeof(float));
    
    if (!d_delta_out || !d_z_silu || !d_z_all || !d_q_norm || !d_k_norm ||
        !d_v_conv || !d_q_conv || !d_k_conv || !d_conv_out || !d_conv_input ||
        !d_beta_flat || !d_gate_flat) {
        fprintf(stderr, "SSM backward: alloc failed\n");
        goto cleanup_bwd;
    }
    
    // === Step 11: Output projection backward ===
    wubu_ssm_backward_output_proj(delta_out, d_output, w->ssm_out_weight,
                                   w->ssm_out_weight_q, w->ssm_out_weight_type,
                                   d_delta_out, d_ssm_out_weight, N);
    
    // === Step 10: Gated normalization backward ===
    // Transform dL/d(post_norm) in d_delta_out to dL/d(pre_norm)
    // First save the post-norm gradient into a temp, then zero d_delta_out for output
    float *d_post_norm = d_delta_out;  // alias for clarity
    float *d_pre_norm = (float *)calloc(N * VALUE_DIM, sizeof(float));
    if (!d_pre_norm) goto cleanup_bwd;
    
    wubu_ssm_backward_gated_norm(delta_out, z_silu, d_post_norm, w->ssm_norm_weight,
                                  d_pre_norm, d_z_silu, B, T);
    
    // Replace d_delta_out with pre-norm gradient and free temp
    memcpy(d_delta_out, d_pre_norm, N * VALUE_DIM * sizeof(float));
    free(d_pre_norm);
    
    // Compute norm weight gradient
    wubu_ssm_backward_gated_norm_weight(delta_out, z_silu, d_post_norm,
                                         d_ssm_norm_weight, B, T);
    
    // Note: d_delta_out now contains dL/d(pre-norm delta_out) for step 9
    // d_z_silu contains gradient w.r.t. silu(z)
    
    // Step 9b: SiLU backward for z
    // z_silu = silu(z_all), so backprop through SiLU
    wubu_silu_backward(N * VALUE_DIM, z_all, z_silu, d_z_silu, d_z_all);
    
    // === Step 9: Delta net recurrence backward ===
    // Compute beta_flat and gate_flat if not provided
    float *beta_flat_computed = NULL;
    float *gate_flat_computed = NULL;
    if (!beta_flat || !gate_flat) {
        beta_flat_computed = (float *)malloc(N * DT_RANK * sizeof(float));
        gate_flat_computed = (float *)malloc(N * DT_RANK * sizeof(float));
        if (!beta_flat_computed || !gate_flat_computed) {
            fprintf(stderr, "SSM backward: beta/gate_flat alloc failed\\n");
            free(beta_flat_computed); free(gate_flat_computed);
            goto cleanup_bwd;
        }
        for (int i = 0; i < N * DT_RANK; i++) {
            beta_flat_computed[i] = 1.0f / (1.0f + expf(-beta_raw[i]));
            float biased = alpha_raw[i] + w->ssm_dt_bias[i % DT_RANK];
            float sp;
            if (biased > 80.0f) sp = biased;
            else if (biased < -80.0f) sp = 0.0f;
            else sp = logf(1.0f + expf(biased));
            gate_flat_computed[i] = sp * w->ssm_a[i % DT_RANK];
        }
        beta_flat = beta_flat_computed;
        gate_flat = gate_flat_computed;
    }
    wubu_ssm_backward_recurrence(B, T, ssm_states, q_norm, k_norm, v_conv,
                                 beta_flat, gate_flat, d_delta_out,
                                 d_q_norm, d_k_norm, d_v_conv,
                                 d_beta_flat, d_gate_flat, d_ssm_state_init);
    
    // === Step 7: L2 normalization backward for Q and K ===
    wubu_l2_norm_backward(B, T, SSM_K_HEADS, SSM_D_STATE, q_conv, g_ssm_l2_eps, d_q_norm, d_q_conv);
    wubu_l2_norm_backward(B, T, SSM_K_HEADS, SSM_D_STATE, k_conv, g_ssm_l2_eps, d_k_norm, d_k_conv);
    
    // === Step 6: Split backward (concatenate Q, K, V gradients) ===
    for (int s = 0; s < N; s++) {
        const float *cq = d_q_conv + s * KEY_DIM;
        const float *ck = d_k_conv + s * KEY_DIM;
        const float *cv = d_v_conv + s * VALUE_DIM;
        float *co = d_conv_out + s * C;
        for (int i = 0; i < KEY_DIM; i++) co[i] += cq[i];
        for (int i = 0; i < KEY_DIM; i++) co[KEY_DIM + i] += ck[i];
        for (int i = 0; i < VALUE_DIM; i++) co[2 * KEY_DIM + i] += cv[i];
    }
    
    // === Step 5b: SiLU backward for conv output ===
    // conv_output from forward is post-silu (silu applied in-place).
    // We need the pre-silu value for silu_backward.
    // Approximate: for large |conv_output|, silu(x) ≈ x so silu'(x) ≈ 1.
    // For accuracy, save pre-silu conv_linear in forward pass in future.
    // For now, use conv_output as both x≈y (reasonable for typical activations).
    // silu'(x) ≈ 1 for |x| > 3, and |conv_output| is typically large.
    // Better: estimate x from y via Newton for small values.
    for (int i = 0; i < N * C; i++) {
        float y = conv_output[i];
        float x_est;
        if (y > 3.0f || y < -3.0f) {
            x_est = y;  // silu(3) = 3*sigmoid(3) = 3*0.953 ≈ 2.86
        } else if (y > -0.1f && y < 0.1f) {
            x_est = y * 2.0f;  // near 0: silu(x) ≈ x/2
        } else {
            // Newton: solve f(x)=x*sigmoid(x)-y=0
            x_est = y;
            for (int iter = 0; iter < 5; iter++) {
                float sig = 1.0f / (1.0f + expf(-x_est));
                float f = x_est * sig - y;
                float df = sig + x_est * sig * (1.0f - sig);
                if (fabsf(df) > 1e-6f) x_est -= f / df;
            }
        }
        float sig = 1.0f / (1.0f + expf(-x_est));
        float silu_grad = y + sig * (1.0f - y);
        d_conv_out[i] *= silu_grad;
    }
    
    // === Step 5a: Conv1d backward ===
    // Need conv_input (from forward) for d_kernel
    // conv_input is not passed as parameter, reconstruct it
    float *conv_input_bwd = (float *)malloc(B * (T + CONV_KERNEL - 1) * C * sizeof(float));
    if (!conv_input_bwd) goto cleanup_bwd;
    
    // Rebuild conv_input the same way as forward
    // Use conv_state for the first CONV_KERNEL-1 elements
    for (int b = 0; b < B; b++) {
        memcpy(conv_input_bwd + b * (T + CONV_KERNEL - 1) * C,
               conv_state + b * (CONV_KERNEL - 1) * C,
               (CONV_KERNEL - 1) * C * sizeof(float));
        memcpy(conv_input_bwd + (b * (T + CONV_KERNEL - 1) + (CONV_KERNEL - 1)) * C,
               qkv_all + b * T * C,
               T * C * sizeof(float));
    }
    
    backward_conv1d(B, T, C, CONV_KERNEL, conv_input_bwd, d_conv_out,
                    w->ssm_conv1d_weight, d_conv_input, d_conv1d_weight);
    
    // === Step 4: Gate computation backward ===
    // Forward: gate = softplus(alpha_raw + dt_bias) * ssm_a
    // beta = sigmoid(beta_raw)
    // d_alpha_raw and d_beta_raw come from d_gate_flat and d_beta_flat
    
    float *d_beta_raw = (float *)calloc(N * DT_RANK, sizeof(float));
    float *d_alpha_raw = (float *)calloc(N * DT_RANK, sizeof(float));
    float *d_alpha_biased = (float *)calloc(N * DT_RANK, sizeof(float));
    float *d_alpha_softplus = (float *)calloc(N * DT_RANK, sizeof(float));
    
    if (!d_beta_raw || !d_alpha_raw || !d_alpha_biased || !d_alpha_softplus)
        goto cleanup_bwd;
    
    // d_alpha_softplus[j] = d_gate_flat[j] * ssm_a[j]
    for (int i = 0; i < N * DT_RANK; i++) {
        d_alpha_softplus[i] = d_gate_flat[i] * w->ssm_a[i % DT_RANK];
    }
    
    // Backward through softplus: forward y = log(1+exp(x)) where x = alpha_raw + bias
    // dy/dx = 1/(1+exp(-x)) = sigmoid(x)
    // But we don't have x saved. Need alpha_biased or alpha_raw.
    // We have alpha_raw but not alpha_biased.
    // Since alpha_biased = alpha_raw + dt_bias, and dt_bias is a constant,
    // d_alpha_raw = d_alpha_biased = sigmoid(alpha_biased) * d_alpha_softplus
    //
    // We need alpha_biased (forward intermediate). Not saved.
    // Recompute it.
    for (int s = 0; s < N; s++) {
        for (int j = 0; j < DT_RANK; j++) {
            float alpha_biased = alpha_raw[s * DT_RANK + j] + w->ssm_dt_bias[j];
            float sig = 1.0f / (1.0f + expf(-alpha_biased));
            d_alpha_raw[s * DT_RANK + j] += d_alpha_softplus[s * DT_RANK + j] * sig;
        }
    }
    
    // Backward through sigmoid: forward y = sigmoid(x), dy/dx = y*(1-y)
    // d_beta_raw = d_beta_flat * sigmoid(beta_raw) * (1 - sigmoid(beta_raw))
    for (int i = 0; i < N * DT_RANK; i++) {
        float sig = beta_flat[i];  // Already sigmoid output
        d_beta_raw[i] = d_beta_flat[i] * sig * (1.0f - sig);
    }
    
    // === Steps 1-3: MatMul backward for QKV, Z, Beta, Alpha ===
    // Dequant weights on-demand if F32 is NULL
    float *dequant_qkv = NULL;
    float *dequant_gate = NULL;
    if (!w->attn_qkv_weight && w->attn_qkv_weight_q) {
        int64_t n = (int64_t)D_MODEL * CONV_DIM;
        dequant_qkv = (float *)malloc(n * sizeof(float));
        if (dequant_qkv) gguf_dequantize(w->attn_qkv_weight_q, w->attn_qkv_weight_type, n, dequant_qkv);
    }
    if (!w->attn_gate_weight && w->attn_gate_weight_q) {
        int64_t n = (int64_t)D_MODEL * VALUE_DIM;
        dequant_gate = (float *)malloc(n * sizeof(float));
        if (dequant_gate) gguf_dequantize(w->attn_gate_weight_q, w->attn_gate_weight_type, n, dequant_gate);
    }
    const float *qkv_w = dequant_qkv ? dequant_qkv : w->attn_qkv_weight;
    const float *gate_w = dequant_gate ? dequant_gate : w->attn_gate_weight;
    
    // QKV: x @ W_qkv -> qkv_all [N, D_MODEL] @ [D_MODEL, C] -> [N, C]
    backward_matmul_nt(N, D_MODEL, C, x, d_conv_out, qkv_w,
                       d_x, d_qkv_weight);
    
    // Z: x @ W_gate -> z_all [N, D_MODEL] @ [D_MODEL, VALUE_DIM] -> [N, VALUE_DIM]
    backward_matmul_nt(N, D_MODEL, VALUE_DIM, x, d_z_all, gate_w,
                       d_x, d_gate_weight);
    
    // Beta: x @ W_beta -> beta_raw [N, D_MODEL] @ [D_MODEL, DT_RANK] -> [N, DT_RANK]
    backward_matmul_nt(N, D_MODEL, DT_RANK, x, d_beta_raw, w->ssm_beta_weight,
                       d_x, d_beta_weight);
    
    // Alpha: x @ W_alpha -> alpha_raw [N, D_MODEL] @ [D_MODEL, DT_RANK] -> [N, DT_RANK]
    backward_matmul_nt(N, D_MODEL, DT_RANK, x, d_alpha_raw, w->ssm_alpha_weight,
                       d_x, d_alpha_weight);
    
cleanup_bwd:
    free(d_delta_out); free(d_z_silu); free(d_z_all);
    free(d_q_norm); free(d_k_norm); free(d_v_conv);
    free(d_q_conv); free(d_k_conv);
    free(d_conv_out); free(d_conv_input);
    free(d_beta_flat); free(d_gate_flat);
    free(d_beta_raw); free(d_alpha_raw);
    free(d_alpha_biased); free(d_alpha_softplus);
    free(conv_input_bwd);
    free(beta_flat_computed); free(gate_flat_computed);
    free(dequant_qkv); free(dequant_gate);
}

// ============================================================
// GQA Layer Backward Pass
// ============================================================

// Backward through GQA attention (Step 5)
// Forward: attn_out = softmax(Q@K^T/sqrt(d)) @ V  [causal, GQA grouped]
void wubu_gqa_backward_attention(
    int B, int T,
    const float *Q_norm,      // [N, GQA_Q_HEADS * GQA_HEAD_DIM]
    const float *K_norm,      // [N, GQA_KV_HEADS * GQA_HEAD_DIM]
    const float *V,           // [N, GQA_KV_HEADS * GQA_HEAD_DIM]
    const float *d_attn_out,  // [N, GQA_Q_HEADS * GQA_HEAD_DIM] upstream grad
    float *d_Q,               // [N, GQA_Q_HEADS * GQA_HEAD_DIM] output
    float *d_K,               // [N, GQA_KV_HEADS * GQA_HEAD_DIM] output
    float *d_V)               // [N, GQA_KV_HEADS * GQA_HEAD_DIM] output
{
    const int hd = GQA_HEAD_DIM;  // 128
    const int n_q = GQA_Q_HEADS;  // 32
    const int n_kv = GQA_KV_HEADS; // 4
    const int q_per_kv = n_q / n_kv;  // 8
    const float scale = 1.0f / sqrtf((float)hd);
    
    for (int b = 0; b < B; b++) {
        for (int t_q = 0; t_q < T; t_q++) {
            for (int h_q = 0; h_q < n_q; h_q++) {
                int h_kv = h_q / q_per_kv;
                
                const float *q_vec = Q_norm + ((b * T + t_q) * n_q + h_q) * hd;
                const float *d_out = d_attn_out + ((b * T + t_q) * n_q + h_q) * hd;
                float *dq = d_Q + ((b * T + t_q) * n_q + h_q) * hd;
                
                // Compute attention scores and softmax
                // Store scores for backward
                int max_t = t_q + 1;  // causal: attend to self and past
                float scores[4096];
                float max_score = -1e30f;
                
                for (int tk = 0; tk < max_t; tk++) {
                    const float *k_vec = K_norm + ((b * T + tk) * n_kv + h_kv) * hd;
                    double s = 0.0;
                    for (int i = 0; i < hd; i++)
                        s += (double)q_vec[i] * (double)k_vec[i];
                    scores[tk] = (float)(s * scale);
                    if (scores[tk] > max_score) max_score = scores[tk];
                }
                
                // Softmax
                double sum_exp = 0.0;
                for (int tk = 0; tk < max_t; tk++) {
                    scores[tk] = expf(scores[tk] - max_score);
                    sum_exp += scores[tk];
                }
                float inv_sum = 1.0f / (float)sum_exp;
                for (int tk = 0; tk < max_t; tk++)
                    scores[tk] *= inv_sum;
                
                // Step 1: dL/dV = d_out * score[t_k]
                for (int tk = 0; tk < max_t; tk++) {
                    float a = scores[tk];
                    float *dv = d_V + ((b * T + tk) * n_kv + h_kv) * hd;
                    for (int i = 0; i < hd; i++)
                        dv[i] += d_out[i] * a;
                }
                
                // Step 2: dL/d(score) = d_out · V
                float d_score[4096];
                for (int tk = 0; tk < max_t; tk++) {
                    const float *v_vec = V + ((b * T + tk) * n_kv + h_kv) * hd;
                    double ds = 0.0;
                    for (int i = 0; i < hd; i++)
                        ds += (double)d_out[i] * (double)v_vec[i];
                    d_score[tk] = (float)ds;
                }
                
                // Step 3: Backprop through softmax
                // d(logit[t_k]) = score[t_k] * (d_score[t_k] - sum_j d_score[j] * score[j])
                double dot = 0.0;
                for (int j = 0; j < max_t; j++)
                    dot += (double)d_score[j] * (double)scores[j];
                
                float d_logit[4096];
                for (int tk = 0; tk < max_t; tk++)
                    d_logit[tk] = scores[tk] * (d_score[tk] - (float)dot);
                
                // Step 4: dL/dQ[t_q] = scale * sum_{t_k} d_logit[t_k] * K[t_k]
                for (int tk = 0; tk < max_t; tk++) {
                    const float *k_vec = K_norm + ((b * T + tk) * n_kv + h_kv) * hd;
                    float dl = d_logit[tk] * scale;
                    for (int i = 0; i < hd; i++)
                        dq[i] += dl * k_vec[i];
                }
                
                // Step 5: dL/dK[t_k] = scale * d_logit[t_k] * Q[t_q]
                for (int tk = 0; tk < max_t; tk++) {
                    float *dk = d_K + ((b * T + tk) * n_kv + h_kv) * hd;
                    float dl = d_logit[tk] * scale;
                    for (int i = 0; i < hd; i++)
                        dk[i] += dl * q_vec[i];
                }
            }
        }
    }
}

// Full GQA layer backward
void wubu_gqa_backward(
    int B, int T,
    const float *x,             // [B, T, D_MODEL] input to forward
    const float *Q_norm,        // [N, q_dim] post-RMSNorm Q
    const float *Q_raw,         // [N, q_dim] pre-RMSNorm Q (from Q_full first half)
    const float *K_norm,        // [N, kv_dim] post-RMSNorm K
    const float *K_raw,         // [N, kv_dim] pre-RMSNorm K (from forward K buffer)
    const float *V,             // [N, kv_dim] raw V
    const float *gate,          // [N, q_dim] raw gate (pre-sigmoid)
    const float *gate_sig,      // [N, q_dim] sigmoid(gate)
    const float *attn_out,      // [N, q_dim] post-gate attn_out
    const float *output,        // [B, T, D_MODEL] forward output
    const float *d_output,      // [B, T, D_MODEL] upstream gradient
    const gqa_layer_weights *w, // weights
    float *d_x,                 // [B, T, D_MODEL] gradient to propagate
    float *d_q_weight,          // [D_MODEL, q_dim*2] fused Q+gate weight grad
    float *d_k_weight,          // [D_MODEL, kv_dim]
    float *d_v_weight,          // [D_MODEL, kv_dim]
    float *d_q_norm_weight,     // [GQA_HEAD_DIM]
    float *d_k_norm_weight,     // [GQA_HEAD_DIM]
    float *d_out_weight)        // [q_dim, D_MODEL]
{
    const int N = B * T;
    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
    
    // Allocate temp buffers
    float *d_attn_out = (float *)calloc(N * q_dim, sizeof(float));
    float *d_gate = (float *)calloc(N * q_dim, sizeof(float));
    float *d_Q_norm = (float *)calloc(N * q_dim, sizeof(float));
    float *d_K_norm = (float *)calloc(N * kv_dim, sizeof(float));
    float *d_V = (float *)calloc(N * kv_dim, sizeof(float));
    float *d_Q_raw = (float *)calloc(N * q_dim, sizeof(float));
    float *d_K_raw = (float *)calloc(N * kv_dim, sizeof(float));
    float *d_Q_full = (float *)calloc(N * q_dim * 2, sizeof(float));
    // d_Q_full: first q_dim for Q grad, second q_dim for gate grad

    // Dequant-on-demand buffers for quantized-only GQA weights
    float *dequant_q = NULL;
    float *dequant_k = NULL;
    float *dequant_v = NULL;
    const float *q_weight = w->attn_q_weight;
    const float *k_weight = w->attn_k_weight;
    const float *v_weight = w->attn_v_weight;
    if (!q_weight && w->attn_q_weight_q) {
        int64_t n = (int64_t)D_MODEL * q_dim * 2;
        dequant_q = (float *)malloc(n * sizeof(float));
        if (dequant_q) { gguf_dequantize(w->attn_q_weight_q, w->attn_q_weight_type, n, dequant_q); q_weight = dequant_q; }
    }
    if (!k_weight && w->attn_k_weight_q) {
        int64_t n = (int64_t)D_MODEL * kv_dim;
        dequant_k = (float *)malloc(n * sizeof(float));
        if (dequant_k) { gguf_dequantize(w->attn_k_weight_q, w->attn_k_weight_type, n, dequant_k); k_weight = dequant_k; }
    }
    if (!v_weight && w->attn_v_weight_q) {
        int64_t n = (int64_t)D_MODEL * kv_dim;
        dequant_v = (float *)malloc(n * sizeof(float));
        if (dequant_v) { gguf_dequantize(w->attn_v_weight_q, w->attn_v_weight_type, n, dequant_v); v_weight = dequant_v; }
    }

    if (!d_attn_out || !d_gate || !d_Q_norm || !d_K_norm || !d_V ||
        !d_Q_raw || !d_K_raw || !d_Q_full) {
        fprintf(stderr, "GQA backward: alloc failed\n");
        goto cleanup_gqa;
    }
    
    // === Step 7: Output projection backward ===
    // Same dims as SSM output proj: [q_dim=4096, D_MODEL=2048]
    // Reuse the SSM function (VALUE_DIM == q_dim)
    wubu_ssm_backward_output_proj(attn_out, d_output, w->attn_output_weight,
                                   w->attn_output_weight_q, w->attn_output_weight_type,
                                   d_attn_out, d_out_weight, N);
    
    // === Step 6: Gate backward ===
    // Forward: attn_out_post = attn_out_pre * sigmoid(gate)
    // d_attn_out_pre = d_attn_out_post * sigmoid(gate)
    // d_gate = d_attn_out_post * attn_out_pre * sigmoid'(gate)
    // But attn_out_pre is not saved. We have attn_out_post = attn_out (from forward).
    // attn_out_pre = attn_out / sigmoid(gate)
    // Safer: compute d_attn_out_from_gate directly
    for (int i = 0; i < N * q_dim; i++) {
        float sig = gate_sig[i];
        float sig_grad = sig * (1.0f - sig);
        // attn_out_post = attn_out_pre * sig
        // attn_out_pre = attn_out / sig  (can be recovered)
        float attn_pre = (sig > 1e-7f) ? attn_out[i] / sig : 0.0f;
        d_gate[i] += d_attn_out[i] * attn_pre * sig_grad;
        d_attn_out[i] *= sig;  // this applies the chain rule: d_attn_pre = d_attn_post * sig
    }
    // Now d_attn_out contains gradient w.r.t. pre-gate attn_out
    
    // === Step 5: Attention backward ===
    wubu_gqa_backward_attention(B, T, Q_norm, K_norm, V, d_attn_out,
                                d_Q_norm, d_K_norm, d_V);
    
    // === Step 4: Q/K RMSNorm backward ===
    // Forward: y[i] = x[i] * r * w[i] where r = 1/sqrt(mean(x²)+eps), w = norm_weight
    // dL/dx[i] = do[i] * w[i] * r - (r³/d) * x[i] * sum_j(do[j] * w[j] * x[j])
    // Q RMSNorm
    for (int s = 0; s < N; s++) {
        for (int h = 0; h < GQA_Q_HEADS; h++) {
            const float *x_h = Q_raw + (s * GQA_Q_HEADS + h) * GQA_HEAD_DIM;
            const float *do_h = d_Q_norm + (s * GQA_Q_HEADS + h) * GQA_HEAD_DIM;
            float *dx_h = d_Q_raw + (s * GQA_Q_HEADS + h) * GQA_HEAD_DIM;
            
            double sum_sq = 0.0;
            for (int i = 0; i < GQA_HEAD_DIM; i++) sum_sq += (double)x_h[i] * (double)x_h[i];
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
            for (int i = 0; i < GQA_HEAD_DIM; i++) sum_sq += (double)x_h[i] * (double)x_h[i];
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
    // d_Q_full = [d_Q_raw | d_gate] where each has dim q_dim
    for (int s = 0; s < N; s++) {
        memcpy(d_Q_full + s * q_dim * 2, d_Q_norm + s * q_dim, q_dim * sizeof(float));
        memcpy(d_Q_full + s * q_dim * 2 + q_dim, d_gate + s * q_dim, q_dim * sizeof(float));
    }
    
    // === Steps 1-3: MatMul backward for Q+gate, K, V ===
    backward_matmul_nt(N, D_MODEL, q_dim * 2, x, d_Q_full,
                       q_weight, d_x, d_q_weight);

    backward_matmul_nt(N, D_MODEL, kv_dim, x, d_K_raw,
                       k_weight, d_x, d_k_weight);

    backward_matmul_nt(N, D_MODEL, kv_dim, x, d_V,
                       v_weight, d_x, d_v_weight);
    
cleanup_gqa:
    free(d_attn_out); free(d_gate);
    free(d_Q_norm); free(d_K_norm); free(d_V);
    free(d_Q_raw); free(d_K_raw); free(d_Q_full);
    free(dequant_q); free(dequant_k); free(dequant_v);
}

// ============================================================
// RMSNorm Backward (model-level helper)
// ============================================================
void wubu_rms_norm_backward(int B, int T, int d,
                            const float *x, const float *weight, float eps,
                            const float *d_out, float *d_x) {
    const int N = B * T;
    for (int s = 0; s < N; s++) {
        const float *inp = x + s * d;
        const float *do_h = d_out + s * d;
        float *dx = d_x + s * d;
        double sum_sq = 0.0;
        for (int i = 0; i < d; i++) sum_sq += (double)inp[i] * (double)inp[i];
        float rms = sqrtf((float)(sum_sq / d) + eps);
        float r = 1.0f / rms;
        float r3 = r * r * r;
        double inner = 0.0;
        for (int j = 0; j < d; j++)
            inner += (double)do_h[j] * (double)weight[j] * (double)inp[j];
        for (int i = 0; i < d; i++)
            dx[i] += do_h[i] * weight[i] * r - (r3 / d) * inp[i] * (float)inner;
    }
}

// ============================================================
// Sequential SSM recurrence (extracted for chunked verification)
// ============================================================
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
