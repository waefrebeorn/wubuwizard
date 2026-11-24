/*
 * wubu_gpu.h -- host side of the CUDA DLL bridge (see src/wubu_gpu.c).
 *
 * All entry points fail closed: wubu_gpu_available() returning 0 means "no CUDA
 * device or no DLL", and every call site must keep working on the CPU path.
 */
#ifndef WUBU_GPU_H
#define WUBU_GPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int  wubu_gpu_init(void);
int  wubu_gpu_available(void);
void wubu_gpu_stats(unsigned long long *total, unsigned long long *freeb);
int  wubu_gpu_fits(unsigned long long bytes, unsigned long long reserve);

void *wubu_gpu_upload(const void *host, size_t bytes);
void  wubu_gpu_free(void *d_ptr);

/* y[n_cols] = W[n_cols][K] . x[K]; W already device-resident as ggml_type. */
int wubu_gpu_matvec(const void *d_w, int ggml_type, const float *x,
                    int K, int n_cols, long long col_stride, float *y);

/* Stage an activation (upload + Q8_K quantise) once, then run several matvecs against
 * it with wubu_gpu_matvec_cached. Used for q/k/v, which share one input buffer. */
int wubu_gpu_stage_input(const float *x, int K);
int wubu_gpu_matvec_cached(const void *d_w, int ggml_type, int K, int n_cols,
                           long long col_stride, float *y);

/* Fused FFN with a type per tensor (gate, up, down). */
int wubu_gpu_ffn3(const float *x, const void *d_wg, const void *d_wu, const void *d_wd,
                  int tg, int tu, int td, int hidden, int ff,
                  long long sg, long long su, long long sd, float *out);

/* Fused q/k/v with a type per projection (see wubu_gpu.c). */
int wubu_gpu_qkv3(const void *d_wq, const void *d_wk, const void *d_wv,
                  int tq, int tk, int tv, const float *x,
                  int K, int qd, int kvd, long long sq, long long sk, long long sv,
                  float *q, float *k, float *v);

/* Device attention half-layer. Prime the per-layer device KV once before the first decode
 * token; returns non-zero when pos is past wubu_gpu_attn_half_capacity() so the caller can
 * stay on the CPU path. */
int wubu_gpu_attn_half_capacity(void);
int wubu_gpu_attn_half_prime(int layer_id, const float *cpu_k, const float *cpu_v, int n, int kvd);
int wubu_gpu_attn_half(int layer_id, const void *d_wq, const void *d_wk, const void *d_wv,
                       const void *d_wo, int tq, int tk, int tv, int to, const float *x,
                       int hidden, int qd, int kvd, int hd, int n_heads, int n_kv_heads,
                       int pos, int kv_start, int kv_len,
                       long long sq, long long sk, long long sv, long long so,
                       const float *qn, const float *kn, const float *freqs,
                       int n_rot, float rope_base, float *out, float *k_out, float *v_out);

/* Fused decode q/k/v: one upload, one quantisation, one download of q|k|v. */
int wubu_gpu_qkv(const void *d_wq, const void *d_wk, const void *d_wv, int ggml_type,
                 const float *x, int K, int qd, int kvd,
                 long long sq, long long sk, long long sv, float *q, float *k, float *v);

/* Fused decode FFN: out = down(gelu(gate(x)) * up(x)), gate/up/down all device-resident
 * and of the same ggml_type. One upload and one download per layer; the tanh GELU runs
 * on the device. Returns 0 on success so callers can fall back to wubu_gpu_matvec. */
int wubu_gpu_ffn(const float *x, const void *d_wg, const void *d_wu, const void *d_wd,
                 int ggml_type, int hidden, int ff,
                 long long sg, long long su, long long sd, float *out);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_GPU_H */
