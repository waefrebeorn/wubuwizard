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
