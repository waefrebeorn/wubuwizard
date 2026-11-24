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

#ifdef __cplusplus
}
#endif
#endif /* WUBU_GPU_H */
