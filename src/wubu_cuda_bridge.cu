/*
 * wubu_cuda_bridge.cu -- CUDA compute behind a dynamically loaded DLL.
 *
 * Why a DLL: nvcc + cudart.lib emit MSVC-ABI objects, and MinGW's ld.exe cannot
 * resolve them (__security_cookie, __GSHandlerCheck, _Init_thread_epoch). Instead
 * of forcing the whole engine into an MSVC build, the GPU kernels live in this
 * DLL (built by nvcc, linked by MSVC) and the MinGW engine loads it at runtime
 * with LoadLibrary + GetProcAddress through plain extern "C" symbols.
 *
 * First entry points prove the bridge and give one real kernel: an F32 GEMV that
 * runs the token-embedding / LM-head style matvec on the device.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#define WUBU_EXPORT extern "C" __declspec(dllexport)
#else
#define WUBU_EXPORT extern "C" __attribute__((visibility("default")))
#endif

__global__ static void sgemv_kernel(const float * __restrict__ w,
                                    const float * __restrict__ x,
                                    float * __restrict__ y,
                                    int rows, int cols)
{
    /* One block per output row, threads walk the row. */
    const int row = blockIdx.x;
    const float *wr = w + (size_t)row * cols;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < cols; k += blockDim.x)
        acc += wr[k] * x[k];
    __shared__ float sdata[256];
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sdata[0];
}

WUBU_EXPORT int wubu_cuda_available(void)
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count > 0 ? 1 : 0;
}

WUBU_EXPORT int wubu_cuda_device_name(char *out, int cap)
{
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return -1;
    if (out && cap > 0) snprintf(out, (size_t)cap, "%s (cc %d.%d, %d MiB)",
                                 prop.name, prop.major, prop.minor,
                                 (int)(prop.totalGlobalMem >> 20));
    return 0;
}

/* y[rows] = W[rows][cols] * x[cols], F32, device-resident W and x. */
WUBU_EXPORT int wubu_cuda_sgemv(const float *d_w, const float *d_x, float *d_y,
                                int rows, int cols)
{
    if (!d_w || !d_x || !d_y || rows <= 0 || cols <= 0) return -1;
    if (cols > 4096) return -2;   /* block reduction is fixed at 256 lanes */
    sgemv_kernel<<<rows, 256>>>(d_w, d_x, d_y, rows, cols);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -3;
}

/* VRAM budget arithmetic, mirroring atomic-engine's fit step: report what the
 * device has, and decide whether a footprint fits after an explicit reserve. */
WUBU_EXPORT int wubu_cuda_meminfo(unsigned long long *total, unsigned long long *freeb)
{
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) return -1;
    if (freeb) *freeb = (unsigned long long)f;
    if (total) *total = (unsigned long long)t;
    return 0;
}

/* 1 = fits (weights + kv + compute <= free - reserve), 0 = does not, <0 = error. */
WUBU_EXPORT int wubu_cuda_fit(unsigned long long weights, unsigned long long kv,
                              unsigned long long compute, unsigned long long reserve)
{
    unsigned long long freeb = 0;
    if (wubu_cuda_meminfo(NULL, &freeb) != 0) return -1;
    if (freeb <= reserve) return 0;
    const unsigned long long usable = freeb - reserve;
    const unsigned long long need = weights + kv + compute;
    /* guard against overflow of the sum */
    if (need < weights) return 0;
    return need <= usable ? 1 : 0;
}

/* Host-side convenience for the bridge test: upload, compute, download. */
WUBU_EXPORT int wubu_cuda_sgemv_host(const float *h_w, const float *h_x, float *h_y,
                                     int rows, int cols)
{
    if (!h_w || !h_x || !h_y || rows <= 0 || cols <= 0) return -1;
    float *d_w = NULL, *d_x = NULL, *d_y = NULL;
    const size_t wb = (size_t)rows * cols * sizeof(float);
    int rc = -1;
    if (cudaMalloc(&d_w, wb) != cudaSuccess) goto done;
    if (cudaMalloc(&d_x, (size_t)cols * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&d_y, (size_t)rows * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMemcpy(d_w, h_w, wb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(d_x, h_x, (size_t)cols * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (wubu_cuda_sgemv(d_w, d_x, d_y, rows, cols) != 0) goto done;
    if (cudaMemcpy(h_y, d_y, (size_t)rows * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    if (d_w) cudaFree(d_w);
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    return rc;
}
