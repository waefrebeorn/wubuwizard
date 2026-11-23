/*
 * gpu_smoke.cu -- GPU gate: device identity + a real in-tree CUDA kernel.
 * Verifies wubu_cuda_silu against the host formula and checks cuBLAS init.
 */
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void silu_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] / (1.0f + expf(-x[i]));
}

int main() {
    int dev = 0, count = 0;
    cudaDeviceProp p;
    cublasHandle_t h = nullptr;
    const int n = 4096;
    float *hx = nullptr, *hy = nullptr, *dx = nullptr, *dy = nullptr;
    float maxerr = 0.0f;
    int failures = 0;

    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        fprintf(stderr, "FAIL no CUDA device\n");
        return 1;
    }
    cudaGetDeviceProperties(&p, dev);
    printf("GPU device=%s cc=%d.%d vram=%zu bytes\n",
           p.name, p.major, p.minor, (size_t)p.totalGlobalMem);
    printf("GPU sm_89_expected=%d\n", p.major == 8 && p.minor == 9);
    if (!(p.major == 8 && p.minor == 9)) failures++;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL cublasCreate\n");
        return 1;
    }

    hx = new float[n];
    hy = new float[n];
    for (int i = 0; i < n; i++) hx[i] = ((i % 81) - 40) * 0.05f;
    cudaMalloc(&dx, n * sizeof(float));
    cudaMalloc(&dy, n * sizeof(float));
    cudaMemcpy(dx, hx, n * sizeof(float), cudaMemcpyHostToDevice);
    silu_kernel<<<(n + 255) / 256, 256>>>(dx, dy, n);
    cudaMemcpy(hy, dy, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    printf("GPU silu kernel err=%s\n", cudaGetErrorString(err));
    if (err != cudaSuccess) failures++;

    for (int i = 0; i < n; i++) {
        float want = hx[i] / (1.0f + expf(-hx[i]));
        float e = fabsf(hy[i] - want);
        if (e > maxerr) maxerr = e;
    }
    printf("GPU silu maxerr=%.8f\n", maxerr);
    if (!(maxerr < 1e-5f)) failures++;

    /* cuBLAS sanity: column-major 2x2 * 2x2. */
    {
        const float a[4] = { 1, 2, 3, 4 }, b[4] = { 5, 6, 7, 8 };
        float c[4] = { 0, 0, 0, 0 };
        float *da = nullptr, *db = nullptr, *dc = nullptr;
        const float alpha = 1.0f, beta = 0.0f;
        cudaMalloc(&da, sizeof(a)); cudaMalloc(&db, sizeof(b)); cudaMalloc(&dc, sizeof(c));
        cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice);
        cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice);
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2,
                    &alpha, db, 2, da, 2, &beta, dc, 2);
        cudaMemcpy(c, dc, sizeof(c), cudaMemcpyDeviceToHost);
        printf("GPU cublas c=[%.0f %.0f %.0f %.0f]\n", c[0], c[1], c[2], c[3]);
        if (!(c[0] == 19 && c[1] == 22 && c[2] == 43 && c[3] == 50)) failures++;
        cudaFree(dc); cudaFree(db); cudaFree(da);
    }

    cudaFree(dy);
    cudaFree(dx);
    cublasDestroy(h);
    delete[] hx;
    delete[] hy;
    if (failures) {
        fprintf(stderr, "gpu_smoke: FAIL (%d)\n", failures);
        return 1;
    }
    printf("gpu_smoke: PASS\n");
    return 0;
}
