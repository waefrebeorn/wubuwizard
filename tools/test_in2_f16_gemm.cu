// bench_gemm_tc.cu — Turing tensor-core reality check on the H3 GEMM shapes.
// Measures cuBLAS FP16 tensor-core rate at FP32 accumulate vs FP16 accumulate,
// and BF16 on the same shapes (Turing has no BF16 tensor cores -> expect a fallback).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA ERR %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)
#define CB(x) do{ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS ERR %d @%d\n",(int)s,__LINE__);return 1;} }while(0)

struct Shape { const char* name; int M, N, K; };

// Inkling-2 dense-tier shapes (the target of the f16 lane): attention/ffn dense GEMM
static Shape shapes[] = {
    {"in2 attn 2304x2304x2304", 2304, 2304, 2304},
    {"in2 ffn  2304x2304x9216", 2304, 2304, 9216},
    {"in2 down 2304x9216x2304", 2304, 9216, 2304},
};

static double bench(cublasHandle_t h, void* A, void* B, void* C,
                    cudaDataType_t ab, cudaDataType_t c_type,
                    cublasComputeType_t comp, int M, int N, int K, int iters) {
    float alpha = 1.0f, beta = 0.0f;
    /* CUBLAS_COMPUTE_16F reads alpha/beta as HALF pointers. Passing a float 1.0f
     * (0x3F800000) makes the low half 0x0000 = 0.0, and BLAS then short-circuits
     * "alpha == 0" into a no-op that still returns SUCCESS — an instant GEMM with
     * an impossible TFLOPS figure. */
    __half h_alpha = __float2half(1.0f), h_beta = __float2half(0.0f);
    const void* palpha = (comp == CUBLAS_COMPUTE_16F) ? (const void*)&h_alpha : (const void*)&alpha;
    const void* pbeta  = (comp == CUBLAS_COMPUTE_16F) ? (const void*)&h_beta  : (const void*)&beta;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
    // C_cm(N,M) = B(N,K) * A(M,K)^T
    for (int i = 0; i < 3 && st == CUBLAS_STATUS_SUCCESS; i++)
        st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, palpha,
                     B, ab, N, A, ab, M, pbeta, C, c_type, N, comp, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) { printf("  [cublas status %d on warmup]", (int)st); cudaEventDestroy(e0); cudaEventDestroy(e1); return -1.0; }
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) {
        st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, palpha,
                     B, ab, N, A, ab, M, pbeta, C, c_type, N, comp, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) break;
    }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    if (st != CUBLAS_STATUS_SUCCESS) { printf("  [cublas status %d mid-run]", (int)st); cudaEventDestroy(e0); cudaEventDestroy(e1); return -1.0; }
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    /* Cross-check the event timer with a host clock: an impossible TFLOPS figure
     * with status SUCCESS means the work was skipped, not that the timer drifted. */
    {
        struct timespec ta, tb;
        timespec_get(&ta, TIME_UTC);
        for (int i = 0; i < iters; i++)
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, palpha,
                         B, ab, N, A, ab, M, pbeta, C, c_type, N, comp, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
        timespec_get(&tb, TIME_UTC);
        double host_ms = ((tb.tv_sec - ta.tv_sec) * 1000.0) + ((tb.tv_nsec - ta.tv_nsec) / 1e6);
        if (host_ms > ms * 3.0 + 1.0)
            printf("  [timer mismatch: event %.3f ms vs host %.3f ms]", ms, host_ms);
    }
    double flops = 2.0 * (double)M * N * K * iters;
    return flops / (ms * 1e-3) / 1e12;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s  SMs=%d  mem %.1f GB\n\n", p.name, p.multiProcessorCount, p.totalGlobalMem / 1e9);
    int maxM = 2304, maxN = 9216, maxK = 9216;
    size_t nA = (size_t)maxM * maxK, nB = (size_t)maxN * maxK, nC = (size_t)maxM * maxN;

    void *A16 = 0, *B16 = 0, *Abf = 0, *Bbf = 0, *C16 = 0, *C32 = 0;
    CK(cudaMalloc(&A16, nA * 2)); CK(cudaMalloc(&B16, nB * 2));
    CK(cudaMalloc(&Abf, nA * 2)); CK(cudaMalloc(&Bbf, nB * 2));
    CK(cudaMalloc(&C16, nC * 2)); CK(cudaMalloc(&C32, nC * 4));
    CK(cudaMemset(A16, 0x11, nA * 2)); CK(cudaMemset(B16, 0x11, nB * 2));
    CK(cudaMemset(Abf, 0x11, nA * 2)); CK(cudaMemset(Bbf, 0x11, nB * 2));

    cublasHandle_t h; CB(cublasCreate(&h));
    CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

    printf("%-24s %14s %14s %14s\n", "shape", "FP16/f32acc", "FP16/f16acc", "BF16/f32acc");
    printf("%-24s %14s %14s %14s\n", "", "TFLOPS", "TFLOPS", "TFLOPS");
    for (size_t s = 0; s < sizeof(shapes)/sizeof(shapes[0]); s++) {
        Shape sh = shapes[s];
        int iters = 10;
        double f32acc = bench(h, A16, B16, C32, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, sh.M, sh.N, sh.K, iters);
        double f16acc = bench(h, A16, B16, C16, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, sh.M, sh.N, sh.K, iters);
        double bf     = bench(h, Abf, Bbf, C32, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, sh.M, sh.N, sh.K, iters);
        printf("%-24s", sh.name);
        if (f32acc >= 0) printf(" %14.2f", f32acc); else printf(" %14s", "FAILED");
        if (f16acc >= 0) printf(" %14.2f", f16acc); else printf(" %14s", "FAILED");
        if (bf     >= 0) printf(" %14.2f", bf);     else printf(" %14s", "FAILED");
        printf("\n");
        fflush(stdout);
    }
    printf("\nreference: 2080 SUPER FP32 core peak 13.45 TFLOPS; Turing FP16 tensor\n"
           "spec 44.6 TFLOPS (f32 accum) / 89.2 (f16 accum). Current Vulkan FP32\n"
           "engine GEMM measured earlier at ~3.6 TFLOPS.\n");
    cublasDestroy(h);
    return 0;
}
