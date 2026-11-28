// gemm_mscale.cu - the arithmetic-intensity law at the ENGINE's shapes: does batching M unlock tensor cores?
// cuBLAS f16xf16->f32 GEMM, K=2304 (d_model), N in {3072 attn, 8192 ffn-ish}, M in {1..256}.
// Reports ms, TFLOPS, and achieved GB/s (bytes = M*K + K*N + M*N, f16). At M=1 this is a GEMV: bound by the
// N*K weight read; at M>=16 the 16x16 tensor-core tile engages and TFLOPS should climb.
#include <cstdio>
#include <cstdlib>
#include <time.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define CB(x) do{cublasStatus_t _s=(x); if(_s!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS %d @%d\n",(int)_s,__LINE__);exit(1);}}while(0)
static int _qf0=0; static long long _qf=0;
static double nowms(){ if(!_qf0){LARGE_INTEGER f;QueryPerformanceFrequency(&f);_qf=f.QuadPart;_qf0=1;} LARGE_INTEGER t;QueryPerformanceCounter(&t); return (double)t.QuadPart*1000.0/(double)_qf; }
static void run(int K,int N,int M,cublasHandle_t h,__half*A,__half*W,float*C){
    float one=1.f,zero=0.f; const int reps = M<=4?20:8;
    // warm
    CB(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&one,W,CUDA_R_16F,K,A,CUDA_R_16F,K,&zero,C,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CK(cudaDeviceSynchronize());
    double t0=nowms();
    for(int r=0;r<reps;r++) CB(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&one,W,CUDA_R_16F,K,A,CUDA_R_16F,K,&zero,C,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CK(cudaDeviceSynchronize()); double ms=(nowms()-t0)/reps;
    double flop=2.0*(double)M*N*K, bytes=2.0*((double)M*K+(double)K*N+(double)M*N);
    printf("  M=%4d  %7.3f ms  %8.1f GFLOP/s  %7.1f GB/s\n",M,ms,flop/(ms/1e3)/1e9,bytes/(ms/1e3)/1e9);
}
int main(){
    const int K=2304; int Ns[2]={3072,8192};
    cublasHandle_t h; CB(cublasCreate(&h));
    for(int ni=0;ni<2;ni++){
        int N=Ns[ni];
        __half *W,*A; float*C; CK(cudaMalloc(&W,(size_t)K*N*2)); CK(cudaMalloc(&A,(size_t)K*256*2)); CK(cudaMalloc(&C,(size_t)256*N*4));
        CK(cudaMemset(W,0,(size_t)K*N*2)); CK(cudaMemset(A,0,(size_t)K*256*2));
        printf("K=%d N=%d (weight bytes = %.1f MB)\n",K,N,(double)K*N*2/1e6);
        int Ms[8]={1,2,4,8,16,32,64,128};
        for(int i=0;i<8;i++) run(K,N,Ms[i],h,A,W,C);
        CK(cudaFree(W));CK(cudaFree(A));CK(cudaFree(C));
    }
    return 0;
}
