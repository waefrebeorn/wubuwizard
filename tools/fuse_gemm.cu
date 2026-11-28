// fuse_gemm.cu - at decode M=1, does ONE big cuBLAS call beat N separate calls at the same total FLOPs?
// Synthetic timing: A = W(O x K) where O = full dense-out count, K=2304, N=1. Compare 1 fused call vs 20.
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA ERR %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define CB(x) do{cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS %d\n",(int)s);return 1;}}while(0)
static double timeit(cublasHandle_t h, __half*dA, int O, int K, int nCalls){
    __half ha=__float2half(1), hb=__float2half(0); float*dC;
    CK(cudaMalloc(&dC, O*4)); __half*dx; CK(cudaMalloc(&dx,K*2)); CK(cudaMemset(dx,0,K*2));
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0,0));
    int Oi=O/nCalls;
    for(int c=0;c<nCalls;c++)
        CB(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,Oi,1,K,&ha, dA+(size_t)c*Oi*K,CUDA_R_16F,K,
                        dx,CUDA_R_16F,K,&hb, dC+(size_t)c*Oi, CUDA_R_32F,Oi,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CK(cudaEventRecord(e1,0)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); cudaFree(dC); cudaFree(dx); return ms;
}
int main(){
    const int K=2304, OTOT=214000; // ~ head 201024 + attn/gate/up ~13k
    __half*dA; CK(cudaMalloc(&dA,(size_t)OTOT*K*2)); CK(cudaMemset(dA,0,(size_t)OTOT*K*2));
    cublasHandle_t h; CB(cublasCreate(&h));
    double best1=1e18,best20=1e18;
    for(int i=0;i<8;i++){ double t1=timeit(h,dA,OTOT,K,1); if(t1<best1)best1=t1;
                          double t20=timeit(h,dA,OTOT,K,20); if(t20<best20)best20=t20; }
    printf("fused 1 call  O=%d : %.2f ms\n",OTOT,best1);
    printf("20 separate calls (O=%d each) : %.2f ms\n",OTOT/20,best20);
    printf("fusion amortization: %.1fx less time\n",best20/best1);
    return 0;
}
