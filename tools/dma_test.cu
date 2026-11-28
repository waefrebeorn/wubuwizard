// dma_test.cu - quantify what the f16 lane did WRONG (DMA anti-patterns) on THIS box.
//  (1) pageable vs pinned H2D upload bandwidth (the driver bounce-buffer tax)
//  (2) 2000 small GEMMs (2304x3072, N=1 = a decode projection) with: per-call cudaDeviceSynchronize (what I did),
//      no sync, and replayed from a CUDA graph (launch overhead removed)
#include <cstdio>
#include <cstdlib>
#include <time.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define CB(x) do{cublasStatus_t _st_=(x); if(_st_!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS %d @%d\n",(int)_st_,__LINE__);exit(1);}}while(0)
static int _qpf=0; static long long _qf=0;
static double nowms(){ if(!_qpf){LARGE_INTEGER f;QueryPerformanceFrequency(&f);_qf=f.QuadPart;_qpf=1;} LARGE_INTEGER t;QueryPerformanceCounter(&t); return (double)t.QuadPart*1000.0/(double)_qf; }
int main(){
    const size_t SZ = 64ull*1024*1024;
    void *pg = malloc(SZ), *pn=NULL; CK(cudaHostAlloc(&pn,SZ,cudaHostAllocDefault));
    for(size_t i=0;i<SZ/sizeof(int);++i) ((int*)pg)[i]=i;
    memcpy(pn,pg,SZ);
    __half *d; CK(cudaMalloc(&d,SZ));
    double t0=nowms(); for(int r=0;r<5;r++) CK(cudaMemcpy(d,pg,SZ,cudaMemcpyHostToDevice)); double tp=nowms()-t0;
    t0=nowms(); for(int r=0;r<5;r++) CK(cudaMemcpy(d,pn,SZ,cudaMemcpyHostToDevice)); double tpn=nowms()-t0;
    printf("H2D 64MB x5: pageable %.0f ms (%.1f GB/s) | pinned %.0f ms (%.1f GB/s)\n",
        tp,tp>0?SZ*5/(tp/1e3)/1e9:0, tpn,tpn>0?SZ*5/(tpn/1e3)/1e9:0);
    // small GEMM loop: 2304x3072 N=1
    const int K=2304,O=3072; __half *W,*x; CK(cudaMalloc(&W,(size_t)K*O*2)); CK(cudaMalloc(&x,K*2));
    float *y; CK(cudaMalloc(&y,O*4)); cublasHandle_t h; CB(cublasCreate(&h));
    float one=1.f,zero=0.f; const int N=2000;
    // (A) per-call device synchronize (what the f16 lane does)
    t0=nowms();
    for(int i=0;i<N;i++){ CB(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,O,1,K,&one,W,CUDA_R_16F,K,x,CUDA_R_16F,K,&zero,y,CUDA_R_32F,O,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)); CK(cudaDeviceSynchronize()); }
    double ta=nowms()-t0;
    // (B) enqueue only, single sync at end
    t0=nowms();
    for(int i=0;i<N;i++) CB(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,O,1,K,&one,W,CUDA_R_16F,K,x,CUDA_R_16F,K,&zero,y,CUDA_R_32F,O,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CK(cudaDeviceSynchronize()); double tb=nowms()-t0;
    // (C) CUDA graph: capture N GEMMs, replay
    cudaStream_t s; CK(cudaStreamCreate(&s)); CB(cublasSetStream(h,s));
    cudaGraph_t g; cudaGraphExec_t ge; CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    for(int i=0;i<N;i++) CB(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,O,1,K,&one,W,CUDA_R_16F,K,x,CUDA_R_16F,K,&zero,y,CUDA_R_32F,O,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CK(cudaStreamEndCapture(s,&g)); CK(cudaGraphInstantiate(&ge,g,0));
    t0=nowms(); CK(cudaGraphLaunch(ge,s)); CK(cudaStreamSynchronize(s)); double tc=nowms()-t0;
    printf("%d small GEMMs (2304x3072 N=1):\n",N);
    printf("  (A) per-call deviceSynchronize : %7.1f ms  (%.1f us/call)  <- what the f16 lane did\n",ta,ta*1000/N);
    printf("  (B) enqueue-only, 1 sync       : %7.1f ms  (%.1f us/call)\n",tb,tb*1000/N);
    printf("  (C) CUDA graph replay          : %7.1f ms  (%.1f us/call)\n",tc,tc*1000/N);
    printf("  A/B = %.1fx, A/C = %.1fx\n",tb>0?ta/tb:0,tc>0?ta/tc:0);
    return 0;
}
