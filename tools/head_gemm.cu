// head_gemm.cu - upload the head's single-f16 resident and time the cuBLAS FP16 f32-accum matvec.
// Reads: <in.f16.bin> = dims0(i64) dims1(i64) + rows*cols f16 (the head: 2304 in x 201024 out).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA ERR %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define CB(x) do{cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS ERR %d @%d\n",(int)s,__LINE__);return 1;}}while(0)
int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: head_gemm <in.f16.bin> [xN]\n");return 2;}
    FILE*f=fopen(argv[1],"rb"); if(!f){perror("open");return 3;}
    long long r0,r1; fread(&r0,8,1,f); fread(&r1,8,1,f);
    long long ne=r0*r1;
    __half*hf=(__half*)malloc((size_t)ne*2);
    size_t got=fread(hf,2,(size_t)ne,f); fclose(f);
    printf("head %lld x %lld  f16 = %.1f MiB (read %zu elems)\n",r0,r1,(double)ne*2/1048576.0,got);
    /* orientation: buffer is [dims0=in= r0][dims1=out= r1], element(in,out)@ in*r1+out. col : W(in,out). */
    int in=(int)r0, out=(int)r1, K=in;
    __half *dW,*dh,*dC; CK(cudaMalloc(&dW,ne*2)); CK(cudaMalloc(&dh, K*2)); CK(cudaMalloc(&dC, out*4));
    CK(cudaMemcpy(dW,hf,ne*2,cudaMemcpyHostToDevice));
    CK(cudaMemset(dh,0,K*2));  // zero activation: timing only, correctness is the dequant tool's job
    cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetMathMode(h,CUBLAS_DEFAULT_MATH));
    __half halo=__float2half(1), hbeta=__float2half(0);
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    // C(out x 1) = W^T(x_out x in) * x -> cublas: OP_T on the stored (in x out= K x M? ) 
    // stored buffer is [in][out]; treat as (out=cols) with leading dim in=K. gemm C(M=out,N=1,K=in), A=Wbuf, OP_T.
    int M=out, N=1, KK=K;
    int iters=20; double best=1e300;
    for(int it=0; it<iters; ++it){
        CK(cudaEventRecord(e0,0));
        cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, KK,
            &halo, dW, CUDA_R_16F, KK, dh, CUDA_R_16F, KK, &hbeta, dC, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        CB(st); CK(cudaEventRecord(e1,0)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms,e0,e1));
        if(ms<best) best=ms;
    }
    float flop=2.0f*M*N*KK; // =2*out*K
    printf("resident f16 head matvec: %.4f ms  (%.2f GFLOP -> %.1f GFLOP/s)\n",best, flop/1e9, (flop/best)/1e6);
    size_t tot=0,freeb=0; CK(cudaMemGetInfo(&tot,&freeb));
    printf("VRAM free now %.2f GB (resident W %.2f GB)\n",(double)freeb/1e9,(double)ne*2/1e9);
    /* also time the plain f32-accum 50-iter batch to show launch-bound floor */
    cublasDestroy(h); return 0;
}
