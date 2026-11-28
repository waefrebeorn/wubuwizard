// resident_dense.cu - upload the full single-f16 dense weight set (3.22 GB) and time the dense tier
// as one N=1 cuBLAS matvec per tensor (decode M=1). Measure upload + sum of matvec times -> dense-tier floor.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <windows.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA ERR %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define CB(x) do{cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS %d @%d\n",(int)s,__LINE__);return 1;}}while(0)
struct T{ size_t off; long long in,out; };   // offset in dW (f16), dims
int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: resident_dense <dir> [xIters]\n");return 2;}
    WIN32_FIND_DATAA fd; char pat[1620]; snprintf(pat,sizeof pat,"%s\\*.f16",argv[1]);
    HANDLE hd=FindFirstFileA(pat,&fd); if(hd==INVALID_HANDLE_VALUE){fprintf(stderr,"no f16\n");return 3;}
    int n=0; long long total=0, maxk=0;
    char**names=NULL; long long*ins=(long long*)malloc(0),*outs=(long long*)malloc(0); size_t*szs=(size_t*)malloc(0);
    int cap=0;
    do{ if(fd.cFileName[0]=='.')continue;
        char p[1600]; snprintf(p,sizeof p,"%s\\%s",argv[1],fd.cFileName);
        FILE*f=fopen(p,"rb"); if(!f)continue; long long d0,d1; fread(&d0,8,1,f); fread(&d1,8,1,f);
        long long el=d0*d1; fclose(f);
        if(n==cap){cap=cap?cap*2:512; names=(char**)realloc(names,cap*sizeof(char*));ins=(long long*)realloc(ins,cap*8);outs=(long long*)realloc(outs,cap*8);szs=(size_t*)realloc(szs,cap*8);}
        names[n]=strdup(p); ins[n]=d0; outs[n]=d1; szs[n]=(size_t)el*2; total+=el; if(el>maxk)maxk=el;
        n++;
    } while(FindNextFileA(hd,&fd)); FindClose(hd);
    printf("%d tensors, %lld elements (%.2f GB) maxK %lld\n",n,total,(double)total*2/1e9,maxk);
    __half*dW; CK(cudaMalloc(&dW,(size_t)total*2));
    /* scratch = the N=1 activation vector, len = max IN-dim (contraction), NOT max elements. */
    long long maxin=0; for(int i=0;i<n;i++){ if(ins[i]>maxin)maxin=ins[i]; }
    __half*scratch; CK(cudaMalloc(&scratch,(size_t)maxin*2)); CK(cudaMemset(scratch,0,(size_t)maxin*2));
    cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetMathMode(h,CUBLAS_DEFAULT_MATH));
    // upload all (host->device, one big memcpy per tensor)
    size_t off=0;
    cudaEvent_t u0,u1; CK(cudaEventCreate(&u0));CK(cudaEventCreate(&u1)); CK(cudaEventRecord(u0,0));
    __half*htmp=(__half*)malloc((size_t)total*2);
    size_t hoff=0;
    for(int i=0;i<n;i++){ FILE*f=fopen(names[i],"rb"); fseek(f,16,SEEK_SET); fread((char*)htmp+hoff,2,(size_t)(ins[i]*outs[i]),f); fclose(f); hoff+=(size_t)(ins[i]*outs[i])*2; }
    CK(cudaMemcpy(dW,htmp,(size_t)total*2,cudaMemcpyHostToDevice));
    CK(cudaEventRecord(u1,0)); CK(cudaEventSynchronize(u1));
    float ups; CK(cudaEventElapsedTime(&ups,u0,u1));
    printf("upload %u tensors = %.2f s (%.1f GB/s)\n",n,ups/1000,(double)total*2/(ups/1000)/1e9);
    // time a full pass: one N=1 cuBLAS matvec per tensor
    __half hab=__float2half(1), hbe=__float2half(0);
    float*dC; CK(cudaMalloc(&dC,(size_t)300000*4));
    int iters=argc>2?atoi(argv[2]):3; double best=1e300;
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
    for(int it=0;it<iters;it++){
        off=0;
        CK(cudaEventRecord(e0,0));
        for(int i=0;i<n;i++){
            int M=(int)outs[i], K=(int)ins[i]; cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,M,1,K,
                &hab, dW+off, CUDA_R_16F, K, scratch, CUDA_R_16F, K, &hbe, dC, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            off+=szs[i]/2;
        }
        CK(cudaEventRecord(e1,0)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); if(ms<best)best=ms;
    }
    long long flops=0; for(int i=0;i<n;i++) flops+=2*ins[i]*outs[i];
    printf("dense tier %d matvecs (N=1) best %.2f ms  (read %.2f GB, %.0f GB/s, %.0f GFLOP)\n",
        n,best,(double)total*2/1e9,(double)total*2/best/1e6,(double)flops/1e9);
    size_t fb,tb; CK(cudaMemGetInfo(&fb,&tb));
    printf("VRAM free %.2f GB / resident W %.2f GB\n",(double)fb/1e9,(double)total*2/1e9);
    return 0;
}
