// lane_check.cu - chase 1:1 on the f16 lane. Load the single-f16 head (the lane's weight) AND the exact-f32
// reference head; for several activations compute lane logits (f16 cuBLAS, f32 accum) vs reference logits
// (double-accum exact f32 dequant) and report logit error + GREEDY ARGMAX AGREEMENT. Greedy identity is what
// "1:1" means for generation: if argmax ever differs, the lane is not token-identical.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define CB(x) do{cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS %d @%d\n",(int)s,__LINE__);exit(1);}}while(0)
int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: lane_check <head.f16> <head_ref.f32> [nAct]\n");return 2;}
    int nAct = argc>3?atoi(argv[3]):8;
    // read f16 head (skip 16-byte header)
    FILE*f=fopen(argv[1],"rb"); if(!f){perror("f16");return 3;}
    long long d0,d1; fread(&d0,8,1,f); fread(&d1,8,1,f); long long ne=d0*d1; long long K=d0, O=d1;
    __half*wh=(__half*)malloc((size_t)ne*2); fread(wh,2,(size_t)ne,f); fclose(f);
    // read f32 ref (raw, no header)
    FILE*r=fopen(argv[2],"rb"); if(!r){perror("f32");return 4;}
    float*rf=(float*)malloc((size_t)ne*4); fread(rf,4,(size_t)ne,r); fclose(r);
    printf("head: K(in)=%lld O(out)=%lld ne=%lld  (f16 + f32 ref loaded)\n",K,O,ne);
    __half*dW; CK(cudaMalloc(&dW,(size_t)ne*2)); CK(cudaMemcpy(dW,wh,(size_t)ne*2,cudaMemcpyHostToDevice));
    __half*dx; CK(cudaMalloc(&dx,(size_t)K*2));
    float*dy; CK(cudaMalloc(&dy,(size_t)O*4));
    float*yh=(float*)malloc((size_t)O*4);
    cublasHandle_t h; CB(cublasCreate(&h));
    float one=1.0f,zero=0.0f;
    double worst_rel=0, worst_abs=0; int argmax_ok=0;
    for(int it=0;it<nAct;it++){
        float*x=(float*)malloc((size_t)K*4); __half*xh=(__half*)malloc((size_t)K*2);
        // realistic-ish activation: random normal-ish, plus a few peaked entries (as real hidden states are)
        srand(2024+it*7);
        for(long long i=0;i<K;i++){ float v=((rand()/(float)RAND_MAX)-0.5f)*2.0f; x[i]=v; }
        for(int p=0;p<8;p++){ x[rand()%K]+=6.0f; }
        for(long long i=0;i<K;i++) xh[i]=__float2half_rn(x[i]);
        CK(cudaMemcpy(dx,xh,(size_t)K*2,cudaMemcpyHostToDevice));
        CB(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)O,1,(int)K,&one,dW,CUDA_R_16F,(int)K,
            dx,CUDA_R_16F,(int)K,&zero,dy,CUDA_R_32F,(int)O,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CK(cudaMemcpy(yh,dy,(size_t)O*4,cudaMemcpyDeviceToHost));
        // reference: exact f32 dequant dot with double accum (ground truth)
        double*ref=(double*)malloc((size_t)O*8);
        for(long long o=0;o<O;o++){ const float*wr=rf+o*K; double s=0; for(long long k=0;k<K;k++) s+=(double)wr[k]*(double)x[k]; ref[o]=s; }
        long long amax_l=0,amax_r=0; double maxabs=0,maxrel=0;
        for(long long o=0;o<O;o++){ double d=fabs((double)yh[o]-ref[o]); if(d>maxabs)maxabs=d;
            double den=fabs(ref[o]); double rel= den>1e-3? d/den : 0; if(rel>maxrel)maxrel=rel;
            if(yh[o]>yh[amax_l])amax_l=o; if(ref[o]>ref[amax_r])amax_r=o; }
        int ok=(amax_l==amax_r); argmax_ok+=ok;
        // flip risk: ref top-2 margin vs the lane logit error AT THE TOP (not the global max, which lands on
        // large-magnitude mid-rank outputs). What matters for greedy is the error on the two contenders.
        double top1=-1e300,top2=-1e300; long long i1=-1,i2=-1;
        for(long long o=0;o<O;o++){ if(ref[o]>top1){top2=top1;i2=i1;top1=ref[o];i1=o;} else if(ref[o]>top2){top2=ref[o];i2=o;} }
        double margin=top1-top2;
        double e1=fabs((double)yh[i1]-ref[i1]), e2=fabs((double)yh[i2]-ref[i2]);
        if(maxabs>worst_abs)worst_abs=maxabs; if(maxrel>worst_rel)worst_rel=maxrel;
        printf("act %d: argmax %s  margin=%.3f  err@top1=%.4f err@top2=%.4f  globalmax=%.3f\n",
            it,ok?"MATCH":"DIVERGE",margin,e1,e2,maxabs);
        free(x);free(xh);free(ref);
    }
    printf("SUMMARY %d acts: greedy argmax %d/%d match, worst|dlogit|=%.4f worst rel=%.2e\n",nAct,argmax_ok,nAct,worst_abs,worst_rel);
    return argmax_ok==nAct?0:1;
}
