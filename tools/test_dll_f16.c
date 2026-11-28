/* test_dll_f16.c - validate the new DLL export wubu_cuda_resident_f16_matvec: dlopen wubu_cuda.dll, run an
 * N=1 f16 matvec on a real materialized tensor, compare against the exact-f32 reference dot. Proves the DLL
 * f16 lane computes correctly before it is wired into the engine. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
typedef int (*f16_mv)(long long, const void*, unsigned long long, const float*, int, float*, int);
int main(int argc,char**argv){
    if(argc<4){fprintf(stderr,"usage: test_dll_f16 <dll> <tensor.f16> <tensor_ref.f32>\n");return 2;}
    HMODULE h=LoadLibraryA(argv[1]); if(!h){printf("load fail %lu\n",(unsigned long)GetLastError());return 3;}
    f16_mv mv=(f16_mv)(void*)GetProcAddress(h,"wubu_cuda_resident_f16_matvec");
    if(!mv){printf("no export\n");return 4;}
    FILE*f=fopen(argv[2],"rb"); long long d0,d1; fread(&d0,8,1,f); fread(&d1,8,1,f); long long K=d0,O=d1;
    unsigned short*wh=malloc((size_t)K*O*2); fread(wh,2,(size_t)K*O,f); fclose(f);
    FILE*r=fopen(argv[3],"rb"); float*rf=malloc((size_t)K*O*4); fread(rf,4,(size_t)K*O,r); fclose(r);
    printf("tensor K=%lld O=%lld\n",K,O);
    float*x=malloc((size_t)K*4); srand(11); for(long long i=0;i<K;i++) x[i]=((rand()/(float)RAND_MAX)-0.5f)*2.0f;
    for(int p=0;p<8;p++) x[rand()%K]+=6.0f;
    float*y=malloc((size_t)O*4);
    int rc=mv(12345,wh,(unsigned long long)K*O*2,x,(int)K,y,(int)O);
    printf("dll rc=%d\n",rc); if(rc){return 5;}
    long long nz=0; for(long long o=0;o<O;o++) if(y[o]!=0) nz++;
    printf("y nonzero=%lld/%lld  y[0..3]=%.4f %.4f %.4f %.4f  x[0..3]=%.4f %.4f %.4f %.4f\n",
        nz,O,y[0],y[1],y[2],y[3],x[0],x[1],x[2],x[3]);
    double maxd=0,mag=0; long long amax=0,aref=0; double best=-1e300,rbest=-1e300;
    for(long long o=0;o<O;o++){ const float*wr=rf+o*K; double s=0; for(long long k=0;k<K;k++) s+=(double)wr[k]*(double)x[k];
        double d=fabs((double)y[o]-s); if(d>maxd)maxd=d; if(fabs(s)>mag)mag=fabs(s);
        if(y[o]>best){best=y[o];amax=o;} if(s>rbest){rbest=s;aref=o;} }
    printf("DLL f16 vs f32 ref: max|d|=%.4f (max|ref|=%.1f) argmax dll=%lld ref=%lld %s\n",
        maxd,mag,amax,aref,amax==aref?"MATCH":"DIVERGE");
    return amax==aref?0:1;
}
