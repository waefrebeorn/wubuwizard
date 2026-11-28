/* f16_ring_test.c -- measure the bounded f16 VRAM ring: dlopen the DLL, drive ALL tensors through
 * wubu_cuda_resident_f16_matvec for several "token" passes, and report per-pass time + ups/reuse/evict/used
 * from wubu_cuda_f16_stats. With IN2_F16_BUDGET_MB < 3.22 GB the ring must evict and re-upload -> this shows
 * whether the dense tier streams or thrashes. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <time.h>
typedef int (*mv_t)(long long, const void*, unsigned long long, const float*, int, float*, int);
typedef int (*stats_t)(long long*, long long*, long long*, long long*, int*);
typedef int (*free_t)(void);
struct T { long long id, K, O; unsigned short* w; };
static int cmpidx(const void*a,const void*b){ long long x=*(long long*)a,y=*(long long*)b; return x<y?-1:x>y; }
int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: f16_ring_test <dll> <f16dir> [passes=3]\n");return 2;}
    int passes=argc>3?atoi(argv[3]):3;
    HMODULE h=LoadLibraryA(argv[1]); if(!h){printf("load fail %lu\n",(unsigned long)GetLastError());return 3;}
    mv_t mv=(mv_t)(void*)GetProcAddress(h,"wubu_cuda_resident_f16_matvec");
    stats_t st=(stats_t)(void*)GetProcAddress(h,"wubu_cuda_f16_stats");
    free_t fr=(free_t)(void*)GetProcAddress(h,"wubu_cuda_vram_free_mb");
    if(!mv||!st){printf("missing export\n");return 4;}
    WIN32_FIND_DATAA fd; char pat[700]; snprintf(pat,sizeof pat,"%s\\*.f16",argv[2]);
    HANDLE hd=FindFirstFileA(pat,&fd); if(hd==INVALID_HANDLE_VALUE){printf("no f16\n");return 5;}
    struct T*ts=NULL; int n=0,cap=0; long long maxO=0,maxK=0;
    do{ const char*cf=fd.cFileName; if(cf[0]=='.'||strstr(cf,"_ref.f32"))continue;
        char p[900]; snprintf(p,sizeof p,"%s\\%s",argv[2],cf);
        FILE*f=fopen(p,"rb"); if(!f)continue; long long d0,d1; fread(&d0,8,1,f); fread(&d1,8,1,f);
        long long nel=d0*d1; unsigned short*w=malloc((size_t)nel*2); fread(w,2,(size_t)nel,f); fclose(f);
        if(n==cap){cap=cap?cap*2:512; ts=realloc(ts,cap*sizeof(struct T));}
        ts[n].id=strtoll(cf+3,NULL,10);   /* filename is idxNNNN_<name>.f16 -> parse NNNN (atoi("idx..")==0!) */ ts[n].K=d0; ts[n].O=d1; ts[n].w=w; if(d1>maxO)maxO=d1; if(d0>maxK)maxK=d0; n++;
    }while(FindNextFileA(hd,&fd)); FindClose(hd);
    printf("loaded %d tensors, maxK=%lld maxO=%lld\n",n,maxK,maxO);
    float*x=malloc((size_t)maxK*4),*y=malloc((size_t)maxO*4);
    srand(5); for(long long i=0;i<maxK;i++) x[i]=((rand()/(float)RAND_MAX)-0.5f)*2.f;
    for(int pass=0; pass<passes; ++pass){
        struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
        for(int i=0;i<n;i++){ mv(ts[i].id, ts[i].w, (unsigned long long)ts[i].K*ts[i].O*2, x, (int)ts[i].K, y, (int)ts[i].O); }
        clock_gettime(CLOCK_MONOTONIC,&t1);
        double ms=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)/1e6;
        long long ups,reuse,evict,usedmb; int slabs; st(&ups,&reuse,&evict,&usedmb,&slabs);
        int fm = fr?fr():-1;
        printf("pass %d: %7.2f ms  ups=%lld reuse=%lld evict=%lld used=%lldMB slabs=%d vram_free=%dMB\n",
            pass, ms, ups, reuse, evict, usedmb, slabs, fm);
    }
    return 0;
}
