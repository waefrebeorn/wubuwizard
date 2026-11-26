#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_tqvec.h"
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb"); if(!f){printf("no file\n");return 1;}
  uint32_t N=0,d=0; if(fread(&N,4,1,f)!=1) return 1; if(fread(&d,4,1,f)!=1) return 1;
  float*X=malloc(sizeof(float)*(size_t)N*d);
  if(fread(X,sizeof(float),(size_t)N*d,f)!=(size_t)N*d){printf("short read\n");return 1;}
  fclose(f);
  const int NQ=200,KK=10,RR=100;
  printf("%-6s %-4s %-11s %-13s %-11s\n","file","bits","recall@10","+rerank@100","bytes/item");
  for(int b=2;b<=8;++b){
    wubu_tqvec*c=wubu_tqvec_init((int)d,b,0xC0FFEEull);
    if(!c){printf("init fail d=%u\n",d);return 1;}
    size_t nb=wubu_tqvec_code_bytes(c);
    unsigned char*codes=malloc(nb*N);
    for(uint32_t i=0;i<N;++i) wubu_tqvec_encode(c,X+(size_t)i*d,codes+nb*i);
    long hit=0,hitrr=0;
    double*sc=malloc(sizeof(double)*N); int*idx=malloc(sizeof(int)*N);
    for(int qq=0;qq<NQ;++qq){
      const uint32_t qi=(uint32_t)((uint64_t)qq*N/NQ);
      const float*qv=X+(size_t)qi*d;
      for(uint32_t i=0;i<N;++i){ double s=0; for(uint32_t j=0;j<d;++j) s+=(double)qv[j]*X[(size_t)i*d+j]; sc[i]=s; idx[i]=(int)i; }
      sc[qi]=-1e30; /* exclude self */
      for(int a=0;a<KK;++a){int bb=a;for(uint32_t t=a+1;t<N;++t)if(sc[t]>sc[bb])bb=(int)t;
        double ts=sc[a];sc[a]=sc[bb];sc[bb]=ts;int ti=idx[a];idx[a]=idx[bb];idx[bb]=ti;}
      int truth[16]; for(int a=0;a<KK;++a) truth[a]=idx[a];
      for(uint32_t i=0;i<N;++i){ sc[i]=wubu_tqvec_score(c,qv,codes+nb*i); idx[i]=(int)i; }
      sc[qi]=-1e30;
      for(int a=0;a<RR;++a){int bb=a;for(uint32_t t=a+1;t<N;++t)if(sc[t]>sc[bb])bb=(int)t;
        double ts=sc[a];sc[a]=sc[bb];sc[bb]=ts;int ti=idx[a];idx[a]=idx[bb];idx[bb]=ti;}
      for(int a=0;a<KK;++a) for(int t=0;t<KK;++t) if(idx[a]==truth[t]){++hit;break;}
      int rr[150]; for(int a=0;a<RR;++a) rr[a]=idx[a];
      double rs[150]; for(int a=0;a<RR;++a){double s=0;const int i=rr[a];for(uint32_t j=0;j<d;++j)s+=(double)qv[j]*X[(size_t)i*d+j];rs[a]=s;}
      for(int a=0;a<KK;++a){int bb=a;for(int t=a+1;t<RR;++t)if(rs[t]>rs[bb])bb=t;
        double ts=rs[a];rs[a]=rs[bb];rs[bb]=ts;int ti=rr[a];rr[a]=rr[bb];rr[bb]=ti;}
      for(int a=0;a<KK;++a) for(int t=0;t<KK;++t) if(rr[a]==truth[t]){++hitrr;break;}
    }
    printf("%-6s %-4d %-11.3f %-13.3f %-11zu\n",argv[2],b,(double)hit/(NQ*KK),(double)hitrr/(NQ*KK),nb);
    free(codes); free(sc); free(idx); wubu_tqvec_free(c);
  }
  free(X); return 0;
}
