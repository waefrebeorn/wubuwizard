#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "wubu_tqvec.h"
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb"); uint32_t N,d; if(fread(&N,4,1,f)!=1)return 1; if(fread(&d,4,1,f)!=1)return 1;
  float*X=malloc(sizeof(float)*(size_t)N*d); if(fread(X,4,(size_t)N*d,f)!=(size_t)N*d)return 1; fclose(f);
  printf("N=%u d=%u\n",N,d);
  int bits=atoi(argv[2]);
  wubu_tqvec*c=wubu_tqvec_init((int)d,bits,0xC0FFEEull);
  size_t nb=wubu_tqvec_code_bytes(c); unsigned char*codes=malloc(nb*N);
  for(uint32_t i=0;i<N;++i) wubu_tqvec_encode(c,X+(size_t)i*d,codes+nb*i);
  /* one query: correlation between exact and quantised scores across the corpus */
  double sx=0,sy=0,sxx=0,syy=0,sxy=0; int M=0;
  for(uint32_t i=0;i<N;++i){
    double ex=0,qs=0;
    for(uint32_t j=0;j<d;++j) ex+=(double)X[j]*X[(size_t)i*d+j];
    qs=wubu_tqvec_score(c,X,codes+nb*i);
    sx+=ex; sy+=qs; sxx+=ex*ex; syy+=qs*qs; sxy+=ex*qs; ++M;
  }
  double mx=sx/M,my=sy/M;
  double cov=sxy/M-mx*my, vx=sxx/M-mx*mx, vy=syy/M-my*my;
  printf("bits=%d  corr(exact,quant)=%.5f  sd_exact=%.5f  sd_quant=%.5f  ratio=%.4f\n",
    bits, cov/sqrt(vx*vy), sqrt(vx), sqrt(vy), sqrt(vy/vx));
  wubu_tqvec_free(c); return 0;
}
