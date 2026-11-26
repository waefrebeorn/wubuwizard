/* Does 4-bit KV survive contact with softmax? Measures |softmax(q.K^T) - softmax(q.Khat^T)| per query over 128 keys,
 * which is the quantity that actually moves logits. q8_0 (8.5 b/elem) is the incumbent baseline. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "wubu_kv_q8.h"
#include "wubu_tqvec.h"
static uint64_t S=777;
static double ur(void){S^=S>>12;S^=S<<25;S^=S>>27;return (double)((S*2685821657736338717ULL)>>11)/9007199254740992.0;}
static double nr(void){double u=ur(),v=ur();if(u<1e-12)u=1e-12;return sqrt(-2*log(u))*cos(6.283185307179586*v);}
#define D 128
#define NK 128
static void softmax(double*s,int n){double m=s[0];for(int i=1;i<n;++i)if(s[i]>m)m=s[i];double t=0;for(int i=0;i<n;++i){s[i]=exp(s[i]-m);t+=s[i];}for(int i=0;i<n;++i)s[i]/=t;}
int main(void){
  setvbuf(stdout,NULL,_IONBF,0);
  float K[NK][D],KH[NK][D],q[D];
  /* keys with realistic spread: a few large-norm "sink" keys and many small ones (attention sinks are real) */
  for(int j=0;j<NK;++j){ double n=0; double sc=(j<8)?3.0:1.0;
    for(int i=0;i<D;++i){ K[j][i]=(float)(nr()*sc*(0.5+1.5*(double)(i%7)/7.0)); n+=(double)K[j][i]*K[j][i]; }
    n=sqrt(n); for(int i=0;i<D;++i) K[j][i]=(float)(K[j][i]/n); }
  printf("%-18s %-9s %-11s %-11s %-11s\n","quantiser","b/elem","tv_dist","max_dp","top1_moved");
  for(int b=4;b<=8;b+=2){
    wubu_tqvec*tq=wubu_tqvec_init(D,b,0xC0FFEEull); size_t nb=wubu_tqvec_code_bytes(tq);
    unsigned char*codes=malloc(nb*NK); for(int j=0;j<NK;++j) wubu_tqvec_encode(tq,K[j],codes+nb*j);
    double tv=0,mdp=0; int moved=0; const int Q=100;
    for(int t=0;t<Q;++t){
      double nq=0; for(int i=0;i<D;++i){ q[i]=(float)nr(); nq+=(double)q[i]*q[i]; } nq=sqrt(nq);
      for(int i=0;i<D;++i) q[i]=(float)(q[i]/nq);
      double a[NK],bb[NK];
      for(int j=0;j<NK;++j){ double s=0; for(int i=0;i<D;++i) s+=(double)q[i]*K[j][i]; a[j]=s/sqrt((double)D); }
      for(int j=0;j<NK;++j) bb[j]=wubu_tqvec_score(tq,q,codes+nb*j)/sqrt((double)D);
      int ta=0,tb=0; for(int j=1;j<NK;++j){ if(a[j]>a[ta])ta=j; if(bb[j]>bb[tb])tb=j; }
      if(ta!=tb)++moved;
      double dmax=0; for(int j=0;j<NK;++j){ double d=fabs(a[j]-bb[j]); if(d>dmax)dmax=d; } if(dmax>mdp)mdp=dmax;
      softmax(a,NK); softmax(bb,NK);
      double s=0; for(int j=0;j<NK;++j) s+=fabs(a[j]-bb[j]); tv+=0.5*s;
    }
    printf("%-18s %-9d %-11.6f %-11.6f %-11.3f\n","turboquant",b,tv/Q,mdp,(double)moved/Q);
    free(codes); wubu_tqvec_free(tq);
  }
  { size_t nb=kvq8_row_bytes(D); unsigned char*codes=malloc(nb*NK);
    for(int j=0;j<NK;++j) kvq8_quantize_row(K[j],codes+nb*j,D);
    double tv=0,mdp=0; int moved=0; const int Q=100;
    for(int t=0;t<Q;++t){
      double nq=0; for(int i=0;i<D;++i){ q[i]=(float)nr(); nq+=(double)q[i]*q[i]; } nq=sqrt(nq);
      for(int i=0;i<D;++i) q[i]=(float)(q[i]/nq);
      double a[NK],bb[NK];
      for(int j=0;j<NK;++j){ double s=0; for(int i=0;i<D;++i) s+=(double)q[i]*K[j][i]; a[j]=s/sqrt((double)D); }
      for(int j=0;j<NK;++j){ kvq8_dequantize_row(codes+nb*j,KH[j],D); double s=0; for(int i=0;i<D;++i) s+=(double)q[i]*KH[j][i]; bb[j]=s/sqrt((double)D); }
      int ta=0,tb=0; for(int j=1;j<NK;++j){ if(a[j]>a[ta])ta=j; if(bb[j]>bb[tb])tb=j; }
      if(ta!=tb)++moved;
      double dmax=0; for(int j=0;j<NK;++j){ double d=fabs(a[j]-bb[j]); if(d>dmax)dmax=d; } if(dmax>mdp)mdp=dmax;
      softmax(a,NK); softmax(bb,NK);
      double s=0; for(int j=0;j<NK;++j) s+=fabs(a[j]-bb[j]); tv+=0.5*s;
    }
    printf("%-18s %-9.1f %-11.6f %-11.6f %-11.3f\n","q8_0 (current KV)",8.5,tv/Q,mdp,(double)moved/Q);
    free(codes);
  }
  return 0;
}
