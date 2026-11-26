/* tq_ball_vs_tangent.c - where should the quantiser sit on a Poincare ball?
 *
 * The design says "score in the tangent space at the origin, where the metric is flat". That was argued, not measured.
 * This measures it. Ground truth is always the exact hyperbolic distance on the ball; the two candidate pipelines are
 *
 *   A) quantise the BALL coordinates directly      u -> wubu_tqvec -> u_hat -> d_H(q, u_hat)
 *   B) quantise in the TANGENT space, exp-map back  t = log(u) -> wubu_tqvec -> t_hat -> exp(t_hat) -> d_H(q, .)
 *
 * and the question is which one keeps the hyperbolic neighbour order. Because the Poincare metric diverges as ||u||->1,
 * A should degrade near the boundary and B should not, if the intuition behind the design is right.
 *
 * d_H(u,v) = arcosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))   for curvature c = 1
 *
 * Build: gcc -O2 -std=gnu11 -static -I include -o tqbt.exe tools/tq_ball_vs_tangent.c src/wubu_tqvec.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubu_tqvec.h"

static double nrm(const float *v, int d){ double s=0; for(int i=0;i<d;++i) s+=(double)v[i]*v[i]; return sqrt(s); }

static double dpoinc(const float *u, const float *v, int d)
{
    double nu=0,nv=0,d2=0;
    for(int i=0;i<d;++i){ const double a=u[i],b=v[i]; nu+=a*a; nv+=b*b; d2+=(a-b)*(a-b); }
    if(nu>=1.0) nu=1.0-1e-9;
    if(nv>=1.0) nv=1.0-1e-9;
    const double arg=1.0+2.0*d2/((1.0-nu)*(1.0-nv));
    return acosh(arg<1.0?1.0:arg);
}

/* exp at origin: t (any vector) -> ball, curvature c=1 => u = tanh(||t||/2) * t/||t|| */
static void expmap0(const float *t, float *u, int d)
{
    const double n=nrm(t,d);
    if(n<1e-12){ memset(u,0,sizeof(float)*(size_t)d); return; }
    const double r=tanh(n*0.5);
    for(int i=0;i<d;++i) u[i]=(float)(t[i]/n*r);
}
/* log at origin: u -> tangent */
static void logmap0(const float *u, float *t, int d)
{
    const double n=nrm(u,d);
    if(n<1e-12){ memset(t,0,sizeof(float)*(size_t)d); return; }
    const double c=tanh(n)>0.999999?0.999999:tanh(n);
    const double r=2.0*atanh(c>0.999999?0.999999:c);
    for(int i=0;i<d;++i) t[i]=(float)(u[i]/n*r);
}

int main(int argc,char**argv)
{
    FILE *f=fopen(argv[1],"rb");
    uint32_t N=0,d=0;
    if(!f||fread(&N,4,1,f)!=1||fread(&d,4,1,f)!=1){ printf("bad file\n"); return 1; }
    float *X=malloc(sizeof(float)*(size_t)N*d);
    if(fread(X,sizeof(float),(size_t)N*d,f)!=(size_t)N*d){ printf("short\n"); return 1; }
    fclose(f);

    /* tangent vectors: scale the unit data so radii spread inside the ball rather than hugging the origin */
    float *T=malloc(sizeof(float)*(size_t)N*d);
    float *U=malloc(sizeof(float)*(size_t)N*d);
    for(uint32_t i=0;i<N;++i){
        const float *x=X+(size_t)i*d;
        const double n=nrm(x,d); const double spread=0.4+2.6*(double)((i*2654435761u)%1000u)/1000.0;  /* tangent norm 0.4..3.0 */
        const double scale=(n>1e-9? spread/n : 0.0);
        for(uint32_t j=0;j<d;++j) T[(size_t)i*d+j]=(float)(x[j]*scale);
        expmap0(T+(size_t)i*d,U+(size_t)i*d,(int)d);
    }
    double rmin=1e9,rmax=0,rmean=0;
    for(uint32_t i=0;i<N;++i){ const double r=nrm(U+(size_t)i*d,(int)d); if(r<rmin)rmin=r; if(r>rmax)rmax=r; rmean+=r; }
    printf("N=%u d=%u  ball radius: min %.4f mean %.4f max %.4f\n",N,d,rmin,rmean/N,rmax);

    const int KK=10, RR=100, NQ=200;
    printf("\n%-14s %-4s %-11s %-13s %-11s\n","pipeline","bits","recall@10","+rerank@100","bytes/item");

    for(int bits=2;bits<=4;++bits){
        wubu_tqvec *c=wubu_tqvec_init((int)d,bits,0xC0FFEEull);
        size_t nb=wubu_tqvec_code_bytes(c);
        unsigned char *code=malloc(nb*N);
        float *hat=malloc(sizeof(float)*(size_t)d);
        double *sc=malloc(sizeof(double)*N); int *idx=malloc(sizeof(int)*N);

        /* --- A: quantise ball coordinates --- */
        for(uint32_t i=0;i<N;++i) wubu_tqvec_encode(c,U+(size_t)i*d,code+nb*i);
        long hit=0,hitrr=0;
        for(int q=0;q<NQ;++q){
            const uint32_t qi=(uint32_t)((uint64_t)q*N/NQ);
            const float *qv=U+(size_t)qi*d;
            for(uint32_t i=0;i<N;++i) sc[i]=dpoinc(qv,U+(size_t)i*d,(int)d);
            sc[qi]=1e30;
            for(int a=0;a<KK;++a){int b=a;for(uint32_t t=a+1;t<N;++t)if(sc[t]<sc[b])b=(int)t;double ts=sc[a];sc[a]=sc[b];sc[b]=ts;int ti=idx[a];idx[a]=idx[b];idx[b]=ti;}
            int truth[16]; for(int a=0;a<KK;++a) truth[a]=idx[a];
            for(uint32_t i=0;i<N;++i){ wubu_tqvec_decode(c,code+nb*i,hat); sc[i]=dpoinc(qv,hat,(int)d); idx[i]=(int)i; }
            sc[qi]=1e30;
            for(int a=0;a<RR;++a){int b=a;for(uint32_t t=a+1;t<N;++t)if(sc[t]<sc[b])b=(int)t;double ts=sc[a];sc[a]=sc[b];sc[b]=ts;int ti=idx[a];idx[a]=idx[b];idx[b]=ti;}
            for(int a=0;a<KK;++a) for(int t=0;t<KK;++t) if(idx[a]==truth[t]){++hit;break;}
            int rr[150]; for(int a=0;a<RR;++a) rr[a]=idx[a];
            double rs[150]; for(int a=0;a<RR;++a) rs[a]=dpoinc(qv,U+(size_t)rr[a]*d,(int)d);
            for(int a=0;a<KK;++a){int b=a;for(int t=a+1;t<RR;++t)if(rs[t]<rs[b])b=t;double ts=rs[a];rs[a]=rs[b];rs[b]=ts;int ti=rr[a];rr[a]=rr[b];rr[b]=ti;}
            for(int a=0;a<KK;++a) for(int t=0;t<KK;++t) if(rr[a]==truth[t]){++hitrr;break;}
        }
        printf("%-14s %-4d %-11.3f %-13.3f %-11zu\n","ball",bits,(double)hit/(NQ*KK),(double)hitrr/(NQ*KK),nb);

        /* --- B: quantise tangent coordinates, exp-map the reconstruction back --- */
        for(uint32_t i=0;i<N;++i) wubu_tqvec_encode(c,T+(size_t)i*d,code+nb*i);
        hit=0; hitrr=0;
        for(int q=0;q<NQ;++q){
            const uint32_t qi=(uint32_t)((uint64_t)q*N/NQ);
            const float *qv=U+(size_t)qi*d;
            for(uint32_t i=0;i<N;++i) sc[i]=dpoinc(qv,U+(size_t)i*d,(int)d);
            sc[qi]=1e30;
            for(int a=0;a<KK;++a){int b=a;for(uint32_t t=a+1;t<N;++t)if(sc[t]<sc[b])b=(int)t;double ts=sc[a];sc[a]=sc[b];sc[b]=ts;int ti=idx[a];idx[a]=idx[b];idx[b]=ti;}
            int truth[16]; for(int a=0;a<KK;++a) truth[a]=idx[a];
            for(uint32_t i=0;i<N;++i){ wubu_tqvec_decode(c,code+nb*i,hat); expmap0(hat,hat,(int)d); sc[i]=dpoinc(qv,hat,(int)d); idx[i]=(int)i; }
            sc[qi]=1e30;
            for(int a=0;a<RR;++a){int b=a;for(uint32_t t=a+1;t<N;++t)if(sc[t]<sc[b])b=(int)t;double ts=sc[a];sc[a]=sc[b];sc[b]=ts;int ti=idx[a];idx[a]=idx[b];idx[b]=ti;}
            for(int a=0;a<KK;++a) for(int t=0;t<KK;++t) if(idx[a]==truth[t]){++hit;break;}
            int rr[150]; for(int a=0;a<RR;++a) rr[a]=idx[a];
            double rs[150]; for(int a=0;a<RR;++a) rs[a]=dpoinc(qv,U+(size_t)rr[a]*d,(int)d);
            for(int a=0;a<KK;++a){int b=a;for(int t=a+1;t<RR;++t)if(rs[t]<rs[b])b=t;double ts=rs[a];rs[a]=rs[b];rs[b]=ts;int ti=rr[a];rr[a]=rr[b];rr[b]=ti;}
            for(int a=0;a<KK;++a) for(int t=0;t<KK;++t) if(rr[a]==truth[t]){++hitrr;break;}
        }
        printf("%-14s %-4d %-11.3f %-13.3f %-11zu\n","tangent",bits,(double)hit/(NQ*KK),(double)hitrr/(NQ*KK),nb);
        free(code); free(hat); free(sc); free(idx); wubu_tqvec_free(c);
    }
    free(X); free(T); free(U);
    return 0;
}
