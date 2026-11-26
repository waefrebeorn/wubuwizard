/* turboquant_recall.c - does the quantiser preserve NEIGHBOUR ORDER, not just error?
 *
 * The probe established the distortion bounds. Bounds on ||x - x_hat||^2 do not by themselves say that the top-10 by
 * inner product survives, and that is the only question retrieval cares about. This answers it on data with the
 * geometry real embeddings have: clustered and anisotropic, not uniform on the sphere.
 *
 * Corpus: N items drawn from C cluster centres in a k-dimensional subspace (k << d) with small noise, renormalised to
 * the unit sphere. That mimics the low intrinsic dimension of real embeddings, which is what makes quantisation
 * survivable. Queries are held-out points from the same generator.
 *
 * Reports recall@10 for exact scoring vs TurboQuant at b = 2/3/4, and for a re-rank stage that takes the quantised
 * top-100 and re-scores those 100 exactly.
 *
 * Build: gcc -O2 -std=gnu11 -static -o tqr.exe tools/turboquant_recall.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define GRID 32768

static uint64_t g_s = 12345;
static double urand(void){ g_s^=g_s>>12; g_s^=g_s<<25; g_s^=g_s>>27; return (double)((g_s*2685821657736338717ULL)>>11)/9007199254740992.0; }
static double nrand(void){ double u=urand(),v=urand(); if(u<1e-12)u=1e-12; return sqrt(-2.0*log(u))*cos(6.283185307179586*v); }

static double coord_density(double t,int d){ if(t<=-1.0||t>=1.0) return 0.0; return pow(1.0-t*t,(double)(d-3)/2.0); }

static void lloyd_max(int d,int bits,double *cent)
{
    const int n=1<<bits;
    double *t=malloc(sizeof(double)*GRID), *w=malloc(sizeof(double)*GRID);
    double tot=0;
    for(int i=0;i<GRID;++i){ t[i]=-1.0+2.0*(i+0.5)/GRID; w[i]=coord_density(t[i],d); tot+=w[i]; }
    for(int i=0;i<GRID;++i) w[i]/=tot;
    double acc=0; int j=0;
    for(int i=0;i<GRID&&j<n;++i){ acc+=w[i]; if(acc>=(double)(j+1)/n-1e-15){ cent[j++]=t[i]; } }
    for(;j<n;++j) cent[j]=t[GRID-1];
    for(int it=0;it<200;++it){
        double bnd[65], sum[65], ws[65];
        for(int k=1;k<n;++k) bnd[k]=0.5*(cent[k-1]+cent[k]);
        bnd[0]=-2; bnd[n]=2;
        memset(sum,0,sizeof sum); memset(ws,0,sizeof ws);
        int seg=0;
        for(int i=0;i<GRID;++i){ while(seg<n-1 && t[i]>bnd[seg+1]) ++seg; sum[seg]+=t[i]*w[i]; ws[seg]+=w[i]; }
        double dl=0;
        for(int k=0;k<n;++k){ if(ws[k]<=1e-18) continue; double c=sum[k]/ws[k]; dl+=fabs(c-cent[k]); cent[k]=c; }
        if(dl<1e-12) break;
    }
    free(t); free(w);
}
static int nearest(const double*c,int n,double v){ int b=0; double bd=fabs(v-c[0]); for(int k=1;k<n;++k){ double dd=fabs(v-c[k]); if(dd<bd){bd=dd;b=k;} } return b; }

int main(int argc,char**argv)
{
    const int d      = argc>1?atoi(argv[1]):32;
    const int N      = argc>2?atoi(argv[2]):20000;
    const int C      = argc>3?atoi(argv[3]):200;    /* clusters */
    const int K      = argc>4?atoi(argv[4]):8;      /* intrinsic subspace dim */
    const int NQ     = 200;
    const int KNN    = 10, RERANK = 100;

    /* random orthogonal rotation, seeded - shared by index and query */
    double *R=malloc(sizeof(double)*d*d);
    for(int i=0;i<d*d;++i) R[i]=nrand();
    for(int i=0;i<d;++i){
        for(int k=0;k<i;++k){ double dot=0; for(int j=0;j<d;++j) dot+=R[i*d+j]*R[k*d+j];
            for(int j=0;j<d;++j) R[i*d+j]-=dot*R[k*d+j]; }
        double nr=0; for(int j=0;j<d;++j) nr+=R[i*d+j]*R[i*d+j];
        nr=sqrt(nr); for(int j=0;j<d;++j) R[i*d+j]/=nr;
    }

    /* generator: cluster centres live in the first K ambient axes; items get small noise and are normalised */
    double *ctr=malloc(sizeof(double)*C*d);
    for(int c=0;c<C;++c){
        double nr=0;
        for(int j=0;j<d;++j){ ctr[c*d+j]= (j<K)? nrand() : 0.0; nr+=ctr[c*d+j]*ctr[c*d+j]; }
        nr=sqrt(nr); for(int j=0;j<d;++j) ctr[c*d+j]/=nr;
    }
    double *X=malloc(sizeof(double)*N*d);
    for(int i=0;i<N;++i){
        int cc=(int)(urand()*C); if(cc>=C)cc=C-1;
        double nr=0;
        for(int j=0;j<d;++j){ double v=ctr[cc*d+j]+0.35*nrand()/sqrt((double)d); X[i*d+j]=v; nr+=v*v; }
        nr=sqrt(nr); for(int j=0;j<d;++j) X[i*d+j]/=nr;
    }

    /* quantise the whole corpus in the rotated frame at each bit width */
    printf("d=%d N=%d clusters=%d intrinsic_dim=%d, recall@%d, re-rank top-%d\n\n", d, N, C, K, KNN, RERANK);
    printf("%-5s %-10s %-12s %-12s %-12s\n","bits","recall@10","+rerank@100","bytes/item","ratio-vs-f32");

    for(int bits=1;bits<=4;++bits){
        const int n=1<<bits;
        double *cent=malloc(sizeof(double)*n);
        lloyd_max(d,bits,cent);

        signed char *code=malloc((size_t)N*d);        /* packed b bits stored as one byte for the probe */
        double *z=malloc(sizeof(double)*d);
        for(int i=0;i<N;++i){
            for(int r=0;r<d;++r){ double s=0; for(int j=0;j<d;++j) s+=R[r*d+j]*X[i*d+j]; z[r]=s; }
            for(int r=0;r<d;++r) code[(size_t)i*d+r]=(signed char)nearest(cent,n,z[r]);
        }

        long hit=0, hitrr=0;
        int *idx=malloc(sizeof(int)*N);
        double *sc=malloc(sizeof(double)*N);
        for(int q=0;q<NQ;++q){
            /* query: a held-out point from the same cluster family */
            int cc=(int)(urand()*C); if(cc>=C)cc=C-1;
            double *qv=calloc((size_t)d,sizeof(double)), nr=0;
            for(int j=0;j<d;++j){ double v=ctr[cc*d+j]+0.35*nrand()/sqrt((double)d); qv[j]=v; nr+=v*v; }
            nr=sqrt(nr); for(int j=0;j<d;++j) qv[j]/=nr;

            /* exact scores for ground truth */
            for(int i=0;i<N;++i){ double s=0; for(int j=0;j<d;++j) s+=qv[j]*X[i*d+j]; sc[i]=s; idx[i]=i; }
            for(int a=0;a<KNN;++a){ int b=a; for(int t=a+1;t<N;++t) if(sc[t]>sc[b]) b=t;
                double ts=sc[a]; sc[a]=sc[b]; sc[b]=ts; int ti=idx[a]; idx[a]=idx[b]; idx[b]=ti; }
            int truth[32]; for(int a=0;a<KNN;++a) truth[a]=idx[a];

            /* quantised score: <q, x_hat> where x_hat = R^T (code-vector) */
            double *qz=malloc(sizeof(double)*d);
            for(int r=0;r<d;++r){ double s=0; for(int j=0;j<d;++j) s+=R[r*d+j]*qv[j]; qz[r]=s; }
            /* dequantise once per item into the score: sum_r qz[r]*cent[code] (no full back-projection needed) */
            for(int i=0;i<N;++i){ double s=0; for(int r=0;r<d;++r) s+=qz[r]*cent[(int)code[(size_t)i*d+r]]; sc[i]=s; idx[i]=i; }
            for(int a=0;a<RERANK;++a){ int b=a; for(int t=a+1;t<N;++t) if(sc[t]>sc[b]) b=t;
                double ts=sc[a]; sc[a]=sc[b]; sc[b]=ts; int ti=idx[a]; idx[a]=idx[b]; idx[b]=ti; }

            for(int a=0;a<KNN;++a) for(int t=0;t<KNN;++t) if(idx[a]==truth[t]){ ++hit; break; }

            /* re-rank the quantised top-100 exactly, then take the best 10 */
            int rr[128]; for(int a=0;a<RERANK;++a) rr[a]=idx[a];
            double rs[128]; for(int a=0;a<RERANK;++a){ double s=0; const int i=rr[a]; for(int j=0;j<d;++j) s+=qv[j]*X[i*d+j]; rs[a]=s; }
            for(int a=0;a<KNN;++a){ int b=a; for(int t=a+1;t<RERANK;++t) if(rs[t]>rs[b]) b=t;
                double ts=rs[a]; rs[a]=rs[b]; rs[b]=ts; int ti=rr[a]; rr[a]=rr[b]; rr[b]=ti; }
            for(int a=0;a<KNN;++a) for(int t=0;t<KNN;++t) if(rr[a]==truth[t]){ ++hitrr; break; }

            free(qv); free(qz);
        }
        const double by=(double)d*bits/8.0+2.0;
        printf("%-5d %-10.3f %-12.3f %-12.1f %-12.1f\n", bits,
               (double)hit/((double)NQ*KNN), (double)hitrr/((double)NQ*KNN),
               by, (double)d*4.0/by);
        free(cent); free(code); free(z); free(idx); free(sc);
    }
    free(R); free(ctr); free(X);
    return 0;
}
