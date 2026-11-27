/* test_dspark_propose.c -- exercise the unified C proposal path end-to-end on
 * real drafter data: dspark_markov_step (state carry + markov chunk) ->
 * dspark_col_output (w2 bias + softcap) -> verify col vs upstream, and
 * dspark_draw (greedy argmax) -> print proposed chain.
 * Uses bins dumped from dump_pj (h, fr, w1chain, W2, b2, base, Jw, b). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dspark_markov.c"
static float *readbin(const char *p, size_t *n){ FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"no %s\n",p);exit(2);} fseek(f,0,SEEK_END); long sz=ftell(f); rewind(f); *n=(size_t)(sz/4); float*a=malloc((size_t)sz); if(fread(a,1,(size_t)sz,f)!=(size_t)sz){fprintf(stderr,"read %s\n",p);exit(2);} fclose(f); return a; }
int main(int argc,char**argv){
    const char *dir=argc>1?argv[1]:"."; char p[640]; size_t n;
    size_t R=256,E=2304,V=200058;
    snprintf(p,sizeof p,"%s/h.bin",dir); float*h=readbin(p,&n);
    snprintf(p,sizeof p,"%s/fr.bin",dir); float*fr=readbin(p,&n);
    snprintf(p,sizeof p,"%s/w1chain.bin",dir); float*w1c=readbin(p,&n);
    snprintf(p,sizeof p,"%s/W2.bin",dir); float*W2=readbin(p,&n);
    snprintf(p,sizeof p,"%s/b2.bin",dir); float*b2=readbin(p,&n);
    snprintf(p,sizeof p,"%s/base.bin",dir); float*base=readbin(p,&n);
    snprintf(p,sizeof p,"%s/Jw.bin",dir); float*Jw=readbin(p,&n);
    snprintf(p,sizeof p,"%s/b.bin",dir); float*b=readbin(p,&n);
    static int proposed[7]; float *state=calloc(R,4), markov[256];
    double worst=0; int worst_col=-1;
    for(int i=1;i<=7;i++){
        dspark_markov_step(Jw,b,w1c+(i-1)*R,h+(size_t)i*E,fr,R,E,state,markov);
        float *col=malloc(V*4); dspark_col_output(base+(size_t)i*V,markov,W2,b2,V,R,0.0f,col);
        int ids[64]; float pr[64];
        proposed[i-1]=dspark_draw(col,V,64,0.0f,0.0f,1,ids,pr); /* greedy */
        snprintf(p,sizeof p,"%s/dspark_col-%d-0.bin",dir,i); FILE*f=fopen(p,"rb");
        if(f){ fseek(f,0,SEEK_END); long sz=ftell(f); rewind(f); float*ref=malloc((size_t)sz); fread(ref,1,(size_t)sz,f); fclose(f);
          double d=0; for(size_t v=0;v<V;v++) d+=fabs(col[v]-ref[v]); d/=V; if(d>worst){worst=d;worst_col=i;}
          printf("step%d col mean|d|=%.4f proposed(greedy)=%d\n",i,d,proposed[i-1]); free(ref); }
        free(col);
    }
    printf("RESULT worst col mean|d|=%.4f at step%d; proposed chain: %d %d %d %d %d %d %d\n",
           worst,worst_col,proposed[0],proposed[1],proposed[2],proposed[3],proposed[4],proposed[5],proposed[6]);
    return worst<1.0f?0:1;
}
