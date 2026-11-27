/* test_dspark_col.c -- full 7-step C markov recurrence -> col vs upstream
 * dspark_col-<i>-0. Reads bins from dump_pj. Compile -O2, run <dir>. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dspark_markov.c"
static float *readbin(const char *p, size_t *n){ FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"no %s\n",p);exit(2);} fseek(f,0,SEEK_END); long sz=ftell(f); rewind(f); *n=(size_t)(sz/4); float*a=malloc((size_t)sz); if(fread(a,1,(size_t)sz,f)!=(size_t)sz){fprintf(stderr,"read %s\n",p);exit(2);} fclose(f); return a; }
int main(int argc,char**argv){
    const char *dir=argc>1?argv[1]:"."; char p[640]; size_t n;
    size_t R=256,E=2304,V=200058;
    snprintf(p,sizeof p,"%s/h.bin",dir); float*h=readbin(p,&n);            /* 8E */
    snprintf(p,sizeof p,"%s/fr.bin",dir); float*fr=readbin(p,&n);
    snprintf(p,sizeof p,"%s/w1chain.bin",dir); float*w1c=readbin(p,&n);    /* 7R */
    snprintf(p,sizeof p,"%s/W2.bin",dir); float*W2=readbin(p,&n);
    snprintf(p,sizeof p,"%s/b2.bin",dir); float*b2=readbin(p,&n);
    snprintf(p,sizeof p,"%s/base.bin",dir); float*base=readbin(p,&n);     /* 8V */
    snprintf(p,sizeof p,"%s/Jw.bin",dir); float*Jw=readbin(p,&n);
    snprintf(p,sizeof p,"%s/b.bin",dir); float*b=readbin(p,&n);
    float *state=calloc(R,4), *x=malloc((2*R+2*E)*4), *out=malloc(3*R*4);
    static int chain[7]={0,149904,150261,149882,149895,149878,220};
    for(int i=1;i<=7;i++){
        memcpy(x,state,R*4); memcpy(x+R,w1c+(i-1)*R,R*4);
        memcpy(x+2*R,h+(i)*E,E*4); memcpy(x+2*R+E,fr,E*4);
        dspark_joint_proj(Jw,b,x,R,E,out);
        float markov[256]; for(size_t r=0;r<R;r++){ float g=1/(1+expf(-out[r])), c=tanhf(out[R+r]); state[r]=c+g*(state[r]-c); markov[r]=tanhf(out[2*R+r]); }
        float *col=malloc(V*4); dspark_col_output(base+i*V, markov, W2, b2, V, R, 0.0f, col);
        snprintf(p,sizeof p,"%s/dspark_col-%d-0.bin",dir,i); FILE*f=fopen(p,"rb");
        double sxy=0,sxx=0,syy=0,d=0; float*ref;
        if(f){ fseek(f,0,SEEK_END); long sz=ftell(f); rewind(f); ref=malloc((size_t)sz); fread(ref,1,(size_t)sz,f); fclose(f);
          double mx=0,my=0; for(size_t v=0;v<V;v++){mx+=col[v];my+=ref[v];} mx/=V;my/=V;
          for(size_t v=0;v<V;v++){double o=col[v]-mx,yy=ref[v]-my;sxy+=o*yy;sxx+=o*o;syy+=yy*yy; d+=fabs(col[v]-ref[v]);}
          printf("step%d corr=%.6f mean|d|=%.4f rms=%.3f/%.3f\n",i,sxy/sqrt(sxx*syy),d/V,sqrt(sxx/V),sqrt(syy/V)); free(col); free(ref); }
        else { printf("step%d: no dspark_col-%d-0 (%p)\n",i,i,(void*)f); free(col); }
    }
    return 0;
}
