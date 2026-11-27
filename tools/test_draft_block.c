/* test_draft_block.c -- faithful C port of the Inkling-2 DSpark drafter 3-layer
 * dense H3-KV block forward (dflash_ref.py block()+ctx_kv()+forward()). Loads
 * the drafter GGUF via the in-engine gguf_reader, builds x0=[anchor;7 mask],
 * runs 3 layers over T ctx taps, writes h (8,2304) and base (8,200058) f32 for
 * comparison against the verified numpy reference (target corr ~0.9999).
 *
 * Precision mirrors dflash_ref: projections/residual in f32, rms reduced in f64,
 * attention einsum/softmax in f64, q/k/v re-cast f32. blk_pos=arange(8)+1,
 * ctx_pos=arange(T). rope is split-half (theta 1e6).
 * usage: test_draft_block <gguf> <anchor_emb.f32> <t1.f32,..,t4.f32> <h_out> <base_out> <rms_log>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gguf_reader.h"

#define E 2304
#define NL 3
#define NH 24
#define NKV 8
#define HD 128
#define FF 9216
#define NMSK 7
#define B 8
#define G 3
#define THETA 1e6
#define EPS 1e-6
#define VV 200058

static float *load_flat(gguf_ctx *ctx, const char *name) {
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "missing %s\n", name); exit(2); }
    int64_t n = 1; for (int d = 0; d < t->n_dims; ++d) n *= t->dims[d];
    float *b = (float*)malloc(sizeof(float)*(size_t)n);
    int rc = gguf_read_tensor_f32(ctx, t, b, (size_t)n);
    if (rc != (int)n) { fprintf(stderr, "read %s rc=%d\n", name, rc); exit(2); }
    return b;
}
/* W is a ggml flat [in,out] tensor -> row-major (out,in). y[o] = sum_k x[k]*W[o*in+k] */
static void gemv(const float *W, int inN, int outN, const float *x, float *y) {
    for (int o = 0; o < outN; ++o) { float a = 0; const float *wr = W + (size_t)o*inN;
        for (int k = 0; k < inN; ++k) a += x[k]*wr[k]; y[o] = a; }
}
static void rms_f64row(const float *x, int len, const float *w, float *out) {
    double s = 0; for (int i = 0; i < len; ++i) s += (double)x[i]*x[i]; s = s/len + EPS; double is = 1.0/sqrt(s);
    if (w) for (int i = 0; i < len; ++i) out[i] = (float)((double)w[i]*x[i]*is);
    else   for (int i = 0; i < len; ++i) out[i] = (float)((double)x[i]*is);
}
/* split-half rope, pos scalar, x is H x D f32 */
static void rope_split_hd(const float *x, int H, int D, int pos, float *o) {
    int half = D/2; double inv[64], c[64], s[64];
    for (int i = 0; i < half; ++i) inv[i] = pow(THETA, -2.0*i/D);
    for (int i = 0; i < half; ++i) { double a = pos*inv[i]; c[i]=cos(a); s[i]=sin(a); }
    for (int h = 0; h < H; ++h) {
        const float *xr = x + h*D; float *or_ = o + h*D;
        for (int i = 0; i < half; ++i) {
            or_[i]       = (float)((double)xr[i]*c[i] - (double)xr[i+half]*s[i]);
            or_[i+half]  = (float)((double)xr[i]*s[i] + (double)xr[i+half]*c[i]);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: test_draft_block <gguf> <emb> <taps> <hout> <bout> <rmslog> <fout_after_layer>\n"); return 2; }
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "OPEN FAIL\n"); return 1; }
    /* anchors + masks -> x0 */
    FILE *f = fopen(argv[2],"rb"); float emb[E]; if(!f||fread(emb,4,E,f)!=(size_t)E){fprintf(stderr,"emb\n");return 2;} fclose(f);
    float *maskraw = load_flat(ctx, "mask_embd.weight");          /* ggml [E,7] -> row-major (7,E): elem(m,e)=raw[m*E+e] */
    float x[B*E]; for(int e=0;e<E;++e){ x[0*E+e]=emb[e]; for(int m=0;m<NMSK;++m) x[(m+1)*E+e]=maskraw[(long)m*E+e]; }
    { FILE*pw=fopen("xin0.bin","wb"); fwrite(x,4,B*E,pw); fclose(pw); }
    /* taps: [4][T][E]; read each file, keep up to T rows at the END (ctx_pos=arange(T)) */
    int T = 1;   /* all tap files have the same T; test_drafter_all used T=1 (single ctx row per tap) */
    float tapbuf[4*4*E]; memset(tapbuf,0,sizeof tapbuf);
    {
        char buf[1024]; strcpy(buf,argv[3]); char *p=strtok(buf,","); int it=0;
        while (p && it<4) { FILE *g=fopen(p,"rb"); float tmp[64*E]; long rows=0; if(g){ long n=fread(tmp,4,sizeof(tmp)/4,g); fclose(g); rows=n/E; if(rows>4)rows=4; }
            if (it==0 && rows>0) T=(int)rows;
            for(long r=0;r<rows;++r) for(int e=0;e<E;++e) tapbuf[((long)it*T + r)*E+e]=tmp[r*E+e];
            p=strtok(NULL,","); ++it; }
    }
    /* weights */
    float *outnorm = load_flat(ctx,"output_norm.weight");
    float *wq[NL],*wk[NL],*wv[NL],*wo[NL],*wkct[NL],*wvct[NL],*wg[NL],*wu[NL],*wd[NL];
    float *an[NL],*qn[NL],*kn[NL],*kcn[NL],*fn[NL],*cmx[NL],*cn[NL];
    for (int L=0;L<NL;++L){ char n[256];
        snprintf(n,sizeof n,"blk.%d.attn_norm.weight",L); an[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.attn_q.weight",L); wq[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.attn_k.weight",L); wk[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.attn_v.weight",L); wv[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.attn_output.weight",L); wo[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.attn_q_norm.weight",L); qn[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.attn_k_norm.weight",L); kn[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.attn_k_ctx.weight",L); wkct[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.attn_v_ctx.weight",L); wvct[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.attn_k_ctx_norm.weight",L); kcn[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.ctx_mix.weight",L); cmx[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.ctx_norm.weight",L); cn[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.ffn_norm.weight",L); fn[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.ffn_gate.weight",L); wg[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.ffn_up.weight",L); wu[L]=load_flat(ctx,n);
        snprintf(n,sizeof n,"blk.%d.ffn_down.weight",L); wd[L]=load_flat(ctx,n);
    }
    float *KCB = 0; (void)KCB;   /* drafter has no attn_k_ctx.bias */
    FILE *rmslog=strlen(argv[6])?fopen(argv[6],"w"):NULL;
    const int Total = T+B;
    float *Kbig = (float*)calloc((size_t)Total*NH*HD,4);
    float *Vbig = (float*)calloc((size_t)Total*NH*HD,4);
    float *mixed = (float*)malloc((size_t)T*E*4);
    float *Kctx = (float*)malloc((size_t)T*NKV*HD*4);
    float *Vctx = (float*)malloc((size_t)T*NKV*HD*4);
    for (int L=0;L<NL;++L){
        /* --- ctx_kv --- */
        float m0[E];
        /* mixed[c][e] = sum_t cmx[t]*tapbuf[t][c][e]; rms(,cn); -> kc/vc. Use T=4 rows. */
        for(int c=0;c<T;++c){
            double mm[E]; double s=0;
            for(int e=0;e<E;++e){ double a=0; for(int t=0;t<4;++t) a+=(double)cmx[L][t]*tapbuf[(t*T+c)*E+e]; mm[e]=a; s+=a*a; } s=s/E+EPS;
            float m[E]; for(int e=0;e<E;++e) m[e]=(float)((double)cn[L][e]*mm[e]/sqrt(s));
            if (L==0&&c==0){ FILE*pw=fopen("cM.bin","wb"); fwrite(m,4,E,pw); FILE*pw2=fopen("cMix.bin","wb"); fwrite(cmx[L],4,4,pw2); FILE*pw3=fopen("cCn.bin","wb"); fwrite(cn[L],4,E,pw3); fclose(pw); fclose(pw2); fclose(pw3); }
            float kc[1024],vc[1024]; gemv(wkct[L],E,1024,m,kc); gemv(wvct[L],E,1024,m,vc);
            for(int kv=0;kv<NKV;++kv){ float nb[HD]; rms_f64row(kc+kv*HD,HD,kcn[L],nb);
                for(int d=0;d<HD;++d) Kctx[(c*NKV+kv)*HD+d]=nb[d];
                for(int d=0;d<HD;++d) Vctx[(c*NKV+kv)*HD+d]=vc[kv*HD+d]; }
        }
        /* rope the ctx K rows (into temp then back) */
        for(int c=0;c<T;++c){ float kr[NKV*HD]; rope_split_hd(&Kctx[c*NKV*HD],NKV,HD,c,kr); memcpy(&Kctx[c*NKV*HD],kr,sizeof kr); }
        if (L==0){ FILE*pw=fopen("cKctx.bin","wb"); fwrite(Kctx,4,T*NKV*HD,pw); fclose(pw); FILE*pw2=fopen("cVctx.bin","wb"); fwrite(Vctx,4,T*NKV*HD,pw2); fclose(pw2); }
        /* --- attn --- */
        float H[B*E];
        for(int b=0;b<B;++b) rms_f64row(&x[b*E],E,an[L],&H[b*E]);
        float Q[B*NH*HD],KK[B*NKV*HD],V[B*NKV*HD];
        for(int b=0;b<B;++b){
            float q[3072],k[1024],v[1024]; gemv(wq[L],E,3072,&H[b*E],q); gemv(wk[L],E,1024,&H[b*E],k); gemv(wv[L],E,1024,&H[b*E],v);
            float qr[NH*HD],krr[NKV*HD];
            for(int h=0;h<NH;++h){ float nh[HD]; rms_f64row(&q[h*HD],HD,qn[L],nh); rope_split_hd(nh,1,HD,b+1,qr); for(int d=0;d<HD;++d) Q[(b*NH+h)*HD+d]=qr[d]; }
            for(int h=0;h<NKV;++h){ float nh[HD]; rms_f64row(&k[h*HD],HD,kn[L],nh); rope_split_hd(nh,1,HD,b+1,krr); for(int d=0;d<HD;++d) KK[(b*NKV+h)*HD+d]=krr[d]; }
            memcpy(&V[b*NKV*HD],v,sizeof v);
        }
        if (L==0){ FILE*pw=fopen("cQ.bin","wb"); fwrite(Q,4,B*NH*HD,pw); fclose(pw); FILE*pw2=fopen("cKK.bin","wb"); fwrite(KK,4,B*NKV*HD,pw2); fclose(pw2); }
        for(int c=0;c<T;++c) for(int h=0;h<NH;++h){ int kv=h/G; memcpy(&Kbig[(c*NH+h)*HD],&Kctx[(c*NKV+kv)*HD],HD*4); memcpy(&Vbig[(c*NH+h)*HD],&Vctx[(c*NKV+kv)*HD],HD*4); }
        for(int b=0;b<B;++b) for(int h=0;h<NH;++h){ int kv=h/G; memcpy(&Kbig[((T+b)*NH+h)*HD],&KK[(b*NKV+kv)*HD],HD*4); memcpy(&Vbig[((T+b)*NH+h)*HD],&V[(b*NKV+kv)*HD],HD*4); }
        double sc=1.0/sqrt((double)HD);
        float oarg[B*NH*HD];
        float *attlog=(float*)calloc((size_t)B*NH*Total,4);
        for(int b=0;b<B;++b) for(int h=0;h<NH;++h){
            double logit[16]; double mx=-1e300;
            for(int j=0;j<Total;++j){ double a=0; const float*qq=&Q[(b*NH+h)*HD],*kk=&Kbig[(j*NH+h)*HD];
                for(int d=0;d<HD;++d) a+=(double)qq[d]*(double)kk[d]; logit[j]=a*sc; if(logit[j]>mx)mx=logit[j];
                attlog[((size_t)b*NH+h)*Total+j]=(float)(a*sc); }
            double sum=0; for(int j=0;j<Total;++j){ logit[j]=exp(logit[j]-mx); sum+=logit[j]; }
            for(int d=0;d<HD;++d){ double a=0; for(int j=0;j<Total;++j) a+= (logit[j]/sum)*(double)Vbig[(j*NH+h)*HD+d]; oarg[(b*NH+h)*HD+d]=(float)a; }
        }
        if (L==0){ FILE*pw=fopen("cattlog.bin","wb"); fwrite(attlog,4,(size_t)B*NH*Total,pw); fclose(pw); }
        float o230[E];
        for(int b=0;b<B;++b){ float oa[3072]; for(int hd=0;hd<NH*HD;++hd) oa[hd]=oarg[b*NH*HD+hd]; gemv(wo[L],NH*HD,E,oa,o230);
            if (L==0){ char pn[128]; snprintf(pn,sizeof pn,"catt%d.bin",b); FILE*pw=fopen(pn,"wb"); fwrite(oa,4,NH*HD,pw); fclose(pw);
                       snprintf(pn,sizeof pn,"cattO%d.bin",b); FILE*pw2=fopen(pn,"wb"); fwrite(o230,4,E,pw2); fclose(pw2); }
            for(int e=0;e<E;++e) x[b*E+e]+=o230[e]; }
        /* --- ffn --- */
        float h2[B*E];
        for(int b=0;b<B;++b) rms_f64row(&x[b*E],E,fn[L],&h2[b*E]);
        for(int b=0;b<B;++b){
            float ga[FF],up[FF],act[FF],dn[E];
            gemv(wg[L],E,FF,&h2[b*E],ga); gemv(wu[L],E,FF,&h2[b*E],up);
            for(int i=0;i<FF;++i) act[i]=ga[i]/(1.0f+expf(-ga[i]))*up[i];
            gemv(wd[L],FF,E,act,dn);
            for(int e=0;e<E;++e) x[b*E+e]+=dn[e];
        }
        if (rmslog){ double s=0; for(int i=0;i<B*E;++i) s+=(double)x[i]*x[i]; fprintf(rmslog,"layer%d rms %.4f ; first x %.4f %.4f\n",L,sqrt(s/(B*E)),x[0],x[1]);
            if (strcmp(argv[7],"-")==0L) { char pn[640]; snprintf(pn,sizeof pn,"xL%d.bin",L); FILE*pw=fopen(pn,"wb"); fwrite(x,4,B*E,pw); fclose(pw); } }
        if (strcmp(argv[7],"-")==0) { } 
    }
    /* h = rms(x, output_norm); base = h @ unembed.T (200058) */
    float h[B*E]; for(int b=0;b<B;++b) rms_f64row(&x[b*E],E,outnorm,&h[b*E]);
    /* unembed: read from a companion .f32 file (200058 rows x 2304) - engine dump; the test harness checks h only */
    FILE *ho=fopen(argv[4],"wb"); fwrite(h,4,B*E,ho); fclose(ho);
    /* base via unembed matrix file (argv[5] base output) - we need unembed; write h only and let harness diff */
    if (rmslog){ double s=0; for(int i=0;i<E;++i)s+=(double)h[i]*h[i]; fprintf(rmslog,"h_slot0 rms %.4f f_rnn slot n/a\n",sqrt(s/E)); fclose(rmslog); }
    gguf_close(ctx);
    printf("DONE wrote %d-byte h to %s\n",(int)(B*E*4),argv[4]);
    return 0;
}
