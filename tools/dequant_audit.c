/* dequant_audit.c - chase 1:1: prove my f32 dequant is BIT-EXACT against the engine's own decode.
 * For a tensor, run engine vec-dot vs my-dequant-f32 dot over MANY random q8_K activations; report the
 * WORST relative diff. If my dequant matches the engine exactly, every diff is pure fp32 accumulation-order
 * (~1e-7). Any subtle transcription bug shows up as a larger diff that some activation exposes.
 * usage: dequant_audit <model.gguf> <tensor> <iters> */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static const int8_t kvalues_iq4nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
#define f16h gguf_f16_to_f32
#include "iq3_xxs_tables.inc"
/* reference dequants (identical to the lane's dequant_all.c) */
static void deq(gguf_ctx*c, gguf_tensor_info*t, float*f){ /* f preallocated to ne */
    long long ne=1; for(int d=0;d<t->n_dims;d++) ne*=t->dims[d];
    const unsigned char*raw=(const unsigned char*)c->data_blob+(size_t)t->data_offset;
    if(t->ggml_type==23){
        const long long nb=(ne+255)/256;
        for(long long b=0;b<nb;b++){ const unsigned char*blk=raw+(size_t)b*136;
            float d=f16h((unsigned short)(blk[0]|(blk[1]<<8))); unsigned short sh=blk[2]|((unsigned short)blk[3]<<8);
            const unsigned char*sl=blk+4; const unsigned char*qs=blk+8; long long base=b*256;
            for(int s=0;s<8;s++){ int pair=s/2,half=s&1; unsigned hhex=(unsigned)(sh>>(4*pair))&0xff; unsigned char lb=sl[pair];
                int ls=half? (int)(((lb>>4)|((hhex<<2)&0x30))) : (int)(((lb&0xf)|((hhex<<4)&0x30)));
                const unsigned char*qb=qs+s*16; long long sb=base+s*32;
                for(int r=0;r<32;r++){ unsigned char qv=qb[r&15]; int nib=(r<16)?(qv&0xf):(qv>>4);
                    f[sb+r]=d*(float)(ls-32)*(float)kvalues_iq4nl[nib]; } } }
    } else if(t->ggml_type==8){ const long long nb=(ne+31)/32;
        for(long long b=0;b<nb;b++){ const unsigned char*blk=raw+(size_t)b*34; long long base=b*32;
            float d=f16h((unsigned short)(blk[0]|(blk[1]<<8))); for(int r=0;r<32;r++) f[base+r]=d*(float)(int8_t)blk[2+r]; }
    } else if(t->ggml_type==0){ const float*src=(const float*)raw; for(long long i=0;i<ne;i++) f[i]=src[i]; }
    else if(t->ggml_type==12){ extern void dequantize_q4_K_row(const uint8_t*,float*,int64_t); dequantize_q4_K_row(raw,f,ne); }
    else if(t->ggml_type==14){ extern void dequantize_q6_K_row(const uint8_t*,float*,int64_t); dequantize_q6_K_row(raw,f,ne); }
    else if(t->ggml_type==13){
        const long long nb=(ne+255)/256;
        for(long long b=0;b<nb;b++){ const unsigned char*blk=raw+(size_t)b*176;
            const unsigned char*sc=blk+4; const unsigned char*qh=blk+16; const unsigned char*ql=blk+48;
            float d=f16h((unsigned short)(blk[0]|(blk[1]<<8))), mn=f16h((unsigned short)(blk[2]|(blk[3]<<8)));
            int is=0; unsigned u1=1,u2=2; long long base=b*256;
            for(int j=0;j<256;j+=64){
                unsigned char s0,m0; if(is<4){s0=sc[is]&63;m0=sc[is+4]&63;}else{s0=(sc[is+4]&0xF)|((sc[is-4]>>6)<<4);m0=(sc[is+4]>>4)|((sc[is]>>6)<<4);}
                unsigned char s1,m1; int is1=is+1; if(is1<4){s1=sc[is1]&63;m1=sc[is1+4]&63;}else{s1=(sc[is1+4]&0xF)|((sc[is1-4]>>6)<<4);m1=(sc[is1+4]>>4)|((sc[is1]>>6)<<4);}
                float d1=d*(float)s0,m1f=mn*(float)m0,d2=d*(float)s1,m2f=mn*(float)m1;
                for(int l=0;l<32;++l) f[base+j+l]=d1*(float)((ql[l]&0xF)+((qh[l]&u1)?16:0))-m1f;
                for(int l=0;l<32;++l) f[base+j+32+l]=d2*(float)((ql[l]>>4)+((qh[l]&u2)?16:0))-m2f;
                ql+=32; is+=2; u1<<=2; u2<<=2; }
        }
    } else if(t->ggml_type==18){
        const long long nb=(ne+255)/256;
        for(long long b=0;b<nb;b++){ const unsigned char*blk=raw+(size_t)b*98;
            float d=f16h((unsigned short)(blk[0]|(blk[1]<<8)));
            const unsigned char*qsr=blk+2; const unsigned char*ss=qsr+64; long long base=b*256;
            for(int ib32=0;ib32<8;++ib32){
                unsigned aux32=ss[4*ib32]|(ss[4*ib32+1]<<8)|(ss[4*ib32+2]<<16)|((unsigned)ss[4*ib32+3]<<24);
                float db=d*(0.5f+(float)(aux32>>28))*0.5f;
                for(int l=0;l<4;++l){ int signs=w_ksigns_iq2xs[(aux32>>(7*l))&127];
                    const uint8_t*g1=(const uint8_t*)(&w_iq3xxs_grid[qsr[2*l]]);
                    const uint8_t*g2=(const uint8_t*)(&w_iq3xxs_grid[qsr[2*l+1]]);
                    for(int j=0;j<4;++j){
                        f[base+ib32*32+l*8+j]=db*(float)g1[j]*(signs&w_kmask_iq2xs[j]?-1.0f:1.0f);
                        f[base+ib32*32+l*8+j+4]=db*(float)g2[j]*(signs&w_kmask_iq2xs[j+4]?-1.0f:1.0f);} }
                qsr+=8; }
        }
    } else if(t->ggml_type==30){ const unsigned short*bsrc=(const unsigned short*)raw;
        for(long long i=0;i<ne;i++){ unsigned u=((unsigned)bsrc[i])<<16; float v; memcpy(&v,&u,4); f[i]=v; } }
}
int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: dequant_audit <model.gguf> <tensor> [iters]\n");return 2;}
    int iters = argc>3 ? atoi(argv[3]) : 50;
    gguf_ctx*c=gguf_open(argv[1]); if(!c)return 3; gguf_buffer_data(c);
    gguf_tensor_info*t=NULL; for(int i=0;i<c->n_tensors;i++) if(!strcmp(c->tensors[i].name,argv[2])){t=&c->tensors[i];break;}
    if(!t){fprintf(stderr,"not found\n");return 4;}
    long long ne=1; for(int d=0;d<t->n_dims;d++) ne*=t->dims[d];
    const unsigned char*raw=(const unsigned char*)c->data_blob+(size_t)t->data_offset;
    float*f=malloc((size_t)ne*sizeof(float)); deq(c,t,f);
    const int K=2304; if(K%256){fprintf(stderr,"K%%256\n");return 5;}
    float*engr=malloc(iters*sizeof(float)); float*mnr=malloc(iters*sizeof(float));
    /* engine dot fn by type */
    extern void ggml_vec_dot_q4_K_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
    extern void ggml_vec_dot_q5_K_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
    extern void ggml_vec_dot_q6_K_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
    extern void ggml_vec_dot_iq3_xxs_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
    extern int wubu_vec_dot_iq4_xs_q8_k(int,float*,const void*,const void*);
    int ty=t->ggml_type; double worst=0; int nok=0;
    for(int it=0;it<iters;it++){
        srand(1000+it); float x[2304]; for(int i=0;i<K;i++) x[i]=0.1f*(float)(rand()%2001-1000);
        block_q8_K q8[9]; quantize_row_q8_K(x,q8,K);
        float eng; float xr[2304]; for(int i=0;i<K;i++) xr[i]=q8[i/256].d*(float)q8[i/256].qs[i%256];
        float mine=0; for(int i=0;i<K;i++) mine+=f[i]*xr[i];
        if(ty==23) wubu_vec_dot_iq4_xs_q8_k(K,&eng,raw,&q8[0]);
        else if(ty==12) ggml_vec_dot_q4_K_q8_K_generic(K,&eng,0,raw,0,&q8[0],0,1);
        else if(ty==13) ggml_vec_dot_q5_K_q8_K_generic(K,&eng,0,raw,0,&q8[0],0,1);
        else if(ty==14) ggml_vec_dot_q6_K_q8_K_generic(K,&eng,0,raw,0,&q8[0],0,1);
        else if(ty==18) ggml_vec_dot_iq3_xxs_q8_K_generic(K,&eng,0,raw,0,&q8[0],0,1);
        else { fprintf(stderr,"no engine dot for ty%d\n",ty); return 6; }
        double rel = fabs((double)(eng-mine)/(eng));
        if(rel>worst)worst=rel; if(rel<1e-6)nok++;
    }
    fprintf(stderr,"%s ty=%d iters=%d: worst rel diff = %.3e (dequant bit-exact vs engine if ~1e-7 accum noise), %.0f%% of %d <= 1e-6\n",
        t->name,ty,iters,worst,100.0*nok/iters,iters);
    /* if worst is tiny, my dequant reproduces the engine's decode to exact-ness */
    return (worst<1e-5)?0:1;
}
