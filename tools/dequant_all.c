/* dequant_all.c - materialize the full single-f16 resident weight set: dequant every non-expert tensor
 * to f16 in ONE model open. Per-type dispatch is copied verbatim from the verified head_hf16.c. */
#include "gguf_reader.h"
#include "wubu_vecdot_iq4xs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const int8_t kvalues_iq4nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
#define f16h gguf_f16_to_f32
#include "iq3_xxs_tables.inc"
static int dequant_one(gguf_ctx*c, gguf_tensor_info*t, const char*outpath){
    long long ne=1; for(int d=0;d<t->n_dims;d++) ne*=t->dims[d];
    const unsigned char*raw=(const unsigned char*)c->data_blob+(size_t)t->data_offset;
    float*f=malloc((size_t)ne*sizeof(float));
    if(!f){fprintf(stderr,"oom\n");return 8;}
    if(t->ggml_type==23){
        const long long nb=(ne+255)/256;
        for(long long b=0;b<nb;b++){
            const unsigned char*blk=raw+(size_t)b*136;
            float d=f16h((unsigned short)(blk[0]|(blk[1]<<8)));
            unsigned short sh=blk[2]|((unsigned short)blk[3]<<8);
            const unsigned char*sl=blk+4; const unsigned char*qs=blk+8; long long base=b*256;
            for(int s=0;s<8;s++){
                int pair=s/2,half=s&1; unsigned hhex=(unsigned)(sh>>(4*pair))&0xffu; unsigned char lb=sl[pair];
                int ls=half? (int)(((lb>>4)|((hhex<<2)&0x30))) : (int)(((lb&0xf)|((hhex<<4)&0x30)));
                const unsigned char*qb=qs+s*16; long long sb=base+s*32;
                for(int r=0;r<32;r++){ unsigned char qv=qb[r&15]; int nib=(r<16)?(qv&0xf):(qv>>4);
                    f[sb+r]=d*(float)(ls-32)*(float)kvalues_iq4nl[nib]; }
            }
        }
    } else if(t->ggml_type==8){
        const long long nb=(ne+31)/32;
        for(long long b=0;b<nb;b++){ const unsigned char*blk=raw+(size_t)b*34; long long base=b*32;
            float d=f16h((unsigned short)(blk[0]|(blk[1]<<8)));
            for(int r=0;r<32;r++) f[base+r]=d*(float)(int8_t)blk[2+r]; }
    } else if(t->ggml_type==0){ const float*src=(const float*)raw; for(long long i=0;i<ne;i++) f[i]=src[i]; }
    else if(t->ggml_type==12){ extern void dequantize_q4_K_row(const uint8_t*,float*,int64_t); dequantize_q4_K_row(raw,f,ne); }
    else if(t->ggml_type==14){ extern void dequantize_q6_K_row(const uint8_t*,float*,int64_t); dequantize_q6_K_row(raw,f,ne); }
    else if(t->ggml_type==13){
        const long long nb=(ne+255)/256;
        for(long long b=0;b<nb;b++){
            const unsigned char*blk=raw+(size_t)b*176;
            const unsigned char*sc=blk+4; const unsigned char*qh=blk+16; const unsigned char*ql=blk+48;
            float d=f16h((unsigned short)(blk[0]|(blk[1]<<8))), mn=f16h((unsigned short)(blk[2]|(blk[3]<<8)));
            int is=0; unsigned u1=1,u2=2; long long base=b*256;
            for(int j=0;j<256;j+=64){
                unsigned char s0,m0; if(is<4){s0=sc[is]&63;m0=sc[is+4]&63;}else{s0=(sc[is+4]&0xF)|((sc[is-4]>>6)<<4);m0=(sc[is+4]>>4)|((sc[is]>>6)<<4);}
                unsigned char s1,m1; int is1=is+1; if(is1<4){s1=sc[is1]&63;m1=sc[is1+4]&63;}else{s1=(sc[is1+4]&0xF)|((sc[is1-4]>>6)<<4);m1=(sc[is1+4]>>4)|((sc[is1]>>6)<<4);}
                float d1=d*(float)s0,m1f=mn*(float)m0,d2=d*(float)s1,m2f=mn*(float)m1;
                for(int l=0;l<32;++l) f[base+j+l]=d1*(float)((ql[l]&0xF)+((qh[l]&u1)?16:0))-m1f;
                for(int l=0;l<32;++l) f[base+j+32+l]=d2*(float)((ql[l]>>4)+((qh[l]&u2)?16:0))-m2f;
                ql+=32; is+=2; u1<<=2; u2<<=2;
            }
        }
    } else if(t->ggml_type==18){
        const long long nb=(ne+255)/256;
        for(long long b=0;b<nb;b++){
            const unsigned char*blk=raw+(size_t)b*98;
            float d=f16h((unsigned short)(blk[0]|(blk[1]<<8)));
            const unsigned char*qsr=blk+2; const unsigned char*ss=qsr+64; long long base=b*256;
            for(int ib32=0;ib32<8;++ib32){
                unsigned aux32=ss[4*ib32]|(ss[4*ib32+1]<<8)|(ss[4*ib32+2]<<16)|((unsigned)ss[4*ib32+3]<<24);
                float db=d*(0.5f+(float)(aux32>>28))*0.5f;
                for(int l=0;l<4;++l){
                    int signs=w_ksigns_iq2xs[(aux32>>(7*l))&127];
                    const uint8_t*g1=(const uint8_t*)(&w_iq3xxs_grid[qsr[2*l+0]]);
                    const uint8_t*g2=(const uint8_t*)(&w_iq3xxs_grid[qsr[2*l+1]]);
                    for(int j=0;j<4;++j){
                        f[base+ib32*32+l*8+j]=db*(float)g1[j]*(signs&w_kmask_iq2xs[j]?-1.0f:1.0f);
                        f[base+ib32*32+l*8+j+4]=db*(float)g2[j]*(signs&w_kmask_iq2xs[j+4]?-1.0f:1.0f);
                    }
                }
                qsr+=8;
            }
        }
    } else if(t->ggml_type==30){ const unsigned short*bsrc=(const unsigned short*)raw;
        for(long long i=0;i<ne;i++){ unsigned u=((unsigned)bsrc[i])<<16; float v; memcpy(&v,&u,4); f[i]=v; } }
    else { fprintf(stderr,"unsupported type %d for %s\n",t->ggml_type,t->name); free(f); return 6; }
    unsigned short*hf=malloc((size_t)ne*2);
    for(long long i=0;i<ne;i++){
        unsigned u; memcpy(&u,&f[i],4); unsigned sig=(u>>16)&0x8000u; int e=((u>>23)&0xff)-127+15; unsigned m=u&0x7fffff;
        unsigned r;
        if(e>30) r=sig|0x7c00;
        else if(e<=0){ m|=0x800000u; int sh2=14-e; m>>=sh2; r=sig|(m>>13); if(m&0x1000) r++; }
        else { r=sig|((unsigned)e<<10)|(m>>13); if(m&0x1000) r++; }
        hf[i]=(unsigned short)r;
    }
    /* optional exact-f32 ref output (DQ_F32=1): write the f32 dequant as a _ref.f32 sibling */
    if(getenv("DQ_F32")){
        char rp[1600]; snprintf(rp,sizeof rp,"%s_ref.f32",outpath);
        FILE*rf=fopen(rp,"wb"); if(rf){ fwrite(f,sizeof(float),(size_t)ne,rf); fclose(rf); }
        fprintf(stderr,"f32 ref written %s\n",rp);
    }
    { size_t fz=(size_t)ne*2; FILE*out=fopen(outpath,"wb"); if(!out){fprintf(stderr,"cannot open %s\n",outpath);free(f);free(hf);return 7;}
      long long d0=t->n_dims>0?t->dims[0]:1,d1=t->n_dims>1?t->dims[1]:1;
      fwrite(&d0,8,1,out); fwrite(&d1,8,1,out); fwrite(hf,2,(size_t)ne,out); fclose(out); (void)fz; }
    free(f); free(hf); return 0;
}
int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: dequant_all <model.gguf> <outdir>\n");return 2;}
    gguf_ctx*c=gguf_open(argv[1]); if(!c){fprintf(stderr,"open fail\n");return 3;}
    gguf_buffer_data(c);
    long long tot=0; int n=0;
    for(int i=0;i<c->n_tensors;i++){
        if(strstr(c->tensors[i].name,"exps")) continue;
        long long ne=1; for(int d=0;d<c->tensors[i].n_dims;d++) ne*=c->tensors[i].dims[d];
        char p[1536];
        snprintf(p,sizeof p,"%s/idx%04d_%s.f16",argv[2],i,c->tensors[i].name);
        int rc=dequant_one(c,&c->tensors[i],p);
        fprintf(stderr,"%s ty=%d %lld el rc=%d\n",c->tensors[i].name,c->tensors[i].ggml_type,ne,rc);
        tot+=ne; n++;
    }
    fprintf(stderr,"DONE %d non-expert tensors, %lld elements = %.2f GB f16 @ %s\n",n,tot,(double)tot*2/1e9,argv[2]);
    return 0;
}
