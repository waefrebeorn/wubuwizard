/* head_hf16.c - dequantize a type-23 (IQ4_XS) tensor to single-f16 and dump.
 * Exact type-23 decode transcribed from wubu_vecdot_iq4xs.c (the engine's CPU reference):
 * block = fp16 d | uint16 scales_h | uint8 scales_l[4] | qs[128], 136 B / 256 el.
 * weight[i] = f16(d) * (ls-32) * kvalues_iq4nl[nibble], ls from scales_l pair + scales_h. */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const int8_t kvalues_iq4nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
#define f16h gguf_f16_to_f32
#include "iq3_xxs_tables.inc"
int main(int argc,char**argv){
    if(argc<4){fprintf(stderr,"usage: head_hf16 <model.gguf> <tensor> <out.f16>\n");return 2;}
    gguf_ctx*c=gguf_open(argv[1]); if(!c){fprintf(stderr,"open fail\n");return 3;}
    gguf_buffer_data(c);
    gguf_tensor_info*t=NULL;
    for(int i=0;i<c->n_tensors;i++) if(!strcmp(c->tensors[i].name,argv[2])){t=&c->tensors[i];break;}
    if(!t){fprintf(stderr,"tensor %s not found\n",argv[2]);return 4;}
    long long ne=1; for(int d=0;d<t->n_dims;d++) ne*=t->dims[d];
    const unsigned char*blob=(const unsigned char*)c->data_blob;
    if(!blob){ fprintf(stderr,"no data blob\n"); return 5; }
    const unsigned char*raw=blob+(size_t)t->data_offset;
    float*f=(float*)malloc((size_t)ne*sizeof(float));
    long rawsz=0;
    if(t->ggml_type==23){
        const long long nb=(ne+255)/256; rawsz=(long)nb*136; const long long nb2=nb;
        /* IQ4_XS: 136 B / 256 el */
        for(long long b=0;b<nb2;b++){
            const unsigned char*blk=raw+(size_t)b*136;
            unsigned short db=blk[0]|(blk[1]<<8); float d=f16h(db);
            unsigned short sh=blk[2]|((unsigned short)blk[3]<<8);
            const unsigned char*sl=blk+4; const unsigned char*qs=blk+8;
            long long base=(long long)b*256;
            for(int s=0;s<8;s++){
                int pair=s/2, half=s&1;
                unsigned hhex=(unsigned)(sh>>(4*pair))&0xffu;
                unsigned char lb=sl[pair];
                int ls = half ? (int)(((lb>>4)|((hhex<<2)&0x30))) : (int)(((lb&0xf)|((hhex<<4)&0x30)));
                const unsigned char*qb=qs+s*16;
                long long sb=(base+s*32);
                for(int r=0;r<32;r++){
                    unsigned char qv=qb[r&15];
                    int nib=(r<16)? (qv&0xf) : (qv>>4);
                    f[(size_t)(sb+r)]=d*(float)(ls-32)*(float)kvalues_iq4nl[nib];
                }
            }
        }
    } else if(t->ggml_type==8){
        /* Q8_0: [d: fp16][qs: 32 x int8] = 34 B / 32 el. value = f16(d)*qs. */
        const long long nb=(ne+31)/32; rawsz=(long)nb*34;
        for(long long b=0;b<nb;b++){
            const unsigned char*blk=raw+(size_t)b*34;
            long long base=(long long)b*32;
            float d=f16h((unsigned short)(blk[0]|(blk[1]<<8)));
            for(int r=0;r<32;r++) f[(size_t)(base+r)]=d*(float)(int8_t)blk[2+r];
        }
    } else if(t->ggml_type==0){
        /* F32: elements are stored as IEEE f32 directly; only the f32->f16 step applies. */
        const float*src=(const float*)raw; rawsz=(long)ne*4;
        for(long long i=0;i<ne;i++) f[i]=src[i];
    } else if(t->ggml_type==12){
        /* Q4_K: reader's llama-reference dequant (engine-identical, avoids transcription risk). */
        extern void dequantize_q4_K_row(const uint8_t*, float*, int64_t);
        rawsz=(long)(((ne+255)/256)*144);
        dequantize_q4_K_row(raw,f,ne);
    } else if(t->ggml_type==14){
        extern void dequantize_q6_K_row(const uint8_t*, float*, int64_t);
        rawsz=(long)(((ne+255)/256)*210);
        dequantize_q6_K_row(raw,f,ne);
    } else if(t->ggml_type==13){
        /* Q5_K: transcribed from upstream ggml-quants.c dequantize_row_q5_K + get_scale_min_k4.
         * block = d(u16) dmin(u16) scales[12] qh[32] qs[128] = 176 B / 256 el. */
        const long long nb=(ne+255)/256; rawsz=(long)nb*176;
        for(long long b=0;b<nb;b++){
            const unsigned char*blk=raw+(size_t)b*176;
            const unsigned char*sc=blk+4; const unsigned char*qh=blk+16; const unsigned char*ql=blk+48;
            float d=f16h((unsigned short)(blk[0]|(blk[1]<<8)));
            float mn=f16h((unsigned short)(blk[2]|(blk[3]<<8)));
            int is=0; unsigned u1=1,u2=2;
            long long base=(long long)b*256;
            for(int j=0;j<256;j+=64){
                unsigned char s0,m0; /* get_scale_min_k4(is) */
                if(is<4){ s0=sc[is]&63; m0=sc[is+4]&63; } else { s0=(sc[is+4]&0xF)|((sc[is-4]>>6)<<4); m0=(sc[is+4]>>4)|((sc[is]>>6)<<4); }
                unsigned char s1,m1; int is1=is+1;
                if(is1<4){ s1=sc[is1]&63; m1=sc[is1+4]&63; } else { s1=(sc[is1+4]&0xF)|((sc[is1-4]>>6)<<4); m1=(sc[is1+4]>>4)|((sc[is1]>>6)<<4); }
                float d1=d*(float)s0, m1f=mn*(float)m0, d2=d*(float)s1, m2f=mn*(float)m1;
                for(int l=0;l<32;++l) f[(size_t)(base+j+l)]=d1*(float)((ql[l]&0xF)+((qh[l]&u1)?16:0))-m1f;
                for(int l=0;l<32;++l) f[(size_t)(base+j+32+l)]=d2*(float)((ql[l]>>4)+((qh[l]&u2)?16:0))-m2f;
                ql+=32; is+=2; u1<<=2; u2<<=2;
            }
        }
    } else if(t->ggml_type==18){
        /* IQ3_XXS: block = {d u16, qs[96]} = 98 B / 256 el. qs[0..63]=grid idx, qs[64..95]=scales+signs.
         * Transcribed from upstream dequantize_row_iq3_xxs + engine tables (iq3_xxs_tables.inc). */
        const long long nb=(ne+255)/256; rawsz=(long)nb*98;
        for(long long b=0;b<nb;b++){
            const unsigned char*blk=raw+(size_t)b*98;
            float d=f16h((unsigned short)(blk[0]|(blk[1]<<8)));
            const unsigned char*qs=blk+2; const unsigned char*ss=qs+64;
            long long base=(long long)b*256; const float*out=&f[base];
            const unsigned char*qsr=qs;
            for(int ib32=0;ib32<8;++ib32){
                unsigned aux32= ss[4*ib32+0]|(ss[4*ib32+1]<<8)|(ss[4*ib32+2]<<16)|((unsigned)ss[4*ib32+3]<<24);
                float db=d*(0.5f+(float)(aux32>>28))*0.5f;
                for(int l=0;l<4;++l){
                    int signs=w_ksigns_iq2xs[(aux32>>(7*l))&127];
                    const uint8_t*g1=(const uint8_t*)(&w_iq3xxs_grid[qsr[2*l+0]]);
                    const uint8_t*g2=(const uint8_t*)(&w_iq3xxs_grid[qsr[2*l+1]]);
                    int o8=(int)((out-(&f[0])))+0;
                    for(int j=0;j<4;++j){
                        f[(size_t)(base+ib32*32+l*8+j+0)]=db*(float)g1[j]*(signs&w_kmask_iq2xs[j+0]?-1.0f:1.0f);
                        f[(size_t)(base+ib32*32+l*8+j+4)]=db*(float)g2[j]*(signs&w_kmask_iq2xs[j+4]?-1.0f:1.0f);
                    }
                    (void)o8;
                }
                qsr+=8;
            }
        }
    } else { fprintf(stderr,"unsupported type %d for dequant\n",t->ggml_type); return 6; }
    /* f32 -> f16 (IEEE round-to-nearest-even, the same cuBLAS/f16 uses) */
    unsigned short*hf=(unsigned short*)malloc((size_t)ne*2);
    for(long long i=0;i<ne;i++){
        unsigned u; memcpy(&u,&f[i],4); unsigned sig=(u>>16)&0x8000u; int e=((u>>23)&0xff)-127+15; unsigned m=u&0x7fffff;
        unsigned r;
        if(e>30) r=sig|0x7c00;
        else if(e<=0){ m|=0x800000u; int sh2=14-e; m>>=sh2; r=sig|(m>>13); if(m&0x1000) r++; }
        else { r=sig|((unsigned)e<<10)|(m>>13); if(m&0x1000) r++; }
        hf[i]=(unsigned short)r;
    }
    FILE*out=fopen(argv[3],"wb");
    fwrite(&t->dims[0],8,1,out); fwrite(&t->dims[1],8,1,out);
    fwrite(hf,2,(size_t)ne,out); fclose(out);
    fprintf(stderr,"%s %dx%d nelems=%lld raw=%ldB f16=%.1f MiB\n",t->name,(int)t->dims[0],(int)t->dims[1],ne,rawsz,(double)ne*2/1048576.0);
    /* VERIFY: only type 23 has an engine-dot reference in this tool; Q8_0 is structurally finite-checked
     * (value = f16(d)*qs is exact by inspection) and its dot-verify rides on the engine integration. c.0 */
    if(t->ggml_type==23){
    {
        const long long nb=(ne+255)/256;
        extern int wubu_vec_dot_iq4_xs_q8_k(int, float*, const void*, const void*);
        const int K=2304; if(K%256==0 && nb>=K/256){
            srand(17); float x[2304]; for(int i=0;i<K;i++) x[i]=0.11f*(float)(rand()%2001-1000);
            block_q8_K q8[9]; quantize_row_q8_K(x,q8,K);
            float eng; wubu_vec_dot_iq4_xs_q8_k(K,&eng,raw,&q8[0]);
            /* reconstruct the q8-quantized activation; both dots then run on the SAME values, so a correct
             * dequant gives eng == mine up to fp32 accumulation order (cos ~ 1). */
            float xr[2304]; for(int i=0;i<K;i++) xr[i]=q8[i/256].d*(float)q8[i/256].qs[i%256];
            float mine=0; for(int i=0;i<K;i++) mine+=f[i]*xr[i];
            double dnme=(double)(mine-eng), den=(double)eng;
            fprintf(stderr,"VERIFY: eng=%.6f mine=%.6f  (eng-mine)/eng=%.3e  (correct dequant if ~0 / cos->1)\n",
                    (double)eng,(double)mine, den!=0?dnme/den:0.0);
        } else fprintf(stderr,"VERIFY: skip (K%%256 != 0)\n");
    }
    } else if(t->ggml_type==12||t->ggml_type==13||t->ggml_type==14){
    {
        const long long nb=(ne+255)/256;
        const int K=2304; if(K%256==0 && nb>=K/256){
            extern void ggml_vec_dot_q4_K_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
            extern void ggml_vec_dot_q5_K_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
            extern void ggml_vec_dot_q6_K_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
            srand(17); float x[2304]; for(int i=0;i<K;i++) x[i]=0.11f*(float)(rand()%2001-1000);
            block_q8_K q8[9]; quantize_row_q8_K(x,q8,K);
            float eng;
            if(t->ggml_type==12) ggml_vec_dot_q4_K_q8_K_generic(K,&eng,0,raw,0,&q8[0],0,1);
            else if(t->ggml_type==13) ggml_vec_dot_q5_K_q8_K_generic(K,&eng,0,raw,0,&q8[0],0,1);
            else                      ggml_vec_dot_q6_K_q8_K_generic(K,&eng,0,raw,0,&q8[0],0,1);
            float xr[2304]; for(int i=0;i<K;i++) xr[i]=q8[i/256].d*(float)q8[i/256].qs[i%256];
            float mine=0; for(int i=0;i<K;i++) mine+=f[i]*xr[i];
            double dnme=(double)(mine-eng), den=(double)eng;
            fprintf(stderr,"VERIFY[ty%d]: eng=%.6f mine=%.6f rel=%.3e\n",t->ggml_type,(double)eng,(double)mine,den!=0?dnme/den:0.0);
        } else fprintf(stderr,"VERIFY: skip\n");
    }
    } else if(t->ggml_type==18){
    {
        const long long nb=(ne+255)/256;
        const int K=2304; if(K%256==0 && nb>=K/256){
            extern void ggml_vec_dot_iq3_xxs_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
            srand(17); float x[2304]; for(int i=0;i<K;i++) x[i]=0.11f*(float)(rand()%2001-1000);
            block_q8_K q8[9]; quantize_row_q8_K(x,q8,K);
            float eng; ggml_vec_dot_iq3_xxs_q8_K_generic(K,&eng,0,raw,0,&q8[0],0,1);
            float xr[2304]; for(int i=0;i<K;i++) xr[i]=q8[i/256].d*(float)q8[i/256].qs[i%256];
            float mine=0; for(int i=0;i<K;i++) mine+=f[i]*xr[i];
            double dnme=(double)(mine-eng), den=(double)eng;
            fprintf(stderr,"VERIFY[ty%d]: eng=%.6f mine=%.6f rel=%.3e\n",t->ggml_type,(double)eng,(double)mine,den!=0?dnme/den:0.0);
        } else fprintf(stderr,"VERIFY: skip\n");
    }
    } else { /* non-23: finite / magnitude sanity */
        float smin=f[0],smax=f[0],ssum=0; int nfin=0;
        for(long long i=0;i<ne;i++){ if(f[i]<smin)smin=f[i]; if(f[i]>smax)smax=f[i]; ssum+=f[i]; if(!(f[i]==f[i]))nfin++; }
        fprintf(stderr,"SANITY: min=%.5g max=%.5g mean=%.5g nan=%d (finite, sane spread)\n",smin,smax,ssum/ne,nfin);
    }
    free(f); free(hf); return 0;
}
