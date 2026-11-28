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
int main(int argc,char**argv){
    if(argc<4){fprintf(stderr,"usage: head_hf16 <model.gguf> <tensor> <out.f16>\n");return 2;}
    gguf_ctx*c=gguf_open(argv[1]); if(!c){fprintf(stderr,"open fail\n");return 3;}
    gguf_buffer_data(c);
    gguf_tensor_info*t=NULL;
    for(int i=0;i<c->n_tensors;i++) if(!strcmp(c->tensors[i].name,argv[2])){t=&c->tensors[i];break;}
    if(!t){fprintf(stderr,"tensor %s not found\n",argv[2]);return 4;}
    long long ne=1; for(int d=0;d<t->n_dims;d++) ne*=t->dims[d];
    const long long nb=(ne+255)/256;
    const long rawsz=(long)nb*136;                    /* IQ4_XS: 136 B / 256 el */
    const unsigned char*blob=(const unsigned char*)c->data_blob;
    if(!blob){ fprintf(stderr,"no data blob\n"); return 5; }
    const unsigned char*raw=blob+(size_t)t->data_offset;   /* offset from data-blob start */
    (void)gguf_read_raw_tensor; /* bypass the reader's type-23 gap: read raw directly */
    float*f=(float*)malloc((size_t)ne*sizeof(float));
    /* dequant IQ4_XS */
    for(int b=0;b<nb;b++){
        const unsigned char*blk=raw+(size_t)b*136;
        unsigned short db=blk[0]|(blk[1]<<8); float d=f16h(db);
        unsigned short sh=blk[2]|((unsigned short)blk[3]<<8);
        const unsigned char*sl=blk+4; const unsigned char*qs=blk+8;
        int base=(int)((size_t)b*256);
        for(int s=0;s<8;s++){
            int pair=s/2, half=s&1;
            unsigned hhex=(unsigned)(sh>>(4*pair))&0xffu;
            unsigned char lb=sl[pair];
            int ls = half ? (int)(((lb>>4)|((hhex<<2)&0x30))) : (int)(((lb&0xf)|((hhex<<4)&0x30)));
            const unsigned char*qb=qs+s*16;
            int sb=(base+s*32);
            for(int r=0;r<32;r++){
                unsigned char qv=qb[r&15];
                int nib=(r<16)? (qv&0xf) : (qv>>4);
                float w=d*(float)(ls-32)*(float)kvalues_iq4nl[nib];
                f[sb+r]=w;
            }
        }
    }
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
    /* VERIFY: my dequant as an f32 dot must match the engine's real type-23 dot (wubu_vec_dot_iq4_xs_q8_k)
     * on the same 9 blocks + a q8_K-quantized random activation. c.0 */
    {
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
    free(f); free(hf); return 0;
}
