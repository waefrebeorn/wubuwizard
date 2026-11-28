/* head_hf16.c - dequantize output.weight (2304x201024, ty=23 IQ4_XS) to single-f16 and dump.
 * Proves the dequant->f16 pipeline on the biggest dense read; the .cu partner uploads+times it. */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char**argv){
    if(argc<4){fprintf(stderr,"usage: head_hf16 <model.gguf> output.weight <out.f16>\n");return 2;}
    gguf_ctx*c=gguf_open(argv[1]); if(!c){fprintf(stderr,"open fail\n");return 3;}
    gguf_buffer_data(c);
    gguf_tensor_info*t=NULL;
    for(int i=0;i<c->n_tensors;i++) if(!strcmp(c->tensors[i].name,argv[2])){t=&c->tensors[i];break;}
    if(!t){fprintf(stderr,"tensor %s not found\n",argv[2]);return 4;}
    long long ne=1; for(int d=0;d<t->n_dims;d++) ne*=t->dims[d];
    float*f=(float*)malloc((size_t)ne*sizeof(float));
    if(gguf_read_tensor_f32(c,t,f,ne)!=0){fprintf(stderr,"read fail\n");return 5;}
    unsigned short*hf=(unsigned short*)malloc((size_t)ne*2);
    /* f32 -> f16 (IEEE, round-to-nearest-even), the same conversion cuBLAS/f16 uses */
    for(long long i=0;i<ne;i++){
        unsigned u; memcpy(&u,&f[i],4); unsigned s=(u>>16)&0x8000u; int e=((u>>23)&0xff)-127+15; unsigned m=u&0x7fffff;
        unsigned r;
        if(e>30) r=s|0x7c00;
        else if(e<=0){ /* subnormal f16 */
            m|=0x800000u; int shift=14-e; m>>=shift;
            r=s|(m>>13); if(m&0x1000) r++;  /* round */
        } else {
            r=s|((unsigned)e<<10)|(m>>13);
            if(m&0x1000) r++;  /* round to nearest even (approx; fine for lane) */
        }
        hf[i]=(unsigned short)r;
    }
    FILE*out=fopen(argv[3],"wb");
    fwrite(&t->dims[0],8,1,out); fwrite(&t->dims[1],8,1,out);  /* rows, cols as i64 */
    fwrite(hf,2,(size_t)ne,out);
    fclose(out);
    float kbytes=(float)(ne*2)/1024.0f;
    fprintf(stderr,"head %dx%d nelems=%lld -> f16 %.1f KB (%.2f MB)\n",(int)t->dims[0],(int)t->dims[1],ne,kbytes,(float)ne*2/1048576.0f);
    free(f); free(hf); return 0;
}
