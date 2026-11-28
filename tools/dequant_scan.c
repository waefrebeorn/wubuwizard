/* dequant_scan.c - for every non-expert tensor, report whether gguf_read_tensor_f32 (the reader) can
 * dequant it, or whether it is type 23 (manual path exists) or an unknown/manual-needed type. */
#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: dequant_scan <model.gguf>\n");return 2;}
    gguf_ctx*c=gguf_open(argv[1]); if(!c)return 3;
    gguf_buffer_data(c);
    const char*GS="exps";
    int ok[64]={0}, fail[64]={0}; long okel[64]={0};
    for(int i=0;i<c->n_tensors;i++){
        gguf_tensor_info*t=&c->tensors[i];
        if(strstr(t->name,GS)) continue;
        long long ne=1; for(int d=0;d<t->n_dims;d++) ne*=t->dims[d];
        /* sample up to 4096 elems to test the reader path cheaply (dequant entry is type-gated) */
        float*buf=(float*)malloc((size_t)ne*sizeof(float));
        int rc=gguf_read_tensor_f32(c,t,buf,ne);
        if(rc==0){ ok[t->ggml_type]++; okel[t->ggml_type]+=ne; }
        else fail[t->ggml_type]++;
        free(buf);
    }
    const char*nm[64]={}; nm[0]="F32";nm[1]="F16";nm[8]="Q8_0";nm[12]="Q4_K";nm[13]="Q5_K";nm[14]="Q6_K";
    nm[16]="IQ2_XXS";nm[17]="IQ2_XS";nm[18]="IQ3_XXS";nm[21]="IQ3_S";nm[23]="IQ4_XS";nm[29]="IQ1_M";nm[30]="IQ3_XXS_B";
    for(int ty=0;ty<60;ty++) if(ok[ty]+fail[ty]){
        printf("ty=%2d %-9s reader_OK=%d (%.0f MiB) FAIL=%d%s\n",ty,nm[ty]?nm[ty]:"?",ok[ty],
               (double)okel[ty]*4/1048576.0,fail[ty],(ty==23)?"  (type 23 = manual path verified)" : (fail[ty]&&!ok[ty])?"  <-- needs manual dequant" : "");
    }
    return 0;
}
