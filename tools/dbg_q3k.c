
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "gguf_reader.h"
#include "wubu_vecdot_q3_k.h"
#define QK_K 256
static uint64_t st=0x5EEDBEEF1234ULL;
static unsigned r32(void){st^=st<<13;st^=st>>7;st^=st<<17;return (unsigned)(st>>32);}
int main(void){
  unsigned char b[110]; float w[QK_K]; float x[QK_K]; block_q8_K q8;
  unsigned d16=0x2C00u+(r32()&0x3FFu); memcpy(&b[108],&d16,2);
  for(int i=0;i<108;i++) b[i]=(unsigned char)r32();
  for(int j=0;j<QK_K;j++) x[j]=((float)(int)(r32()&0xFFFF)-32768.f)/4096.f;
  quantize_row_q8_K(x,&q8,QK_K);
  float got=0, sc=0; wubu_vec_dot_q3_k_q8_k(QK_K,&got,b,&q8); wubu_vec_dot_q3_k_q8_k_scalar(QK_K,&sc,b,&q8);
  gguf_dequantize(b,GGML_TYPE_Q3_K,QK_K,w);
  double acc=0; for(int j=0;j<QK_K;j++) acc += (double)w[j]*q8.d*q8.qs[j];
  printf("avx=%.6f scalar=%.6f ref=%.6f\n",got,sc,(float)acc);
  printf("first elems w: %.4f %.4f %.4f  q8: %d %d %d\n", w[0],w[1],w[2], q8.qs[0],q8.qs[1],q8.qs[2]);
  return 0;
}
