/* q8_0 row quantiser for the KV cache - see src/wubu_kv_q8.c */
#ifndef WUBU_KV_Q8_H
#define WUBU_KV_Q8_H
#include <stddef.h>
size_t kvq8_quantize_row(const float *x, unsigned char *out, int n);
void   kvq8_dequantize_row(const unsigned char *in, float *x, int n);
size_t kvq8_row_bytes(int n);
#endif
