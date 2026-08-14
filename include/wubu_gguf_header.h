#ifndef WUBU_GGUF_HEADER_H
#define WUBU_GGUF_HEADER_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* These are declared in gguf_reader.h (the master header) and defined here.
 * Other modules include this header to get forward declarations of the
 * internal helpers exported across the GGUF subsystem. */

/* File I/O context lifecycle */
gguf_ctx* gguf_open(const char *path);
void gguf_close(gguf_ctx *ctx);
int gguf_buffer_data(gguf_ctx *ctx);

/* Tensor lookup */
gguf_tensor_info* gguf_find_tensor(gguf_ctx *ctx, const char *name);

/* KV access (numeric + string) */
float gguf_read_kv_f32(const char *path, const char *key, float default_val);
int64_t gguf_read_kv_i64(gguf_ctx *ctx, const char *want, int64_t def);
int gguf_kv_get_i32(gguf_ctx *ctx, const char *key, int *out);
int gguf_kv_get_f32(gguf_ctx *ctx, const char *key, float *out);
int gguf_kv_get_i32_arr(gguf_ctx *ctx, const char *key, int *out, int max);

/* Tokenizer vocab count */
int64_t gguf_tokenizer_token_count(gguf_ctx *ctx);

/* Float16 -> Float32 (exported for dequant paths) */
float gguf_f16_to_f32(uint16_t h);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_GGUF_HEADER_H */