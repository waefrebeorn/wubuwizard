/*
 * wubu_dims_stub.c — stubs for the safetensors probe, so the byte-budget
 * test can link wubu_runtime_dims.c WITHOUT the full safetensors reader (the
 * test uses wubu_runtime_dims_default(), not the checkpoint probe).
 */
#include "safetensors_reader.h"

st_ctx *st_open(const char *path) { (void)path; return NULL; }
const st_tensor_info *st_find_tensor(const st_ctx *ctx, const char *name)
    { (void)ctx; (void)name; return NULL; }
int64_t st_n_tensors(const st_ctx *ctx) { (void)ctx; return 0; }
const st_tensor_info *st_tensor_info_by_index(const st_ctx *ctx, int64_t idx)
    { (void)ctx; (void)idx; return NULL; }
void st_close(st_ctx *ctx) { (void)ctx; }
