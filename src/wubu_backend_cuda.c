/* wubu_backend_cuda.c — CUDA backend for wubu_backend_t vtable.
 * Populates the vtable with CUDA function pointers when GPU_SUPPORT
 * is defined.  This replaces the #ifdef GPU_SUPPORT blocks in
 * wubu_model.c with a unified dispatch.
 *
 * ADR-002 opaque-struct seam: wubu_backend_t is an opaque pointer
 * stored in wubu_model_t.  Consumers include wubu_backend.h and
 * call wubu_backend_*() — never touch the struct fields directly.
 */

#include "wubu_backend.h"
#include "wubu_model_gpu.h"
#include "wubu_kvfs.h"   /* ADR-003: namespace I/O (flat-tensor route) */
#include <cuda_runtime.h> /* cudaStream_t — nvcc does not auto-include for .c */

static int cuda_backend_init(wubu_model_t *model, int max_ctx, int chunk_sz) {
    if (chunk_sz <= 0) {
        fprintf(stderr, "GPU: invalid chunk_sz=%d (must be >0)\n", chunk_sz);
        return 0;
    }
    return wubu_model_gpu_init(model, max_ctx, chunk_sz);
}

static void cuda_backend_free(wubu_model_t *model) {
    wubu_model_gpu_free(model);
}

static int cuda_backend_gqa_forward(wubu_model_t *model, int layer_idx,
                                         const float *h_norm, int C,
                                         float *h_attn) {
    return wubu_model_gpu_gqa_forward(model, layer_idx, h_norm, C, h_attn);
}

static int cuda_backend_ssm_forward_prefill(wubu_model_t *model, int layer_idx,
                                                 const float *h_norm, int C,
                                                 float *h_attn_out) {
    return wubu_model_gpu_ssm_forward_full(model, layer_idx, h_norm, C, h_attn_out);
}

static int cuda_backend_ssm_project(wubu_model_t *model, int layer_idx,
                                         const float *h_norm, int C,
                                         float *qkv_out, float *z_out,
                                         float *ssm_out_out) {
    return wubu_model_gpu_ssm_project(model, layer_idx, h_norm, C,
                                         qkv_out, z_out, ssm_out_out);
}

static void cuda_backend_moe_experts(const moe_weights_t *w,
                                         const float *x_s,
                                         const int *indices_s, const float *weights_s,
                                         float *expert_contribs,
                                         void *model_ptr) {
    wubu_model_gpu_moe_experts(w, x_s, indices_s, weights_s, expert_contribs, model_ptr);
}

static int cuda_backend_quant_matmul(const float *x, int n_rows, int n_cols,
                                        const uint8_t *d_W_q, int quant_type,
                                        float *y, void *stream) {
    return wubu_model_gpu_quant_matmul(x, n_rows, n_cols, d_W_q, quant_type, y,
                                       (cudaStream_t)stream);
}

/* SSM hybrid decode: route through the backend vtable so CPU builds
 * (which have no device buffers) get a no-op and GPU builds get the
 * real CUDA-backed implementation. Keeps wubu_model.c free of
 * #ifdef GPU_SUPPORT. */
static void cuda_backend_set_ssm_hybrid(wubu_model_t *model, int layer_idx,
                                         struct ssm_layer_weights *ssm) {
    if (!model || !model->gpu_ctx) return;
    wubu_gpu_set_ssm_hybrid(model->gpu_ctx, layer_idx, ssm);
}

static void cuda_backend_sync_ssm_state_to_gpu(wubu_model_t *model,
                                                int layer_idx,
                                                const float *cpu_ssm_state,
                                                const float *cpu_conv_state) {
    if (!model || !model->gpu_ctx) return;
    wubu_gpu_sync_ssm_state_to_gpu(model->gpu_ctx, layer_idx,
                                   cpu_ssm_state, cpu_conv_state);
}

static int cuda_backend_chunk_size(wubu_model_t *model) {
    return wubu_model_gpu_chunk_sz(model);
}

/* ADR-003: KV cache as filesystem — backend-accelerated namespace I/O.
 * For now the CUDA backend routes namespace reads/writes through the
 * host wubu_kvfs handle passed per-call by wubu_generate (the model
 * struct has no kvfs field in the current architecture); NULL entries
 * mean pure CPU fallback. Device-resident KV pages are a later
 * milestone. */
static int cuda_backend_kvfs_read(wubu_model_t *model, const char *path,
                                  float *dst, size_t n_floats) {
    (void)model; (void)path; (void)dst; (void)n_floats;
    return -1;  /* host-side KVFS is wired via wubu_generate's handle */
}

static int cuda_backend_kvfs_write(wubu_model_t *model, const char *path,
                                   const float *src, size_t n_floats) {
    (void)model; (void)path; (void)src; (void)n_floats;
    return -1;  /* host-side KVFS is wired via wubu_generate's handle */
}

static int cuda_backend_kvfs_snapshot(wubu_model_t *model,
                                      const char *path,
                                      float *dst, size_t n_floats) {
    (void)model; (void)path; (void)dst; (void)n_floats;
    return -1;  /* device-resident snapshot not yet implemented */
}

static wubu_backend_t wubu_backend_cuda = {
    .capabilities = WUBU_BACKEND_CPU | WUBU_BACKEND_CUDA_GEMV | WUBU_BACKEND_CUDA_SSM |
                    WUBU_BACKEND_CUDA_MOE | WUBU_BACKEND_CUDA_ATTN,
    .init = cuda_backend_init,
    .free = cuda_backend_free,
    .gqa_forward = cuda_backend_gqa_forward,
    .ssm_forward_prefill = cuda_backend_ssm_forward_prefill,
    .ssm_project = cuda_backend_ssm_project,
    .moe_experts = cuda_backend_moe_experts,
    .quant_matmul = cuda_backend_quant_matmul,
    .set_ssm_hybrid = cuda_backend_set_ssm_hybrid,
    .sync_ssm_state_to_gpu = cuda_backend_sync_ssm_state_to_gpu,
    .chunk_size = cuda_backend_chunk_size,
    .kvfs_read = cuda_backend_kvfs_read,
    .kvfs_write = cuda_backend_kvfs_write,
    .kvfs_snapshot = cuda_backend_kvfs_snapshot,
};

wubu_backend_t *wubu_backend_cuda_get(void) {
    return &wubu_backend_cuda;
}