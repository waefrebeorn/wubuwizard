/*
 * wubu_cuda_quant.cu -- device-side quantized matvec kernels behind the DLL.
 *
 * Layout is the GGUF block layout as-is (one column of output = a contiguous run
 * of 256-element blocks), because repacking to tiles is a separate step
 * (docs/GPU-LANE-PLAN.md, port order 1). One warp handles one output column:
 * lane `l` owns block `l`, `l+32`, ... and the warp reduces at the end.
 *
 * Ported from the CPU kernels that the gates already prove exact:
 *   src/wubu_vecdot_iq.c      (IQ2_S, IQ3_S)
 *   src/wubu_vecdot_q3_k.c    (Q3_K)
 *   src/wubu_vecdot_q4_0.c    (Q4_0)
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define QK_K 256

#ifdef _WIN32
#define WUBU_EXPORT extern "C" __declspec(dllexport)
#else
#define WUBU_EXPORT extern "C" __attribute__((visibility("default")))
#endif

/* ---------- tables shared with the CPU kernels ---------- */

__constant__ uint64_t c_iq2s_grid[1024];
__constant__ uint32_t c_iq3s_grid[512];

WUBU_EXPORT int wubu_cuda_upload_tables(const void *iq2s_1024_u64, const void *iq3s_512_u32)
{
    if (iq2s_1024_u64 &&
        cudaMemcpyToSymbol(c_iq2s_grid, iq2s_1024_u64, 1024 * sizeof(uint64_t)) != cudaSuccess)
        return -1;
    if (iq3s_512_u32 &&
        cudaMemcpyToSymbol(c_iq3s_grid, iq3s_512_u32, 512 * sizeof(uint32_t)) != cudaSuccess)
        return -1;
    return 0;
}

/* ---------- block views ---------- */

typedef struct { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t scales[8]; } c_iq2_s;
typedef struct { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t signs[32]; uint8_t scales[4]; } c_iq3_s;
typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; } c_q3_k;
typedef struct { uint16_t d; uint8_t qs[16]; } c_q4_0;
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } c_q8_k;

__device__ __forceinline__ float f16_to_f32_dev(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    const uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign << 31;
        else {
            int shift = 0;
            while (!(mant & (1u << (10 - shift)))) ++shift;   /* normalize */
            const uint32_t m = (mant << shift) & 0x3FFu;
            bits = (sign << 31) | ((113u - (uint32_t)shift) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

__device__ __forceinline__ float iq2s_block_dot(const c_iq2_s *x, const c_q8_k *y)
{
    const int8_t *q8 = y->qs;
    const uint8_t *qs = x->qs;
    const uint8_t *qh = x->qh;
    const uint8_t *signs = qs + QK_K / 8;
    int bsum = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const int ls1 = 1 + 2 * (x->scales[ib32] & 0xf);
        const int ls2 = 1 + 2 * (x->scales[ib32] >> 4);
        int s1 = 0, s2 = 0;
        for (int l = 0; l < 2; ++l) {
            const uint8_t *g = (const uint8_t *)(c_iq2s_grid + (uint32_t)(qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
            for (int j = 0; j < 8; ++j)
                s1 += q8[j] * g[j] * ((signs[l] >> j) & 1 ? -1 : 1);
            q8 += 8;
        }
        for (int l = 2; l < 4; ++l) {
            const uint8_t *g = (const uint8_t *)(c_iq2s_grid + (uint32_t)(qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
            for (int j = 0; j < 8; ++j)
                s2 += q8[j] * g[j] * ((signs[l] >> j) & 1 ? -1 : 1);
            q8 += 8;
        }
        bsum += ls1 * s1 + ls2 * s2;
        qs += 4;
        signs += 4;
    }
    return 0.125f * (f16_to_f32_dev(x->d) * y->d) * (float)bsum;
}

__device__ __forceinline__ float iq3s_block_dot(const c_iq3_s *x, const c_q8_k *y)
{
    const int8_t *q8 = y->qs;
    const uint8_t *qs = x->qs;
    const uint8_t *qh = x->qh;
    const uint8_t *signs = x->signs;
    int bsum = 0;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        const int ls1 = 2 * (x->scales[ib32 / 2] & 0xf) + 1;
        const int ls2 = 2 * (x->scales[ib32 / 2] >> 4) + 1;
        for (int half = 0; half < 2; ++half) {
            int sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *g1 = (const uint8_t *)(c_iq3s_grid + (uint32_t)(qs[2 * l + 0] | ((qh[ib32 + half] << (8 - 2 * l)) & 256)));
                const uint8_t *g2 = (const uint8_t *)(c_iq3s_grid + (uint32_t)(qs[2 * l + 1] | ((qh[ib32 + half] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += g1[j] * q8[j + 0] * ((signs[l] >> j) & 1 ? -1 : 1);
                    sumi += g2[j] * q8[j + 4] * ((signs[l] >> (j + 4)) & 1 ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * (half ? ls2 : ls1);
        }
    }
    return (f16_to_f32_dev(x->d) * y->d) * (float)bsum;
}

__device__ __forceinline__ float q4_0_block_dot(const c_q4_0 *blks, int nsub, const c_q8_k *y)
{
    const int8_t *q8 = y->qs;
    float acc = 0.0f;
    for (int b = 0; b < nsub; ++b) {
        const c_q4_0 *blk = &blks[b];
        int s = 0;
        for (int j = 0; j < 16; ++j) {
            s += ((blk->qs[j] & 0x0f) - 8) * q8[b * 32 + j];
            s += ((blk->qs[j] >> 4) - 8) * q8[b * 32 + 16 + j];
        }
        acc += f16_to_f32_dev(blk->d) * (float)s;
    }
    return acc * y->d;
}

__device__ __forceinline__ float q3_k_block_dot(const c_q3_k *x, const c_q8_k *y)
{
    int8_t a[QK_K];
    const uint8_t *q3 = x->qs;
    const uint8_t *hm = x->hmask;
    uint8_t m = 1;
    int idx = 0;
    for (int j = 0; j < QK_K; j += 128) {
        for (int sh = 0; sh < 8; sh += 2) {
            for (int l = 0; l < 32; ++l) a[idx++] = (int8_t)((q3[l] >> sh) & 3) - ((hm[l] & m) ? 0 : 4);
            m <<= 1;
        }
        q3 += 32;
    }
    uint32_t auxs[4];
    memcpy(auxs, x->scales, 12);
    const uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
    auxs[3] = ((auxs[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
    auxs[0] = (auxs[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
    auxs[1] = (auxs[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
    const int8_t *scales = (const int8_t *)auxs;
    int32_t acc = 0;
    for (int g = 0; g < QK_K / 16; ++g) {
        int s = 0;
        for (int l = 0; l < 16; ++l) s += y->qs[g * 16 + l] * a[g * 16 + l];
        acc += (scales[g] - 32) * s;
    }
    return (f16_to_f32_dev(x->d) * y->d) * (float)acc;
}

/* ---------- warp-per-column matvec ---------- */

static const int WARP = 32;

__global__ void matvec_kernel(const void * __restrict__ w, int type,
                              const void * __restrict__ xq8, float * __restrict__ y,
                              int64_t col_stride_bytes, int nblocks, int n_cols)
{
    const int col = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
    const int lane = threadIdx.x % WARP;
    if (col >= n_cols) return;

    const uint8_t *wcol = (const uint8_t *)w + (int64_t)col * col_stride_bytes;
    const c_q8_k *q8 = (const c_q8_k *)xq8;

    float sum = 0.0f;
    for (int b = lane; b < nblocks; b += WARP) {
        switch (type) {
            case 22: /* IQ2_S */
                sum += iq2s_block_dot((const c_iq2_s *)wcol + b, q8 + b);
                break;
            case 21: /* IQ3_S */
                sum += iq3s_block_dot((const c_iq3_s *)wcol + b, q8 + b);
                break;
            case 11: /* Q3_K */
                sum += q3_k_block_dot((const c_q3_k *)wcol + b, q8 + b);
                break;
            case 2:  /* Q4_0: 8 sub-blocks per 256 elements */
                sum += q4_0_block_dot((const c_q4_0 *)wcol + (int64_t)b * 8, 8, q8 + b);
                break;
            default:
                break;
        }
    }
    for (int off = WARP / 2; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) y[col] = sum;
}

/* y[n_cols] = W[n_cols][K] . x[K], x already quantized to Q8_K on the device.
 * n_cols is passed explicitly because one warp per column needs the exact count. */
WUBU_EXPORT int wubu_cuda_quant_matvec(const void *d_w, int type,
                                       const void *d_xq8, float *d_y,
                                       int K, int n_cols, int64_t col_stride_bytes)
{
    if (!d_w || !d_xq8 || !d_y || K <= 0 || n_cols <= 0 || (K % QK_K) != 0) return -1;
    const int warps_per_block = 4;
    const int threads = WARP * warps_per_block;
    const int blocks = (n_cols + warps_per_block - 1) / warps_per_block;
    matvec_kernel<<<blocks, threads>>>((const void *)d_w, type, d_xq8, d_y,
                                       col_stride_bytes, K / QK_K, n_cols);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -2;
}

WUBU_EXPORT int wubu_cuda_upload_tables2(const void *iq2s, const void *iq3s)
{
    return wubu_cuda_upload_tables(iq2s, iq3s);
}

/* ---------- device-resident weights + on-device activation quantizer ---------- */

WUBU_EXPORT void *wubu_cuda_alloc(unsigned long long bytes)
{
    void *p = NULL;
    if (bytes == 0) return NULL;
    if (cudaMalloc(&p, (size_t)bytes) != cudaSuccess) return NULL;
    return p;
}

WUBU_EXPORT int wubu_cuda_upload(void *d_dst, const void *h_src, unsigned long long bytes)
{
    if (!d_dst || !h_src || bytes == 0) return -1;
    return cudaMemcpy(d_dst, h_src, (size_t)bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -2;
}

WUBU_EXPORT int wubu_cuda_download(void *h_dst, const void *d_src, unsigned long long bytes)
{
    if (!h_dst || !d_src || bytes == 0) return -1;
    return cudaMemcpy(h_dst, d_src, (size_t)bytes, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -2;
}

WUBU_EXPORT int wubu_cuda_free_dev(void *d_ptr)
{
    if (!d_ptr) return 0;
    return cudaFree(d_ptr) == cudaSuccess ? 0 : -1;
}

/* Same arithmetic as quantize_row_q8_K() in src/quantized_dot_generic.c. */
__global__ void quantize_q8k_kernel(const float * __restrict__ x, c_q8_k * __restrict__ y, int K)
{
    const int blk = blockIdx.x;
    const int lane = threadIdx.x;
    const int base = blk * QK_K;
    __shared__ float amax_sh;
    __shared__ float d_sh;

    /* Must mirror quantize_row_q8_K() exactly: max_val is the SIGNED element with
     * the largest magnitude, iscale = -127/max_val, and the stored block scale is
     * -max_val/127 (not the reciprocal). */
    float local_abs = 0.0f, local_val = 0.0f;
    for (int i = lane; i < QK_K; i += blockDim.x) {
        const float v = x[base + i];
        const float a = fabsf(v);
        if (a > local_abs) { local_abs = a; local_val = v; }
    }
    /* argmax over the block: reduce pairs of (abs, value) */
    for (int off = 16; off > 0; off >>= 1) {
        const float oa = __shfl_xor_sync(0xffffffffu, local_abs, off);
        const float ov = __shfl_xor_sync(0xffffffffu, local_val, off);
        if (oa > local_abs) { local_abs = oa; local_val = ov; }
    }
    if (lane == 0) { amax_sh = local_abs; d_sh = local_val; }
    __syncthreads();

    const float max_val = d_sh;
    if (max_val == 0.0f) {
        for (int i = lane; i < QK_K; i += blockDim.x) y[blk].qs[i] = 0;
        if (lane == 0) y[blk].d = 0.0f;
        __syncthreads();
        return;
    }
    const float iscale = -127.0f / max_val;
    for (int i = lane; i < QK_K; i += blockDim.x) {
        float v = iscale * x[base + i];
        v = (x[base + i] >= 0.0f) ? floorf(v + 0.5f) : ceilf(v - 0.5f);
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        y[blk].qs[i] = (int8_t)v;
    }
    if (lane == 0) y[blk].d = -max_val / 127.0f;
}

/* Quantize K F32 activations (already on device) into Q8_K blocks on device. */
__global__ void bsums_kernel(c_q8_k *y, int nblk)
{
    const int blk = blockIdx.x;
    if (blk >= nblk) return;
    const int i = threadIdx.x;
    if (i < QK_K / 16) {
        int sum = 0;
        for (int j = 0; j < 16; ++j) sum += y[blk].qs[i * 16 + j];
        y[blk].bsums[i] = (int16_t)sum;
    }
}

WUBU_EXPORT int wubu_cuda_quantize_q8k_dev(const float *d_x, void *d_q8, int K)
{
    if (!d_x || !d_q8 || K <= 0 || (K % QK_K) != 0) return -1;
    /* one warp per block: the max reduction is a warp shuffle */
    quantize_q8k_kernel<<<K / QK_K, 32>>>(d_x, (c_q8_k *)d_q8, K);
    if (cudaDeviceSynchronize() != cudaSuccess) return -2;
    bsums_kernel<<<K / QK_K, 32>>>((c_q8_k *)d_q8, K / QK_K);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -3;
}

/* Fully device-resident matvec: weights and Q8_K row already on the device. */
WUBU_EXPORT int wubu_cuda_matvec_dev(const void *d_w, int type, const void *d_q8,
                                     float *d_y, int K, int n_cols, long long col_stride)
{
    return wubu_cuda_quant_matvec(d_w, type, d_q8, d_y, K, n_cols, (int64_t)col_stride);
}

/* Host wrapper for gates: takes F32 activations, quantizes on the host, uploads. */
WUBU_EXPORT int wubu_cuda_quant_matvec_host(const void *h_w, int type,
                                            const void *h_xq8, float *h_y,
                                            int K, int n_cols, int64_t col_stride_bytes)
{
    if (!h_w || !h_xq8 || !h_y || K <= 0 || n_cols <= 0 || (K % QK_K) != 0) return -1;
    void *d_w = NULL, *d_xq8 = NULL;
    float *d_y = NULL;
    int rc = -1;
    const size_t wb = (size_t)n_cols * (size_t)col_stride_bytes;
    const size_t xb = (size_t)(K / QK_K) * sizeof(c_q8_k);
    if (cudaMalloc(&d_w, wb) != cudaSuccess) goto done;
    if (cudaMalloc(&d_xq8, xb) != cudaSuccess) goto done;
    if (cudaMalloc(&d_y, (size_t)n_cols * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMemcpy(d_w, h_w, wb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(d_xq8, h_xq8, xb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (wubu_cuda_quant_matvec(d_w, type, d_xq8, d_y, K, n_cols, col_stride_bytes) != 0) goto done;
    if (cudaMemcpy(h_y, d_y, (size_t)n_cols * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    if (d_w) cudaFree(d_w);
    if (d_xq8) cudaFree(d_xq8);
    if (d_y) cudaFree(d_y);
    return rc;
}
