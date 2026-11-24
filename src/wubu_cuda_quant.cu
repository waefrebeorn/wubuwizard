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

/* Global, not __constant__: every lane indexes a different grid entry, and
 * constant memory serializes divergent addresses (32-way replay). */
__device__ uint64_t c_iq2s_grid[1024];
__device__ uint32_t c_iq3s_grid[512];
/* IQ3_XXS: 256 grid entries of 4 magnitudes each, and the 7-bit sign masks. */
__device__ uint32_t c_iq3xxs_grid[256];
__device__ uint8_t  c_ksigns_iq2xs[128];
/* IQ4_XS: 16 signed int8 code values. */
__device__ int8_t   c_iq4nl[16];

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
                /* __dp4a: 4 int8 MACs per instruction, signs folded into the grid bytes. */
                int32_t a0, a1, b0, b1;
                memcpy(&a0, g1, 4);
                memcpy(&a1, g2, 4);
                for (int j = 0; j < 4; ++j) {
                    if ((signs[l] >> j) & 1) ((int8_t *)&a0)[j] = (int8_t)(-((int8_t *)&a0)[j]);
                    if ((signs[l] >> (j + 4)) & 1) ((int8_t *)&a1)[j] = (int8_t)(-((int8_t *)&a1)[j]);
                }
                memcpy(&b0, q8 + 0, 4);
                memcpy(&b1, q8 + 4, 4);
                sumi += __dp4a(a0, b0, 0) + __dp4a(a1, b1, 0);
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

/* Lane-per-grid-entry mapping for IQ2_S: one warp per (column, 256-element
 * block). Lane l handles group l>>2 and grid entry l&3, so the 32 lanes cover
 * the block exactly and the 8-byte grid loads of adjacent lanes share cache
 * lines — no per-lane serial loop over sub-blocks. */
__global__ void iq2s_matvec_v2_kernel(const uint8_t * __restrict__ w, int type,
                                      const c_q8_k * __restrict__ xq8,
                                      float * __restrict__ y,
                                      int64_t col_stride_bytes, int nblocks, int n_cols)
{
    const int warps_per_block = blockDim.x / 32;
    const int wid = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int total = n_cols * nblocks;
    if (wid >= total) return;

    const int col = wid / nblocks;
    const int b = wid % nblocks;

    const uint8_t *blk = w + (int64_t)col * col_stride_bytes + (int64_t)b * 82;
    const uint16_t d_bits = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
    const uint8_t *qs = blk + 2;
    const uint8_t *qh = blk + 66;
    const uint8_t *scales = blk + 74;
    const c_q8_k *q8b = xq8 + b;

    const int group = lane >> 2;
    const int entry = lane & 3;

    const uint32_t idx = (uint32_t)(qs[group * 4 + entry] |
                         ((qh[group] << (8 - 2 * entry)) & 0x300));
    const uint8_t *grid = (const uint8_t *)(c_iq2s_grid + idx);
    const uint8_t signs = qs[32 + group * 4 + entry];
    const int8_t *q8 = q8b->qs + group * 32 + entry * 8;

    /* __dp4a: 4 int8 MACs per instruction. Grid magnitudes are 0..15 and the
     * signs are applied to the grid bytes, so both operands stay int8. */
    const int32_t g0 = (int32_t)((uint32_t)grid[0] | ((uint32_t)grid[1] << 8) |
                                 ((uint32_t)grid[2] << 16) | ((uint32_t)grid[3] << 24));
    const int32_t g1 = (int32_t)((uint32_t)grid[4] | ((uint32_t)grid[5] << 8) |
                                 ((uint32_t)grid[6] << 16) | ((uint32_t)grid[7] << 24));
    int32_t a0 = g0, a1 = g1;
    for (int j = 0; j < 4; ++j) {
        if ((signs >> j) & 1) ((int8_t *)&a0)[j] = (int8_t)(-((int8_t *)&a0)[j]);
        if ((signs >> (j + 4)) & 1) ((int8_t *)&a1)[j] = (int8_t)(-((int8_t *)&a1)[j]);
    }
    int32_t b0, b1;
    memcpy(&b0, q8 + 0, 4);
    memcpy(&b1, q8 + 4, 4);
    int sum = __dp4a(a0, b0, 0) + __dp4a(a1, b1, 0);

    const int ls = (entry < 2) ? (1 + 2 * (scales[group] & 0x0f))
                               : (1 + 2 * (scales[group] >> 4));
    int contrib = sum * ls;

    for (int off = 16; off > 0; off >>= 1)
        contrib += __shfl_down_sync(0xffffffffu, contrib, off);

    if (lane == 0)
        atomicAdd(&y[col], 0.125f * f16_to_f32_dev(d_bits) * q8b->d * (float)contrib);
}

/* Same for one whole column at once: y must be zeroed by the caller. */
WUBU_EXPORT int wubu_cuda_matvec_iq2s_v2(const void *d_w, const void *d_q8, float *d_y,
                                         int K, int n_cols, long long col_stride)
{
    if (!d_w || !d_q8 || !d_y || K <= 0 || n_cols <= 0 || (K % QK_K) != 0) return -1;
    const int nblocks = K / QK_K;
    const int warps_per_block = 8;
    const int total = n_cols * nblocks;
    const int blocks = (total + warps_per_block - 1) / warps_per_block;
    cudaMemsetAsync(d_y, 0, (size_t)n_cols * sizeof(float));
    iq2s_matvec_v2_kernel<<<blocks, warps_per_block * 32>>>(
        (const uint8_t *)d_w, 22, (const c_q8_k *)d_q8, d_y, col_stride, nblocks, n_cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

/* Same lane mapping as v2, but one warp owns a whole column and loops over its
 * 256-element blocks in registers: no atomicAdd, one write per column. */
__global__ void iq2s_matvec_v3_kernel(const uint8_t * __restrict__ w,
                                      const c_q8_k * __restrict__ xq8,
                                      float * __restrict__ y,
                                      int64_t col_stride_bytes, int nblocks, int n_cols)
{
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (col >= n_cols) return;

    const int group = lane >> 2;
    const int entry = lane & 3;

    /* Measured: staging the 8 KiB IQ2_S grid in shared memory was neutral
     * (0.26 vs 0.25 ms), so the lookups stay in global memory. */
    float sum = 0.0f;
    const uint8_t *wcol = w + (int64_t)col * col_stride_bytes;
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t *blk = wcol + (int64_t)b * 82;
        const uint16_t d_bits = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        const uint8_t *qs = blk + 2;
        const uint8_t *qh = blk + 66;
        const uint8_t *scales = blk + 74;
        const c_q8_k *q8b = xq8 + b;

        const uint32_t idx = (uint32_t)(qs[group * 4 + entry] |
                             ((qh[group] << (8 - 2 * entry)) & 0x300));
        const uint8_t *grid = (const uint8_t *)(c_iq2s_grid + idx);
        const uint8_t signs = qs[32 + group * 4 + entry];
        const int8_t *q8 = q8b->qs + group * 32 + entry * 8;

        int32_t a0, a1;
        memcpy(&a0, grid, 4);
        memcpy(&a1, grid + 4, 4);
        for (int j = 0; j < 4; ++j) {
            if ((signs >> j) & 1) ((int8_t *)&a0)[j] = (int8_t)(-((int8_t *)&a0)[j]);
            if ((signs >> (j + 4)) & 1) ((int8_t *)&a1)[j] = (int8_t)(-((int8_t *)&a1)[j]);
        }
        int32_t b0, b1;
        memcpy(&b0, q8 + 0, 4);
        memcpy(&b1, q8 + 4, 4);
        const int sumi = __dp4a(a0, b0, 0) + __dp4a(a1, b1, 0);
        const int ls = (entry < 2) ? (1 + 2 * (scales[group] & 0x0f))
                                   : (1 + 2 * (scales[group] >> 4));
        sum += 0.125f * f16_to_f32_dev(d_bits) * q8b->d * (float)(sumi * ls);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) y[col] = sum;
}

WUBU_EXPORT int wubu_cuda_matvec_iq2s_v3(const void *d_w, const void *d_q8, float *d_y,
                                         int K, int n_cols, long long col_stride)
{
    if (!d_w || !d_q8 || !d_y || K <= 0 || n_cols <= 0 || (K % QK_K) != 0) return -1;
    const int nblocks = K / QK_K;
    const int warps_per_block = 8;
    const int blocks = (n_cols + warps_per_block - 1) / warps_per_block;
    iq2s_matvec_v3_kernel<<<blocks, warps_per_block * 32>>>(
        (const uint8_t *)d_w, (const c_q8_k *)d_q8, d_y, col_stride, nblocks, n_cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
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
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

WUBU_EXPORT int wubu_cuda_upload_tables2(const void *iq2s, const void *iq3s)
{
    return wubu_cuda_upload_tables(iq2s, iq3s);
}

WUBU_EXPORT int wubu_cuda_upload_table_iq4nl(const void *tbl_16_i8)
{
    if (tbl_16_i8 && cudaMemcpyToSymbol(c_iq4nl, tbl_16_i8, 16) != cudaSuccess) return -1;
    return 0;
}

WUBU_EXPORT int wubu_cuda_upload_tables_iq3xxs(const void *grid_256_u32, const void *signs_128_u8)
{
    if (grid_256_u32 &&
        cudaMemcpyToSymbol(c_iq3xxs_grid, grid_256_u32, 256 * sizeof(uint32_t)) != cudaSuccess)
        return -1;
    if (signs_128_u8 &&
        cudaMemcpyToSymbol(c_ksigns_iq2xs, signs_128_u8, 128) != cudaSuccess)
        return -1;
    return 0;
}

/* IQ3_XXS x Q8_K, warp per column. A 256-element block is 8 sub-blocks of 32
 * elements x 4 grid-pair groups, so lane -> (ib32 = lane>>2, l = lane&3) covers
 * exactly one block with 32 lanes. Layout: fp16 d, qs[64] grid indices (8 per
 * sub-block), gas[32] (4 bytes per sub-block: 7-bit signs per group in bits 0-27,
 * 4-bit scale in bits 28-31). Mirrors ggml's scalar reference. */
__global__ void iq3xxs_matvec_kernel(const uint8_t * __restrict__ w,
                                     const c_q8_k * __restrict__ xq8,
                                     float * __restrict__ y,
                                     int64_t col_stride_bytes, int nblocks, int n_cols)
{
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (col >= n_cols) return;

    const int ib32 = lane >> 2;
    const int l = lane & 3;
    float sum = 0.0f;
    const uint8_t *wcol = w + (int64_t)col * col_stride_bytes;

    for (int b = 0; b < nblocks; ++b) {
        const uint8_t *blk = wcol + (int64_t)b * 98;
        const uint16_t d_bits = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        const uint8_t *q3 = blk + 2;
        const uint8_t *gas = blk + 66;
        const c_q8_k *q8b = xq8 + b;

        uint32_t aux32;
        memcpy(&aux32, gas + 4 * ib32, 4);
        const uint32_t ls = 2u * (aux32 >> 28) + 1u;
        const uint8_t signs = c_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];

        int32_t a0, a1;
        memcpy(&a0, &c_iq3xxs_grid[q3[8 * ib32 + 2 * l + 0]], 4);
        memcpy(&a1, &c_iq3xxs_grid[q3[8 * ib32 + 2 * l + 1]], 4);
        for (int j = 0; j < 4; ++j) {
            if ((signs >> j) & 1)       ((int8_t *)&a0)[j] = (int8_t)(-((int8_t *)&a0)[j]);
            if ((signs >> (j + 4)) & 1) ((int8_t *)&a1)[j] = (int8_t)(-((int8_t *)&a1)[j]);
        }
        const int8_t *q8 = q8b->qs + ib32 * 32 + l * 8;
        int32_t b0, b1;
        memcpy(&b0, q8 + 0, 4);
        memcpy(&b1, q8 + 4, 4);
        const int sumi = __dp4a(a0, b0, 0) + __dp4a(a1, b1, 0);
        sum += 0.25f * f16_to_f32_dev(d_bits) * q8b->d * (float)(sumi * (int)ls);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) y[col] = sum;
}

/* IQ4_XS x Q8_K, warp per column. Per 256-element block: 8 sub-blocks of 32
 * elements; lane -> (ib32 = lane>>2, quarter = lane&3) covers 4 low-nibble and 4
 * high-nibble elements, so 32 lanes cover the block exactly.
 * Layout: fp16 d | uint16 scales_h | scales_l[4] | qs[128]. */
__global__ void iq4xs_matvec_kernel(const uint8_t * __restrict__ w,
                                    const c_q8_k * __restrict__ xq8,
                                    float * __restrict__ y,
                                    int64_t col_stride_bytes, int nblocks, int n_cols)
{
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (col >= n_cols) return;

    const int ib32 = lane >> 2;
    const int quarter = lane & 3;
    float sum = 0.0f;
    const uint8_t *wcol = w + (int64_t)col * col_stride_bytes;

    for (int b = 0; b < nblocks; ++b) {
        const uint8_t *blk = wcol + (int64_t)b * 136;
        const uint16_t d_bits = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        uint16_t h;
        memcpy(&h, blk + 2, 2);
        const uint8_t *scales_l = blk + 4;
        const uint8_t *qs = blk + 8 + ib32 * 16;
        const c_q8_k *q8b = xq8 + b;

        /* A 16-byte chunk is one sub-block: low nibbles are elements 0..15 and
         * high nibbles elements 16..31 of the SAME sub-block, so both share its
         * scale. The two scales in scales_l[pair] belong to the even and odd
         * sub-blocks of the pair, not to the two nibble halves. */
        const int pair = ib32 >> 1;
        const uint16_t hp = (uint16_t)(h >> (4 * pair));
        const uint8_t ls1 = (uint8_t)((scales_l[pair] & 0x0f) | ((hp << 4) & 0x30));
        const uint8_t ls2 = (uint8_t)((scales_l[pair] >> 4) | ((hp << 2) & 0x30));
        const int ls = (ib32 & 1) ? (int)ls2 : (int)ls1;

        int8_t cl[4], ch[4];
        for (int k = 0; k < 4; ++k) {
            const uint8_t byte = qs[quarter * 4 + k];
            cl[k] = c_iq4nl[byte & 0x0f];
            ch[k] = c_iq4nl[byte >> 4];
        }
        int32_t al, ah, ql, qh;
        memcpy(&al, cl, 4);
        memcpy(&ah, ch, 4);
        /* q8 for this sub-block starts at ib32*32: each ib32 owns 32 activations. */
        const int8_t *q8 = q8b->qs + ib32 * 32 + quarter * 4;
        memcpy(&ql, q8, 4);
        memcpy(&qh, q8 + 16, 4);

        const int sl = __dp4a(al, ql, 0);
        const int sh = __dp4a(ah, qh, 0);
        const float d4d8 = f16_to_f32_dev(d_bits) * q8b->d;
        sum += d4d8 * (float)(ls - 32) * (float)(sl + sh);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) y[col] = sum;
}

WUBU_EXPORT int wubu_cuda_matvec_iq4xs(const void *d_w, const void *d_q8, float *d_y,
                                       int K, int n_cols, long long col_stride)
{
    if (!d_w || !d_q8 || !d_y || K <= 0 || n_cols <= 0 || (K % QK_K) != 0) return -1;
    const int warps_per_block = 8;
    const int blocks = (n_cols + warps_per_block - 1) / warps_per_block;
    iq4xs_matvec_kernel<<<blocks, warps_per_block * 32>>>(
        (const uint8_t *)d_w, (const c_q8_k *)d_q8, d_y, col_stride, K / QK_K, n_cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

WUBU_EXPORT int wubu_cuda_matvec_iq3xxs(const void *d_w, const void *d_q8, float *d_y,
                                        int K, int n_cols, long long col_stride)
{
    if (!d_w || !d_q8 || !d_y || K <= 0 || n_cols <= 0 || (K % QK_K) != 0) return -1;
    const int warps_per_block = 8;
    const int blocks = (n_cols + warps_per_block - 1) / warps_per_block;
    iq3xxs_matvec_kernel<<<blocks, warps_per_block * 32>>>(
        (const uint8_t *)d_w, (const c_q8_k *)d_q8, d_y, col_stride, K / QK_K, n_cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
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
    /* one warp per block: the max reduction is a warp shuffle. No sync here —
     * the kernels below are ordered on the same stream and the result download
     * is what synchronizes. bsums are unused by the four dots we run. */
    quantize_q8k_kernel<<<K / QK_K, 32>>>(d_x, (c_q8_k *)d_q8, K);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

/* Fully device-resident matvec: weights and Q8_K row already on the device. */
WUBU_EXPORT int wubu_cuda_matvec_dev(const void *d_w, int type, const void *d_q8,
                                     float *d_y, int K, int n_cols, long long col_stride)
{
    return wubu_cuda_quant_matvec(d_w, type, d_q8, d_y, K, n_cols, (int64_t)col_stride);
}

WUBU_EXPORT int wubu_cuda_matvec_iq2s_v2_host(const void *h_w, const void *h_xq8, float *h_y,
                                              int K, int n_cols, long long col_stride)
{
    if (!h_w || !h_xq8 || !h_y || K <= 0 || n_cols <= 0 || (K % QK_K) != 0) return -1;
    void *d_w = NULL, *d_q8 = NULL;
    float *d_y = NULL;
    int rc = -1;
    if (cudaMalloc(&d_w, (size_t)n_cols * (size_t)col_stride) != cudaSuccess) goto done;
    if (cudaMalloc(&d_q8, (size_t)(K / QK_K) * sizeof(c_q8_k)) != cudaSuccess) goto done;
    if (cudaMalloc(&d_y, (size_t)n_cols * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMemcpy(d_w, h_w, (size_t)n_cols * (size_t)col_stride, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(d_q8, h_xq8, (size_t)(K / QK_K) * sizeof(c_q8_k), cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (wubu_cuda_matvec_iq2s_v2(d_w, d_q8, d_y, K, n_cols, col_stride) != 0) goto done;
    if (cudaMemcpy(h_y, d_y, (size_t)n_cols * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    if (d_w) cudaFree(d_w);
    if (d_q8) cudaFree(d_q8);
    if (d_y) cudaFree(d_y);
    return rc;
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
    /* Dispatch the same way the engine's wubu_gpu_matvec() does, so this host
     * entry point can gate the type-specific kernels too. */
    int mrc;
    if (type == 18)      mrc = wubu_cuda_matvec_iq3xxs(d_w, d_xq8, d_y, K, n_cols, col_stride_bytes);
    else if (type == 23) mrc = wubu_cuda_matvec_iq4xs(d_w, d_xq8, d_y, K, n_cols, col_stride_bytes);
    else                 mrc = wubu_cuda_quant_matvec(d_w, type, d_xq8, d_y, K, n_cols, col_stride_bytes);
    if (mrc != 0) goto done;
    if (cudaMemcpy(h_y, d_y, (size_t)n_cols * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    if (d_w) cudaFree(d_w);
    if (d_xq8) cudaFree(d_xq8);
    if (d_y) cudaFree(d_y);
    return rc;
}
