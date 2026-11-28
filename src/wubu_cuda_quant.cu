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
#include "wubu_expf.h"   /* one deterministic expf, shared with the CPU attention */
#include "wubu_trig.h"   /* one deterministic sin/cos, likewise */
#include <cuda_fp16.h>
#include <cublas_v2.h>
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
__device__ uint64_t c_iq2xxs_grid[256];
typedef struct { uint16_t d; int8_t qs[32]; } c_q8_0;
/* IQ4_XS: 16 signed int8 code values. */
__device__ int8_t   c_iq4nl[16];
/* IQ2_XS: 512 grid entries of 8 unsigned bytes each. */
__device__ uint64_t c_iq2xs_grid[512];

__global__ void expf_probe_kernel(const float * __restrict__ x, float * __restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = wubu_expf(x[i]);
}

/* Gate hook: does the device expf agree with the CPU's bit for bit? The fused attention
 * path's residual 3.5e-07 is either this or the value loop, and guessing has cost enough. */
WUBU_EXPORT int wubu_cuda_expf_probe(const float *h_x, float *h_y, int n)
{
    if (!h_x || !h_y || n <= 0) return -1;
    float *dx = NULL, *dy = NULL;
    int rc = -2;
    if (cudaMalloc(&dx, (size_t)n * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&dy, (size_t)n * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMemcpy(dx, h_x, (size_t)n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    expf_probe_kernel<<<(n + 255) / 256, 256>>>(dx, dy, n);
    if (cudaGetLastError() != cudaSuccess) goto done;
    if (cudaMemcpy(h_y, dy, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    if (dx) cudaFree(dx);
    if (dy) cudaFree(dy);
    return rc;
}

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
/* The M-row IQ3_XXS/IQ4_XS path is on by default: it matches the per-token reference more often than
 * the kernels-off path (13 of 16 against 9 of 16) and is 1.8x faster at prefill. G4_XXS_MMA=0 disables. */
static int xxs_mrow_enabled(void)
{
    const char *e = getenv("G4_XXS_MMA");
    return !(e && e[0] == '0');
}

typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; } c_q3_k;
typedef struct { uint16_t d; uint8_t qs[16]; } c_q4_0;
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } c_q8_k;

/* int8 tensor-core matrix multiply-accumulate: 16 rows x 8 cols x 32 k in one instruction, against
 * dp4a's four multiply-adds per lane. Used by the prefill path only. */
__device__ __forceinline__ void mma_i8_s32(int32_t c[4], const uint32_t a[4], const uint32_t b[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/* Hardware fp16 -> fp32 when the toolkit exposes it: the hand-written version below
 * has a normalise loop for subnormals, and at one conversion per lane per block it is
 * a measurable share of the kernel's instructions. */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
__device__ __forceinline__ float f16_fast(uint16_t h)
{
    return __half2float(__ushort_as_half(h));
}
#else
__device__ __forceinline__ float f16_fast(uint16_t h);
#endif

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



/* Same idle-lane problem at the attention projections: q/k/v are 15 QK_K blocks per column, so the
 * warp-per-column mapping leaves 17 of 32 lanes idle and the attention phase ran at ~28 GB/s against
 * the FFN's 65. Two columns per warp, reduction inside each 16-lane half. */
__global__ void iq2s_matvec_w16_kernel(const uint8_t * __restrict__ w,
                                       const c_q8_k * __restrict__ xq8,
                                       float * __restrict__ y,
                                       int64_t col_stride_bytes, int nblocks, int n_cols)
{
    const int wid = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int sub = lane >> 4, blk = lane & 15;
    const int col = wid * 2 + sub;
    if (col >= n_cols) return;
    const uint8_t *wcol = w + (int64_t)col * col_stride_bytes;
    float sum = 0.0f;
    for (int b = blk; b < nblocks; b += 16)
        sum += iq2s_block_dot((const c_iq2_s *)wcol + b, xq8 + b);
    for (int off = 8; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (blk == 0) y[col] = sum;
}

__global__ void iq3s_matvec_w16_kernel(const uint8_t * __restrict__ w,
                                       const c_q8_k * __restrict__ xq8,
                                       float * __restrict__ y,
                                       int64_t col_stride_bytes, int nblocks, int n_cols)
{
    const int wid = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int sub = lane >> 4, blk = lane & 15;
    const int col = wid * 2 + sub;
    if (col >= n_cols) return;
    const uint8_t *wcol = w + (int64_t)col * col_stride_bytes;
    float sum = 0.0f;
    for (int b = blk; b < nblocks; b += 16)
        sum += iq3s_block_dot((const c_iq3_s *)wcol + b, xq8 + b);
    for (int off = 8; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (blk == 0) y[col] = sum;
}

/* Q3_K decode matvec with two columns per warp. matvec_kernel gives one warp per column, but a
 * 3840-wide row is 15 QK_K blocks, so 17 of 32 lanes sat idle and the LM head measured ~36 GB/s
 * against the FFN's ~65. Here lanes 0-15 and 16-31 take one column each, so all but two lanes are
 * doing work. Reduction stays inside each 16-lane half. */
__global__ void q3k_matvec_w16_kernel(const uint8_t * __restrict__ w,
                                      const c_q8_k * __restrict__ xq8,
                                      float * __restrict__ y,
                                      int64_t col_stride_bytes, int nblocks, int n_cols)
{
    const int wid = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int sub = lane >> 4;
    const int blk = lane & 15;
    const int col = wid * 2 + sub;
    if (col >= n_cols) return;
    const uint8_t *wcol = w + (int64_t)col * col_stride_bytes;
    float sum = 0.0f;
    for (int b = blk; b < nblocks; b += 16)
        sum += q3_k_block_dot((const c_q3_k *)wcol + b, xq8 + b);
    for (int off = 8; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (blk == 0) y[col] = sum;
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
        atomicAdd(&y[col], 0.125f * f16_fast(d_bits) * q8b->d * (float)contrib);
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
        sum += 0.125f * f16_fast(d_bits) * q8b->d * (float)(sumi * ls);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) y[col] = sum;
}


/* ---------------- device workspace pool ----------------
 *
 * Every device scratch buffer in this file used to be its own static with its own cudaMalloc:
 * the decode q8 row, the M-row x/y, the M-row q8 block. They are all the same thing - a staging
 * area for a shape - and the contents are irrelevant because the producer always writes the whole
 * thing before the consumer reads it. So they are declared here by ROLE and capacity, materialised
 * on first use, and reused for any later tensor of the same shape even though the data differs.
 *
 * That is the whole point: the staging is recycled, not the data. A prompt of 87 rows and a
 * decode step are the same declarations at different sizes, and nothing allocates inside a
 * forward pass once the shapes have been seen once. */
typedef struct { const char *role; size_t bytes; void *ptr; unsigned uses; } c_pool_slot_t;
static c_pool_slot_t g_pool[24];
static int g_pool_n;
static unsigned long long g_pool_bytes;

static void *pool_get(const char *role, size_t bytes)
{
    if (bytes == 0) return NULL;
    for (int i = 0; i < g_pool_n; ++i) {
        c_pool_slot_t *s = &g_pool[i];
        if (s->role != role) continue;
        if (s->bytes >= bytes) { ++s->uses; return s->ptr; }
        /* Same role, bigger shape: grow in place, keeping the slot. */
        void *np = NULL;
        if (cudaMalloc(&np, bytes) != cudaSuccess) return NULL;
        if (s->ptr) cudaFree(s->ptr);
        g_pool_bytes += bytes - s->bytes;
        s->ptr = np; s->bytes = bytes; ++s->uses;
        return s->ptr;
    }
    if (g_pool_n >= (int)(sizeof(g_pool) / sizeof(g_pool[0]))) return NULL;
    void *np = NULL;
    if (cudaMalloc(&np, bytes) != cudaSuccess) return NULL;
    g_pool[g_pool_n].role = role;
    g_pool[g_pool_n].bytes = bytes;
    g_pool[g_pool_n].ptr = np;
    g_pool[g_pool_n].uses = 1;
    ++g_pool_n;
    g_pool_bytes += bytes;
    return np;
}

/* G4_POOL_STATS=1 prints this: role, bytes, and how many times each was handed out. */
WUBU_EXPORT void wubu_cuda_pool_stats(void)
{
    fprintf(stderr, "[POOL] %d slots, %.1f MiB total\n", g_pool_n, (double)g_pool_bytes / (1024.0 * 1024.0));
    for (int i = 0; i < g_pool_n; ++i)
        fprintf(stderr, "[POOL]   %-12s %8.2f MiB  uses=%u\n", g_pool[i].role,
                (double)g_pool[i].bytes / (1024.0 * 1024.0), g_pool[i].uses);
}

/* ---------------- M-row IQ2_S: amortise the weight decode over prompt rows ----------------
 *
 * Prefill is the profile that matters and it was 7.3 tok/s at 1717 tokens: 235 s before the
 * first output token. The per-row path cannot fix it - M rows through an M==1 kernel is M
 * synchronised round trips (~400k for that prompt) and measured 3.4 tok/s - and the CPU
 * batched path loses to the GPU because a Ryzen 5 has to stream 4 GB of weights the 4050
 * already holds.
 *
 * The cost that a GEMM actually amortises here is not the dp4a, it is the IQ2_S decode: the
 * grid lookup, the eight sign tests, the scale nibble and the block scale are identical for
 * every prompt row and were being redone per row. This kernel decodes a block ONCE into
 * registers and reuses it across MT rows, so the same weight pass serves MT tokens.
 *
 * M=1 takes the same arithmetic path as iq2s_matvec_v3_kernel (same block order, same lane
 * sum, same shuffle), so the two are comparable bit for bit - EG-130 checks that. */
__global__ void quantize_q8k_kernel(const float * __restrict__ x, c_q8_k * __restrict__ y, int K,
                                    int xrow_stride, int yrow_stride);
WUBU_EXPORT int wubu_cuda_quant_matvec_host(const void *h_w, int type, const void *h_xq8,
                                            float *h_y, int K, int n_cols,
                                            int64_t col_stride_bytes);

/* MT=8 measured 16.1 tok/s against MT=4's 17.2 at 87 tokens: more rows per weight pass
 * costs occupancy and loses. */
#define IQ2S_MT 16
__global__ void iq2s_matvec_mrow_kernel(const uint8_t * __restrict__ w,
                                        const c_q8_k * __restrict__ xq8m,
                                        float * __restrict__ y,
                                        int64_t col_stride_bytes, int nblocks,
                                        int n_cols, int M, int q8_row_blocks)
{
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (col >= n_cols) return;

    const int group = lane >> 2;
    const int entry = lane & 3;
    const uint8_t *wcol = w + (int64_t)col * col_stride_bytes;

    for (int m0 = 0; m0 < M; m0 += IQ2S_MT) {
        const int mt = (M - m0) < IQ2S_MT ? (M - m0) : IQ2S_MT;
        float acc[IQ2S_MT];
        for (int t = 0; t < IQ2S_MT; ++t) acc[t] = 0.0f;

        for (int b = 0; b < nblocks; ++b) {
            const uint8_t *blk = wcol + (int64_t)b * 82;
            const uint16_t d_bits = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
            const uint8_t *qs = blk + 2;
            const uint8_t *qh = blk + 66;
            const uint8_t *scales = blk + 74;

            /* decoded once, reused for every row in this tile */
            const uint32_t idx = (uint32_t)(qs[group * 4 + entry] |
                                 ((qh[group] << (8 - 2 * entry)) & 0x300));
            const uint8_t *grid = (const uint8_t *)(c_iq2s_grid + idx);
            const uint8_t signs = qs[32 + group * 4 + entry];
            int32_t a0, a1;
            memcpy(&a0, grid, 4);
            memcpy(&a1, grid + 4, 4);
            /* Eight predicated negations become two byte-wise ones. The nibble is spread to one
             * byte per sign bit by multiplying through 2^0,2^7,2^14,2^21 (bits land on 0,8,16,24),
             * then widened to 0xFF so __vsub4 negates exactly the bytes whose sign was set. */
            const uint32_t m_lo = (((uint32_t)(signs & 0x0F) * 0x00204081u) & 0x01010101u) * 0xFFu;
            const uint32_t m_hi = (((uint32_t)(signs >> 4) * 0x00204081u) & 0x01010101u) * 0xFFu;
            a0 = (int32_t)__vsub4((uint32_t)a0 ^ m_lo, m_lo);
            a1 = (int32_t)__vsub4((uint32_t)a1 ^ m_hi, m_hi);
            const int ls = (entry < 2) ? (1 + 2 * (scales[group] & 0x0f))
                                       : (1 + 2 * (scales[group] >> 4));
            const float wd = 0.125f * f16_fast(d_bits) * (float)ls;

            const c_q8_k *rowbase = xq8m + (int64_t)m0 * q8_row_blocks;
            for (int t = 0; t < mt; ++t) {
                const c_q8_k *q8b = rowbase + (int64_t)t * q8_row_blocks + b;
                const int8_t *q8 = q8b->qs + group * 32 + entry * 8;
                int32_t b0, b1;
                memcpy(&b0, q8 + 0, 4);
                memcpy(&b1, q8 + 4, 4);
                const int sumi = __dp4a(a0, b0, 0) + __dp4a(a1, b1, 0);
                acc[t] += wd * q8b->d * (float)sumi;
            }
        }

        for (int t = 0; t < mt; ++t) {
            float s = acc[t];
            for (int off = 16; off > 0; off >>= 1)
                s += __shfl_down_sync(0xffffffffu, s, off);
            if (lane == 0) y[(int64_t)(m0 + t) * n_cols + col] = s;
        }
    }
}

/* ---------------- IQ2_XXS (type 16) ----------------
 *
 * This format had no device path at all: wubu_cuda_quant_matvec_host fell through to the generic
 * dispatcher for type 16, which returns success while writing nothing, so a caller saw rc=0 and a buffer of
 * zeros (tools/gpu_iq_wall.c caught it). 25.5% of this model's weight bytes are IQ2_XXS, so the VRAM tier
 * cannot hold the dense weights without it.
 *
 * Decode, straight from src/dequant_iq2_xxs.c: one fp16 scale per block, a 4-bit subgroup scale in the top
 * nibble of aux32[1], a 7-bit sign index in the bits below it, and a 256-entry grid of 8 int8 values.
 */
#define IQ2X_MT 16
__global__ void iq2xxs_matvec_mrow_kernel(const uint8_t * __restrict__ w,
                                          const c_q8_k * __restrict__ xq8m,
                                          float * __restrict__ y,
                                          int64_t col_stride_bytes, int nblocks,
                                          int n_cols, int M, int q8_row_blocks)
{
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (col >= n_cols) return;

    const int group = lane >> 2;      /* ib32, 0..7 */
    const int entry = lane & 3;       /* l, 0..3 */
    const uint8_t *wcol = w + (int64_t)col * col_stride_bytes;

    for (int m0 = 0; m0 < M; m0 += IQ2X_MT) {
        const int mt = (M - m0) < IQ2X_MT ? (M - m0) : IQ2X_MT;
        float acc[IQ2X_MT];
        for (int t = 0; t < IQ2X_MT; ++t) acc[t] = 0.0f;

        for (int b = 0; b < nblocks; ++b) {
            const uint8_t *blk = wcol + (int64_t)b * 66;
            const uint16_t d_bits = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
            const uint8_t *aux = blk + 2 + 8 * group;
            uint32_t a0, a1;
            memcpy(&a0, aux, 4);
            memcpy(&a1, aux + 4, 4);
            const uint8_t gidx = ((const uint8_t *)&a0)[entry];
            const uint8_t signs = c_ksigns_iq2xs[(a1 >> (7 * entry)) & 127];
            const uint8_t *grid = (const uint8_t *)(c_iq2xxs_grid + gidx);
            int32_t g0, g1;
            memcpy(&g0, grid, 4);
            memcpy(&g1, grid + 4, 4);
            /* same predicated-negation trick as the IQ2_S kernel above */
            const uint32_t m_lo = (((uint32_t)(signs & 0x0F) * 0x00204081u) & 0x01010101u) * 0xFFu;
            const uint32_t m_hi = (((uint32_t)(signs >> 4) * 0x00204081u) & 0x01010101u) * 0xFFu;
            g0 = (int32_t)__vsub4((uint32_t)g0 ^ m_lo, m_lo);
            g1 = (int32_t)__vsub4((uint32_t)g1 ^ m_hi, m_hi);
            const float wd = f16_fast(d_bits) * (0.5f + (float)(a1 >> 28)) * 0.25f;

            const c_q8_k *rowbase = xq8m + (int64_t)m0 * q8_row_blocks;
            for (int t = 0; t < mt; ++t) {
                const c_q8_k *q8b = rowbase + (int64_t)t * q8_row_blocks + b;
                const int8_t *q8 = q8b->qs + group * 32 + entry * 8;
                int32_t b0, b1;
                memcpy(&b0, q8 + 0, 4);
                memcpy(&b1, q8 + 4, 4);
                const int sumi = __dp4a(g0, b0, 0) + __dp4a(g1, b1, 0);
                acc[t] += wd * q8b->d * (float)sumi;
            }
        }

        for (int t = 0; t < mt; ++t) {
            float s = acc[t];
            for (int off = 16; off > 0; off >>= 1)
                s += __shfl_down_sync(0xffffffffu, s, off);
            if (lane == 0) y[(int64_t)(m0 + t) * n_cols + col] = s;
        }
    }
}

WUBU_EXPORT int wubu_cuda_upload_iq2xxs(const void *grid_256_u64, const void *ksigns_128_u8)
{
    if (!grid_256_u64 || !ksigns_128_u8) return -1;
    if (cudaMemcpyToSymbol(c_iq2xxs_grid, grid_256_u64, 256 * sizeof(uint64_t)) != cudaSuccess) return -2;
    if (cudaMemcpyToSymbol(c_ksigns_iq2xs, ksigns_128_u8, 128) != cudaSuccess) return -3;
    return 0;
}

/* ---------------- Q8_0 (type 8) ----------------
 *
 * There was no Q8_0 device kernel: type 8 fell through to wubu_cuda_quant_matvec, which returned success and wrote
 * nothing (tools/gpu_q8_check.c: 0 of 512 rows non-zero). Q8_0 is k, v, attn_output and attn_rel - four of the five
 * dense attention tensors - so the resident tier could not hold any of them.
 *
 * Block is 34 bytes: fp16 d then 32 int8, and the activation is the same block type. One lane per block, eight
 * __dp4a per block, then a warp reduction over the blocks a warp covered.
 */
__global__ void q8_0_matvec_kernel(const uint8_t * __restrict__ w,
                                   const c_q8_0 * __restrict__ xq8,
                                   float * __restrict__ y,
                                   int64_t row_stride, int nblocks, int n_cols)
{
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (col >= n_cols) return;

    const uint8_t *wrow = w + (int64_t) col * row_stride;
    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        const uint8_t *blk = wrow + (int64_t) b * 34;
        const uint16_t wd_bits = (uint16_t)(blk[0] | ((uint16_t) blk[1] << 8));
        int32_t s = 0;
        for (int g = 0; g < 8; ++g) {
            int32_t wv, av;
            memcpy(&wv, blk + 2 + g * 4, 4);
            memcpy(&av, xq8[b].qs + g * 4, 4);
            s = __dp4a(wv, av, s);
        }
        /* Q8_0 stores both scales as fp16 BITS, unlike c_q8_k which holds a float. Using the raw uint16 here
         * multiplied every block by ~7.8e5 and still produced plausible-looking large numbers. */
        acc += f16_fast(wd_bits) * f16_fast(xq8[b].d) * (float) s;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[col] = acc;
}

WUBU_EXPORT int wubu_cuda_matvec_q8_0(const void *d_w, const void *d_q8, float *d_y,
                                      int K, int n_cols, int64_t col_stride_bytes)
{
    if (!d_w || !d_q8 || !d_y || K <= 0 || n_cols <= 0 || (K % 32) != 0) return -1;
    const int nblocks = K / 32;
    const int threads = 256;
    const int warps = threads / 32;
    const int blocks = (n_cols + warps - 1) / warps;
    q8_0_matvec_kernel<<<blocks, threads>>>((const uint8_t *) d_w, (const c_q8_0 *) d_q8, d_y,
                                            col_stride_bytes, nblocks, n_cols);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -2;
}

/* ---------------- resident weight tier (LP-22) ----------------
 *
 * Keyed on the HOST pointer of the mapped weights: the GGUF mapping is immutable for the life of the process, so a
 * given pointer always names the same bytes and needs no invalidation. Slots are limited and never evicted - the
 * dense tier of this model is 48 layers x 5 tensors plus the head = 241 tensors. The first value of 96 here was
 * way to be wrong. Experts are deliberately excluded by the caller: 48x128 slabs is far beyond any slot count.
 */
#define WUBU_RES_SLOTS 1536   /* was 320: the table, not the byte budget, was the binding limit - at ~1-2.4 MB per slab
                                * 320 slots sat at 320 MB while a 3 GB budget went unused, and every expert beyond the first
                                * 320 unique slabs was refused. 1536 slots lets the LRU actually evict. */
/* Declared here because this block sits above both definitions; the entry points are defined further down. */
WUBU_EXPORT int wubu_cuda_quant_matvec(const void *d_w, int type, const void *d_xq8, float *d_y,
                                       int K, int n_cols, int64_t col_stride_bytes);
WUBU_EXPORT int wubu_cuda_matvec_iq2xxs(const void *d_w, const void *d_q8, float *d_y,
                                        int K, int n_cols, int64_t col_stride_bytes, int n_tok);
WUBU_EXPORT int wubu_cuda_matvec_iq2xs(const void *, const void *, float *, int, int, int64_t);
WUBU_EXPORT int wubu_cuda_matvec_iq3xxs(const void *, const void *, float *, int, int, int64_t);
WUBU_EXPORT int wubu_cuda_matvec_iq4xs(const void *, const void *, float *, int, int, int64_t);
WUBU_EXPORT int wubu_cuda_matvec_q8_0(const void *, const void *, float *, int, int, int64_t);
static void   *g_res_d[WUBU_RES_SLOTS];
static const void *g_res_key[WUBU_RES_SLOTS];
/* LRU: the slab used to be permanent, so expert tensors filled VRAM and then every further tensor failed with
 * -2 (full) or -4 (copy failed under pressure) - measured as refused ty=16 rc=-4 and ty=22 rc=-8 with the expert
 * gate open. Routing analysis says an LRU of 48 experts/layer hits 85.7 %, so eviction is what makes the
 * expert types usable at all rather than a tuning knob. */
static unsigned long long g_res_use[WUBU_RES_SLOTS];
static unsigned long long g_res_clock = 0;
static size_t g_res_budget = 0;
static size_t g_res_used = 0;
static long g_res_evict = 0, g_res_hit = 0, g_res_miss = 0;


static size_t  g_res_bytes[WUBU_RES_SLOTS];
/* Stable identity (tensor file offset + element offset) beside the host-pointer key. The pointer key is not an
 * identity for expert slices: the engine stages a miss into a per-call buffer that the allocator hands back at the
 * same address for every expert of the same size, so two experts of one tensor collide and the second is served the
 * first one's bytes. Measured: 66 of 153 distinct src pointers were reused inside a single tensor while the device
 * returned rel error 3.7-5.3 on expert rows. */
static int64_t g_res_ident[WUBU_RES_SLOTS];
/* a caller-supplied ident wins when non-zero (expert slices); pointer key otherwise (dense tensors) */
static int     g_res_n;
/* The resident tier keeps ONE activation staging buffer, ONE output buffer and ONE slab table, all shared, and the
 * engine calls it from OpenMP threads. That race was the whole of the corrupt-expert mystery: 8 threads -> 111-114 of
 * 242 expert rows correct; 1 thread -> 242/242 exact, same kernel and same slabs. Serialise the entry. */
#include <atomic>
static std::atomic<int> g_res_lock{0};
static void res_lock(void)   { int e = 0; while (!g_res_lock.compare_exchange_weak(e, 1, std::memory_order_acquire)) e = 0; }
static void res_unlock(void) { g_res_lock.store(0, std::memory_order_release); }
struct ResGuard {
    ResGuard(void) { res_lock(); }
    ~ResGuard(void) { res_unlock(); }
};

static int     g_res_full_warned;   /* a full slab is a silent CPU fallback for every later tensor */
static void   *g_res_act, *g_res_y;
static void   *g_pin_act, *g_pin_y;      /* page-locked staging: the pageable copies were the per-call cost */

static size_t  g_res_act_cap, g_res_y_cap;

WUBU_EXPORT int wubu_cuda_resident_matvec_id(int type, int64_t ident, const void *h_w, size_t wbytes,
                                            const void *h_act, int act_bytes, float *h_y,
                                            int K, int n_cols, int64_t col_stride_bytes);

/* Back-compat: pointer-keyed callers keep the old name and pass ident = 0. */
WUBU_EXPORT int wubu_cuda_resident_matvec(int type, const void *h_w, size_t wbytes,
                                        const void *h_act, int act_bytes, float *h_y,
                                        int K, int n_cols, int64_t col_stride_bytes)
{
    return wubu_cuda_resident_matvec_id(type, 0, h_w, wbytes, h_act, act_bytes, h_y, K, n_cols, col_stride_bytes);
}

WUBU_EXPORT int wubu_cuda_resident_matvec_id(int type, int64_t ident, const void *h_w, size_t wbytes,
                                            const void *h_act, int act_bytes, float *h_y,
                                            int K, int n_cols, int64_t col_stride_bytes)
{
    ResGuard _guard;  /* serialises the shared staging and output buffers against the engine threads */
    if (!h_w || !h_act || !h_y || wbytes == 0 || act_bytes <= 0 || K <= 0 || n_cols <= 0) return -1;
    /* Only types with a kernel whose numbers have been checked against a reference may be resident. The generic
     * wubu_cuda_quant_matvec returns success and writes nothing (Q8_0 did exactly that: 0 of 512 rows non-zero), so
     * letting an unverified type through here produces zeros for that tensor and quietly wrong tokens - measured as
     * the last token of a 6-token greedy run moving from 220 to 167815 the first time this gate was missing. */
    if (type != 8 && type != 16 && type != 22 && type != 23) return -1;

    int slot = -1;
    for (int i = 0; i < g_res_n; ++i) {
        if (g_res_bytes[i] != wbytes) continue;
        if (ident != 0) { if (g_res_ident[i] == ident) { slot = i; break; } continue; }
        if (g_res_key[i] == h_w) { slot = i; break; }
    }
    if (slot >= 0) { g_res_use[slot] = ++g_res_clock; g_res_hit++; } else g_res_miss++;
    if (slot < 0) {
        if (!g_res_budget) {
            const char *bg = getenv("IN2_RES_GB");
            g_res_budget = bg && bg[0] ? (size_t) (atof(bg) * 1073741824.0) : 0;
        }
        /* Evict until there is room, oldest use first. A budget of 0 keeps the old permanent behaviour. */
        int guard = 0;
        while (g_res_budget && g_res_used + wbytes > g_res_budget && guard++ < WUBU_RES_SLOTS) {
            int victim = -1;
            unsigned long long oldest = ~0ull;
            for (int i = 0; i < g_res_n; ++i)
                if (g_res_d[i] && g_res_use[i] < oldest) { oldest = g_res_use[i]; victim = i; }
            if (victim < 0) break;
            /* the engine issues the activation DMA async on stream 0 and never syncs, so freeing a slab that a
             * pending copy or kernel still references is a use-after-free. Sync before evicting. */
            cudaDeviceSynchronize();
            cudaFree(g_res_d[victim]);
            g_res_d[victim] = NULL;
            g_res_used -= g_res_bytes[victim];
            g_res_bytes[victim] = 0;
            g_res_key[victim] = NULL;
            g_res_ident[victim] = 0;
            g_res_evict++;
            /* reuse the freed slot index */
            slot = victim;
            break;
        }
        if (slot < 0) {
            if (g_res_n >= WUBU_RES_SLOTS) {
                if (!g_res_full_warned) {
                    g_res_full_warned = 1;
                    fprintf(stderr, "resident slab FULL at %d slots - later tensors fall back to CPU\n",
                            WUBU_RES_SLOTS);
                }
                return -2;
            }
            slot = g_res_n++;
        }
        if (cudaMalloc(&g_res_d[slot], wbytes) != cudaSuccess) { g_res_d[slot] = NULL; g_res_used += 0; return -3; }
        if (cudaMemcpy(g_res_d[slot], h_w, wbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(g_res_d[slot]); g_res_d[slot] = NULL; return -4;
        }
        g_res_key[slot] = h_w;
        g_res_bytes[slot] = wbytes;
        g_res_ident[slot] = ident;
        g_res_use[slot] = ++g_res_clock;
        g_res_used += wbytes;
    }
    if ((size_t) act_bytes > g_res_act_cap) {
        if (g_res_act) { cudaFree(g_res_act); g_res_act = NULL; }
        if (cudaMalloc(&g_res_act, (size_t) act_bytes) != cudaSuccess) { g_res_act_cap = 0; return -5; }
        if (g_pin_act) { cudaFreeHost(g_pin_act); g_pin_act = NULL; }
        if (cudaHostAlloc(&g_pin_act, (size_t) act_bytes, cudaHostAllocDefault) != cudaSuccess) {
            g_pin_act = NULL; g_res_act_cap = 0; return -5;
        }
        g_res_act_cap = (size_t) act_bytes;
    }
    const size_t ybytes = (size_t) n_cols * sizeof(float);
    if (ybytes > g_res_y_cap) {
        if (g_res_y) { cudaFree(g_res_y); g_res_y = NULL; }
        if (cudaMalloc(&g_res_y, ybytes) != cudaSuccess) { g_res_y_cap = 0; return -6; }
        if (g_pin_y) { cudaFreeHost(g_pin_y); g_pin_y = NULL; }
        if (cudaHostAlloc(&g_pin_y, ybytes, cudaHostAllocDefault) != cudaSuccess) { g_pin_y = NULL; g_res_y_cap = 0; return -6; }
        g_res_y_cap = ybytes;
    }
    
    /* host -> pinned -> device: the middle hop is a plain memcpy of a few KB, the DMA is the fast pinned one */
    memcpy(g_pin_act, h_act, (size_t) act_bytes);
    if (cudaMemcpyAsync(g_res_act, g_pin_act, (size_t) act_bytes, cudaMemcpyHostToDevice, 0)
            != cudaSuccess) return -7;

    const int rc = (type == 8)
        ? wubu_cuda_matvec_q8_0(g_res_d[slot], g_res_act, (float *) g_res_y, K, n_cols, col_stride_bytes)
        : (type == 16)
        ? wubu_cuda_matvec_iq2xxs(g_res_d[slot], g_res_act, (float *) g_res_y, K, n_cols, col_stride_bytes, 1)
        : (type == 17)
        ? wubu_cuda_matvec_iq2xs(g_res_d[slot], g_res_act, (float *) g_res_y, K, n_cols, col_stride_bytes)
        : (type == 18)
        ? wubu_cuda_matvec_iq3xxs(g_res_d[slot], g_res_act, (float *) g_res_y, K, n_cols, col_stride_bytes)
        : (type == 23)
        ? wubu_cuda_matvec_iq4xs(g_res_d[slot], g_res_act, (float *) g_res_y, K, n_cols, col_stride_bytes)
        : (type == 22)
        ? wubu_cuda_matvec_iq2s_v2(g_res_d[slot], g_res_act, (float *) g_res_y, K, n_cols, col_stride_bytes)
        : wubu_cuda_quant_matvec(g_res_d[slot], type, g_res_act, (float *) g_res_y, K, n_cols, col_stride_bytes);
    if (rc != 0) return -8;
    if (cudaMemcpyAsync(g_pin_y, g_res_y, ybytes, cudaMemcpyDeviceToHost, 0) != cudaSuccess) return -9;
    if (cudaDeviceSynchronize() != cudaSuccess) return -9;
    memcpy(h_y, g_pin_y, ybytes);
    return 0;
}

WUBU_EXPORT int wubu_cuda_resident_stats(int *slots, double *mb)
{
    double m = 0.0;
    for (int i = 0; i < g_res_n; ++i) m += (double) g_res_bytes[i] / 1048576.0;
    if (slots) *slots = g_res_n;
    if (mb) *mb = m;
    return 0;
}

/* Resident-slab benchmark: upload the weights ONCE, then time only the kernel.
 *
 * Every other entry in this file pays a host->device copy per call, which at a 1.3 MB slab is the whole measurement.
 * This one answers the question the tandem design actually depends on: how fast does a slab that is already in VRAM
 * stream, which is the regime the dense tier lives in once it is resident. Returns the best of `reps` launches in
 * microseconds through *out_us, and 0 on success.
 */
WUBU_EXPORT int wubu_cuda_iq2xxs_bench_host(const void *h_w, const void *h_q8, float *h_y,
                                           int K, int n_cols, int64_t col_stride_bytes,
                                           int reps, double *out_us)
{
    if (!h_w || !h_q8 || !h_y || !out_us || K <= 0 || n_cols <= 0 || reps <= 0 || (K % QK_K) != 0) return -1;
    const size_t wb = (size_t)n_cols * (size_t)col_stride_bytes;
    const size_t xb = (size_t)(K / QK_K) * sizeof(c_q8_k);
    void *d_w = NULL, *d_xq8 = NULL;
    float *d_y = NULL;
    int rc = -1;
    if (cudaMalloc(&d_w, wb) != cudaSuccess) goto done;
    if (cudaMalloc(&d_xq8, xb) != cudaSuccess) goto done;
    if (cudaMalloc(&d_y, (size_t)n_cols * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMemcpy(d_w, h_w, wb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(d_xq8, h_q8, xb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    {
        const int nblocks = K / QK_K;
        const int threads = 256;
        const int warps = threads / 32;
        const int blocks = (n_cols + warps - 1) / warps;
        cudaEvent_t e0, e1;
        if (cudaEventCreate(&e0) != cudaSuccess) goto done;
        if (cudaEventCreate(&e1) != cudaSuccess) { cudaEventDestroy(e0); goto done; }
        double best = 1e18;
        for (int r = 0; r < reps; ++r) {
            cudaEventRecord(e0, 0);
            iq2xxs_matvec_mrow_kernel<<<blocks, threads>>>((const uint8_t *)d_w, (const c_q8_k *)d_xq8, d_y,
                                                          col_stride_bytes, nblocks, n_cols, 1, nblocks);
            cudaEventRecord(e1, 0);
            if (cudaEventSynchronize(e1) != cudaSuccess) { cudaEventDestroy(e0); cudaEventDestroy(e1); goto done; }
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            if ((double) ms * 1000.0 < best) best = (double) ms * 1000.0;
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        if (cudaMemcpy(h_y, d_y, (size_t)n_cols * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
        *out_us = best;
        rc = 0;
    }
done:
    if (d_w) cudaFree(d_w);
    if (d_xq8) cudaFree(d_xq8);
    if (d_y) cudaFree(d_y);
    return rc;
}

__global__ void iq2xxs_matvec_8col_kernel(const uint8_t * __restrict__, const c_q8_k * __restrict__,
                                          float * __restrict__, int64_t, int, int, int, int);

/* Device-resident entry: weights already on the card. n_tok rows of q8_K in, n_tok*n_cols floats out. */
WUBU_EXPORT int wubu_cuda_matvec_iq2xxs(const void *d_w, const void *d_q8, float *d_y,
                                       int K, int n_cols, int64_t col_stride_bytes, int n_tok)
{
    if (!d_w || !d_q8 || !d_y || K <= 0 || n_cols <= 0 || n_tok <= 0 || (K % QK_K) != 0) return -1;
    const int nblocks = K / QK_K;
    const int threads = 256;
    const int warps = threads / 32;
    /* Eight columns per warp pays only when the batch is real. Measured in the engine at n_tok=1: 7031 ms for 24
     * tokens against 5138 ms for the one-column kernel, because groups = n_cols/8 gives the CTA 8x fewer warps for the
     * same work. The tile wins at M>=8 (1.73x, tools/iq2x8_check.c) and loses at M=1, so it is gated on the batch. */
    if (n_tok >= 8) {
        const int groups = (n_cols + 7) / 8;
        const int blocks = (groups + warps - 1) / warps;
        iq2xxs_matvec_8col_kernel<<<blocks, threads>>>((const uint8_t *)d_w, (const c_q8_k *)d_q8, d_y,
                                                       col_stride_bytes, nblocks, n_cols, n_tok, nblocks);
        return cudaDeviceSynchronize() == cudaSuccess ? 0 : -2;
    }
    const int blocks = (n_cols + warps - 1) / warps;
    iq2xxs_matvec_mrow_kernel<<<blocks, threads>>>((const uint8_t *)d_w, (const c_q8_k *)d_q8, d_y,
                                                  col_stride_bytes, nblocks, n_cols, n_tok, nblocks);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -2;
}

/* Scratch for M quantised rows, and the host entry that drives the kernel above. */
static c_q8_k *g_q8m;
static int g_q8m_cap;
static float *g_mrow_x, *g_mrow_y;
static size_t g_mrow_xcap, g_mrow_ycap;

__global__ void iq2s_matvec_mrow_kernel(const uint8_t * __restrict__ w,
                                        const c_q8_k * __restrict__ xq8m,
                                        float * __restrict__ y, int64_t col_stride_bytes,
                                        int nblocks, int n_cols, int M, int q8_row_blocks);
__global__ void iq3s_matvec_mrow_kernel(const uint8_t * __restrict__ w,
                                        const c_q8_k * __restrict__ xq8m,
                                        float * __restrict__ y, int64_t col_stride_bytes,
                                        int nblocks, int n_cols, int M, int q8_row_blocks);

/* ---------------- fused Q/K/V over M rows ----------------
 *
 * Prefill had no fused path (wubu_gpu_qkv3 is N == 1), so every layer uploaded the normalised
 * hidden state three times and ran the Q8_K quantiser three times on the same bytes before the
 * three projections. Here the staging and the quantisation happen once and all three M-row kernels
 * read the same q8 rows. Type ids follow ggml: 21 = IQ3_S, 22 = IQ2_S. Anything else returns
 * non-zero and the caller falls back to the per-tensor path it used before. */
static int qkv_mrow_one(int type, const void *dw, const c_q8_k *q8m, float *my,
                        int64_t stride, int nblocks, int cols, int M, float *out)
{
    const int wpb = 8;
    const int blocks = (cols + wpb - 1) / wpb;
    if (type == 22)
        iq2s_matvec_mrow_kernel<<<blocks, wpb * 32>>>((const uint8_t *)dw, q8m, my, stride,
                                                     nblocks, cols, M, nblocks);
    else if (type == 21)
        iq3s_matvec_mrow_kernel<<<blocks, wpb * 32>>>((const uint8_t *)dw, q8m, my, stride,
                                                     nblocks, cols, M, nblocks);
    else
        return -2;
    if (cudaGetLastError() != cudaSuccess) return -3;
    return cudaMemcpy(out, my, (size_t)cols * (size_t)M * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -4;
}

WUBU_EXPORT int wubu_cuda_qkv_mrow(const void *d_wq, int tq, const void *d_wk, int tk,
                                   const void *d_wv, int tv, const float *h_x, int K, int M,
                                   int nq, int nk, int nv, long long sq, long long sk, long long sv,
                                   float *h_q, float *h_k, float *h_v)
{
    if (!d_wq || !d_wk || !d_wv || !h_x || !h_q || !h_k || !h_v) return -1;
    if (K <= 0 || M <= 0 || nq <= 0 || nk <= 0 || nv <= 0 || (K % QK_K) != 0) return -1;
    if ((tq != 21 && tq != 22) || (tk != 21 && tk != 22) || (tv != 21 && tv != 22)) return -1;
    const int nblocks = K / QK_K;
    c_q8_k *q8m = (c_q8_k *)pool_get("q8.mrow", (size_t)nblocks * (size_t)M * sizeof(c_q8_k));
    float *mx = (float *)pool_get("mrow.x", (size_t)K * (size_t)M * sizeof(float));
    const int ymax = nq > nk ? (nq > nv ? nq : nv) : (nk > nv ? nk : nv);
    float *my = (float *)pool_get("mrow.y", (size_t)ymax * (size_t)M * sizeof(float));
    if (!q8m || !mx || !my) return -2;
    if (cudaMemcpy(mx, h_x, (size_t)K * (size_t)M * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        return -4;
    quantize_q8k_kernel<<<dim3(nblocks, M), 32>>>(mx, q8m, K, K, nblocks);
    if (cudaGetLastError() != cudaSuccess) return -5;

    int rc = qkv_mrow_one(tq, d_wq, q8m, my, (int64_t)sq, nblocks, nq, M, h_q);
    if (rc == 0) rc = qkv_mrow_one(tk, d_wk, q8m, my, (int64_t)sk, nblocks, nk, M, h_k);
    if (rc == 0) rc = qkv_mrow_one(tv, d_wv, q8m, my, (int64_t)sv, nblocks, nv, M, h_v);
    return rc;
}

/* ---------------- IQ2_S x Q8_K through tensor cores (prefill) ----------------
 *
 * Verified in isolation against the dp4a kernel before being wired in here: uniform data to 1.6e-06
 * relative, random data 0 of 1336320 outputs outside 1% (worst abs 1.2e-03 on a 3415 maximum), and
 * 1.67x on one pass over the weights against six. The decode is the dp4a kernel's own decode - same
 * lane mapping, same grid, same sign masks, same nibbles - and only its destination changes: shared
 * memory as int8, from which the b-fragment is read.
 *
 * Two things that cost real debugging and must not be undone:
 *   1. c_q8_k is { float d; int16 bsums[16]; int8 qs[256] }, so activations for block b start at
 *      b*sizeof(c_q8_k) + 36, and sub-block g occupies qs[g*32 .. g*32+31].
 *   2. A 32-k mma chunk straddles four IQ2_S grid entries: entries 0,1 take the low scale nibble and
 *      entries 2,3 the high. The b-fragment splits exactly on that boundary, so two mma per chunk,
 *      each scaled by its own nibble, is exact. One scale per chunk is not expressible.
 *
 * The uniform test passed while the scaling was still wrong, which is why both tests exist. */
#define IQ2S_MMA_COLS (IQ2S_MMA_WPB * 8)
/* 260 bytes = 65 words per row, an odd stride: with 256 every column's tile started at the same
 * bank offset and the b-fragment reads collided. chondrule-free staging is what upstream's
 * "+1 int per row" padding is for. */
#define IQ2S_MMA_PAD  260
/* Row-halves per warp: reuse one decoded tile for 32 rows without adding warps, which is what cost
 * 25% at 1717 when the block went to 8 warps. */
#define IQ2S_MMA_RH   6
#define IQ2S_MMA_WPB  4

__global__ void iq2s_matvec_mma_kernel(const uint8_t * __restrict__ w,
                                       const c_q8_k * __restrict__ xq8m,
                                       float * __restrict__ y,
                                       int64_t col_stride_bytes, int nblocks,
                                       int n_cols, int M, int q8_row_blocks)
{
    __shared__ int8_t sW[IQ2S_MMA_COLS][IQ2S_MMA_PAD];
    __shared__ float  sLo[IQ2S_MMA_COLS][8];
    __shared__ float  sHi[IQ2S_MMA_COLS][8];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tid = threadIdx.x;
    const int gid = lane >> 2, tig = lane & 3;
    const int col0 = blockIdx.x * IQ2S_MMA_COLS;
    const int row0 = blockIdx.y * (16 * IQ2S_MMA_RH);   /* one block covers RH row-halves */
    if (row0 >= M) return;

    float accf[IQ2S_MMA_RH][4];
    for (int r = 0; r < IQ2S_MMA_RH; ++r)
        for (int q = 0; q < 4; ++q) accf[r][q] = 0.0f;

    for (int b = 0; b < nblocks; ++b) {
        {
            const int c = tid / 4;
            const int entry = tid & 3;
            const int gc = col0 + c;
            if (gc < n_cols) {
                const uint8_t *blk = w + (int64_t)gc * col_stride_bytes + (int64_t)b * 82;
                const uint16_t db = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
                const uint8_t *qs = blk + 2, *qh = blk + 66, *sc = blk + 74;
                for (int g = 0; g < 8; ++g) {
                    const uint32_t idx = (uint32_t)(qs[g * 4 + entry] |
                                         ((qh[g] << (8 - 2 * entry)) & 0x300));
                    const uint8_t *gr = (const uint8_t *)(c_iq2s_grid + idx);
                    const uint8_t signs = qs[32 + g * 4 + entry];
                    int32_t a0, a1;
                    memcpy(&a0, gr, 4);
                    memcpy(&a1, gr + 4, 4);
                    const uint32_t ml = (((uint32_t)(signs & 0x0F) * 0x00204081u) & 0x01010101u) * 0xFFu;
                    const uint32_t mh = (((uint32_t)(signs >> 4) * 0x00204081u) & 0x01010101u) * 0xFFu;
                    a0 = (int32_t)__vsub4((uint32_t)a0 ^ ml, ml);
                    a1 = (int32_t)__vsub4((uint32_t)a1 ^ mh, mh);
                    memcpy(&sW[c][g * 32 + entry * 8 + 0], &a0, 4);
                    memcpy(&sW[c][g * 32 + entry * 8 + 4], &a1, 4);
                    if (entry == 0) sLo[c][g] = 0.125f * f16_fast(db) * (float)(1 + 2 * (sc[g] & 0x0f));
                    else if (entry == 2) sHi[c][g] = 0.125f * f16_fast(db) * (float)(1 + 2 * (sc[g] >> 4));
                }
            } else {
                memset(&sW[c][0], 0, IQ2S_MMA_PAD);
                for (int g = 0; g < 8; ++g) { sLo[c][g] = 0.0f; sHi[c][g] = 0.0f; }
            }
        }
        __syncthreads();

        for (int rh = 0; rh < IQ2S_MMA_RH; ++rh) {
        const int rbase = row0 + rh * 16;
        float dq0 = 0.0f, dq1 = 0.0f;
        {
            const int r0 = rbase + gid, r1 = r0 + 8;
            if (r0 < M) dq0 = xq8m[(int64_t)r0 * q8_row_blocks + b].d;
            if (r1 < M) dq1 = xq8m[(int64_t)r1 * q8_row_blocks + b].d;
        }
        const int cl0 = warp * 8 + tig * 2;

        for (int g = 0; g < 8; ++g) {
            const int r0 = rbase + gid, r1 = r0 + 8;
            const size_t qoff = (size_t)g * 32;
            uint32_t a[4] = {0, 0, 0, 0};
            if (r0 < M) {
                const int8_t *q0 = xq8m[(int64_t)r0 * q8_row_blocks + b].qs + qoff;
                memcpy(&a[0], q0 + tig * 4, 4);
                memcpy(&a[2], q0 + tig * 4 + 16, 4);
            }
            if (r1 < M) {
                const int8_t *q1 = xq8m[(int64_t)r1 * q8_row_blocks + b].qs + qoff;
                memcpy(&a[1], q1 + tig * 4, 4);
                memcpy(&a[3], q1 + tig * 4 + 16, 4);
            }
            uint32_t b2[2] = {0, 0};
            const int cl = warp * 8 + gid;
            memcpy(&b2[0], &sW[cl][g * 32 + tig * 4], 4);
            memcpy(&b2[1], &sW[cl][g * 32 + tig * 4 + 16], 4);

            int32_t accLo[4] = {0, 0, 0, 0}, accHi[4] = {0, 0, 0, 0};
            const uint32_t aLo[4] = {a[0], a[1], 0, 0}, aHi[4] = {0, 0, a[2], a[3]};
            const uint32_t bLo[2] = {b2[0], 0}, bHi[2] = {0, b2[1]};
            mma_i8_s32(accLo, aLo, bLo);
            mma_i8_s32(accHi, aHi, bHi);
            const float lo0 = sLo[cl0][g], lo1 = sLo[cl0 + 1][g];
            const float hi0 = sHi[cl0][g], hi1 = sHi[cl0 + 1][g];
            accf[rh][0] += ((float)accLo[0] * lo0 + (float)accHi[0] * hi0) * dq0;
            accf[rh][1] += ((float)accLo[1] * lo1 + (float)accHi[1] * hi1) * dq0;
            accf[rh][2] += ((float)accLo[2] * lo0 + (float)accHi[2] * hi0) * dq1;
            accf[rh][3] += ((float)accLo[3] * lo1 + (float)accHi[3] * hi1) * dq1;
        }
        }
        __syncthreads();
    }

    const int oc = col0 + warp * 8 + tig * 2;
    for (int rh = 0; rh < IQ2S_MMA_RH; ++rh) {
        const int orow = row0 + rh * 16 + gid;
        if (orow < M) {
            if (oc < n_cols) y[(int64_t)orow * n_cols + oc] = accf[rh][0];
            if (oc + 1 < n_cols) y[(int64_t)orow * n_cols + oc + 1] = accf[rh][1];
        }
        if (orow + 8 < M) {
            if (oc < n_cols) y[(int64_t)(orow + 8) * n_cols + oc] = accf[rh][2];
            if (oc + 1 < n_cols) y[(int64_t)(orow + 8) * n_cols + oc + 1] = accf[rh][3];
        }
    }
}

WUBU_EXPORT int wubu_cuda_matvec_iq2s_mrow(const void *d_w, const float *h_x,
                                           int K, int n_cols, long long col_stride,
                                           int M, float *h_y)
{
    if (!d_w || !h_x || !h_y || K <= 0 || n_cols <= 0 || M <= 0 || (K % QK_K) != 0) return -1;
    const int nblocks = K / QK_K;
    /* Declared by role; recycled across tensors and across prompt sizes. */
    c_q8_k *q8m = (c_q8_k *)pool_get("q8.mrow", (size_t)nblocks * (size_t)M * sizeof(c_q8_k));
    float *mx = (float *)pool_get("mrow.x", (size_t)K * (size_t)M * sizeof(float));
    float *my = (float *)pool_get("mrow.y", (size_t)n_cols * (size_t)M * sizeof(float));
    if (!q8m || !mx || !my) return -2;
    if (cudaMemcpy(mx, h_x, (size_t)K * (size_t)M * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        return -4;
    for (int m = 0; m < M; ++m)
        quantize_q8k_kernel<<<nblocks, 32>>>(mx + (size_t)m * K, q8m + (size_t)m * nblocks, K, 0, 0);
    const int warps_per_block = 8;
    const int blocks = (n_cols + warps_per_block - 1) / warps_per_block;
    /* Tensor cores for prompts: one pass over the weights instead of ceil(M/16). Decode (M == 1)
     * stays on dp4a, where the four-MAC-per-lane path is already within 1.8x of llama.cpp. */
    if (M >= 16 && !getenv("G4_NO_MMA")) {
        dim3 mgrid((n_cols + IQ2S_MMA_COLS - 1) / IQ2S_MMA_COLS, (M + 16 * IQ2S_MMA_RH - 1) / (16 * IQ2S_MMA_RH));
        iq2s_matvec_mma_kernel<<<mgrid, IQ2S_MMA_WPB * 32>>>(
            (const uint8_t *)d_w, q8m, my, col_stride, nblocks, n_cols, M, nblocks);
        if (cudaGetLastError() != cudaSuccess) return -5;
        return cudaMemcpy(h_y, my, (size_t)n_cols * (size_t)M * sizeof(float),
                          cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -6;
    }
    iq2s_matvec_mrow_kernel<<<blocks, warps_per_block * 32>>>(
        (const uint8_t *)d_w, q8m, my, col_stride, nblocks, n_cols, M, nblocks);
    if (cudaGetLastError() != cudaSuccess) return -5;
    return cudaMemcpy(h_y, my, (size_t)n_cols * (size_t)M * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -6;
}


/* Host entry for the M-row path: uploads the weight bytes once, then runs the M-row kernel.
 * The gate for it (EG-130) compares against an independent dequantise-and-dot in the test
 * tool, not against the kernel's own family. */
WUBU_EXPORT int wubu_cuda_matvec_iq2s_mrow_host(const void *h_w, const float *h_x,
                                                int K, int n_cols, int M, float *h_y)
{
    if (!h_w || !h_x || !h_y || K <= 0 || n_cols <= 0 || M <= 0 || (K % QK_K) != 0) return -1;
    const int nblocks = K / QK_K;
    const size_t wbytes = (size_t)n_cols * (size_t)nblocks * 82;
    void *d_w = NULL;
    if (cudaMalloc(&d_w, wbytes) != cudaSuccess) return -2;
    int rc = -3;
    if (cudaMemcpy(d_w, h_w, wbytes, cudaMemcpyHostToDevice) == cudaSuccess)
        rc = wubu_cuda_matvec_iq2s_mrow(d_w, h_x, K, n_cols, (long long)nblocks * 82, M, h_y);
    cudaFree(d_w);
    return rc;
}



/* ---------------- M-row IQ3_S: same idea as the IQ2_S path above ----------------
 * IQ3_S owns attn_v, attn_out and ffn_down in this model - ffn_down alone is 15360x3840 -
 * so leaving it on the CPU would cap the batched prefill no matter how good the IQ2_S
 * kernel gets. Lane-per-block as in matvec_kernel, but the block is decoded once and
 * consumed by MT rows; only the q8 bytes and the row scale differ per row. */
#define IQ3S_MT 4
__global__ void iq3s_matvec_mrow_kernel(const uint8_t * __restrict__ w,
                                        const c_q8_k * __restrict__ xq8m,
                                        float * __restrict__ y,
                                        int64_t col_stride_bytes, int nblocks,
                                        int n_cols, int M, int q8_row_blocks)
{
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (col >= n_cols) return;

    const uint8_t *wcol = w + (int64_t)col * col_stride_bytes;

    for (int m0 = 0; m0 < M; m0 += IQ3S_MT) {
        const int mt = (M - m0) < IQ3S_MT ? (M - m0) : IQ3S_MT;
        float acc[IQ3S_MT];
        for (int t = 0; t < IQ3S_MT; ++t) acc[t] = 0.0f;

        for (int b = lane; b < nblocks; b += 32) {
            const c_iq3_s *xb = (const c_iq3_s *)wcol + b;
            const uint8_t *qs = xb->qs;
            const uint8_t *qh = xb->qh;
            const uint8_t *signs = xb->signs;
            int bsum[IQ3S_MT];
            for (int t = 0; t < IQ3S_MT; ++t) bsum[t] = 0;

            for (int ib32 = 0; ib32 < 8; ib32 += 2) {
                const int ls1 = 2 * (xb->scales[ib32 / 2] & 0xf) + 1;
                const int ls2 = 2 * (xb->scales[ib32 / 2] >> 4) + 1;
                for (int half = 0; half < 2; ++half) {
                    int sumi[IQ3S_MT];
                    for (int t = 0; t < IQ3S_MT; ++t) sumi[t] = 0;
                    for (int l = 0; l < 4; ++l) {
                        const uint8_t *g1 = (const uint8_t *)(c_iq3s_grid +
                            (uint32_t)(qs[2 * l + 0] | ((qh[ib32 + half] << (8 - 2 * l)) & 256)));
                        const uint8_t *g2 = (const uint8_t *)(c_iq3s_grid +
                            (uint32_t)(qs[2 * l + 1] | ((qh[ib32 + half] << (7 - 2 * l)) & 256)));
                        int32_t a0, a1;
                        memcpy(&a0, g1, 4);
                        memcpy(&a1, g2, 4);
                        /* decoded once for every row in the tile */
                        for (int j = 0; j < 4; ++j) {
                            if ((signs[l] >> j) & 1) ((int8_t *)&a0)[j] = (int8_t)(-((int8_t *)&a0)[j]);
                            if ((signs[l] >> (j + 4)) & 1) ((int8_t *)&a1)[j] = (int8_t)(-((int8_t *)&a1)[j]);
                        }
                        for (int t = 0; t < mt; ++t) {
                            const c_q8_k *q8b = xq8m + (int64_t)(m0 + t) * q8_row_blocks + b;
                            const int8_t *q8 = q8b->qs + (ib32 >> 1) * 64 + half * 32 + l * 8;
                            int32_t b0, b1;
                            memcpy(&b0, q8 + 0, 4);
                            memcpy(&b1, q8 + 4, 4);
                            sumi[t] += __dp4a(a0, b0, 0) + __dp4a(a1, b1, 0);
                        }
                    }
                    for (int t = 0; t < mt; ++t) bsum[t] += sumi[t] * (half ? ls2 : ls1);
                    /* Advance per HALF, exactly as iq3s_block_dot does: 8 qs bytes and 4 sign
                     * bytes cover one half-block. Advancing per ib32 pair instead was the bug
                     * that made the first version of this kernel produce garbage. */
                    qs += 8;
                    signs += 4;
                }
            }
            const float dx = f16_to_f32_dev(xb->d);
            for (int t = 0; t < mt; ++t) {
                const c_q8_k *q8b = xq8m + (int64_t)(m0 + t) * q8_row_blocks + b;
                acc[t] += dx * q8b->d * (float)bsum[t];
            }
        }

        for (int t = 0; t < mt; ++t) {
            float s = acc[t];
            for (int off = 16; off > 0; off >>= 1)
                s += __shfl_down_sync(0xffffffffu, s, off);
            if (lane == 0) y[(int64_t)(m0 + t) * n_cols + col] = s;
        }
    }
}

/* ---------------- IQ3_S x Q8_K through tensor cores (prefill) ----------------
 *
 * IQ3_S owns attn_v, attn_out and ffn_down - most of the prefill bytes - and was still on the dp4a
 * path after IQ2_S moved. Its structure differs from IQ2_S in a way that helps: a 32-value chunk is
 * exactly one half-block, so it carries ONE scale nibble, where IQ2_S's chunk straddled two. That
 * removes the two-mma-per-chunk trick the IQ2_S kernel needs; one mma per chunk, one scale.
 *
 * Layout per block: 8 sub-blocks of 32 values; sub-block g takes 4 grid entries l = 0..3, each
 * contributing 4 magnitudes from c_iq3s_grid plus 4 sign bits, so lane l covers values
 * [l*8 .. l*8+7] of the chunk - which is exactly the ordering the dp4a kernel already uses. */
__global__ void iq3s_matvec_mma_kernel(const uint8_t * __restrict__ w,
                                       const c_q8_k * __restrict__ xq8m,
                                       float * __restrict__ y,
                                       int64_t col_stride_bytes, int nblocks,
                                       int n_cols, int M, int q8_row_blocks)
{
    __shared__ int8_t sW[IQ2S_MMA_COLS][IQ2S_MMA_PAD];
    __shared__ float  sSc[IQ2S_MMA_COLS][8];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tid = threadIdx.x;
    const int gid = lane >> 2, tig = lane & 3;
    const int col0 = blockIdx.x * IQ2S_MMA_COLS;
    const int row0 = blockIdx.y * 16;
    if (row0 >= M) return;

    float accf[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int b = 0; b < nblocks; ++b) {
        {
            const int c = tid / 4;
            const int l = tid & 3;
            const int gc = col0 + c;
            if (gc < n_cols) {
                const c_iq3_s *xb = (const c_iq3_s *)(w + (int64_t)gc * col_stride_bytes) + b;
                const float dx = f16_fast(xb->d);
                for (int g = 0; g < 8; ++g) {
                    const uint8_t *qsb = xb->qs + g * 8;
                    const uint8_t sg = xb->signs[g * 4 + l];
                    const uint8_t *p1 = (const uint8_t *)(c_iq3s_grid +
                        (uint32_t)(qsb[2 * l + 0] | ((xb->qh[g] << (8 - 2 * l)) & 256)));
                    const uint8_t *p2 = (const uint8_t *)(c_iq3s_grid +
                        (uint32_t)(qsb[2 * l + 1] | ((xb->qh[g] << (7 - 2 * l)) & 256)));
                    int32_t a0, a1;
                    memcpy(&a0, p1, 4);
                    memcpy(&a1, p2, 4);
                    for (int j = 0; j < 4; ++j) {
                        if ((sg >> j) & 1) ((int8_t *)&a0)[j] = (int8_t)(-((int8_t *)&a0)[j]);
                        if ((sg >> (j + 4)) & 1) ((int8_t *)&a1)[j] = (int8_t)(-((int8_t *)&a1)[j]);
                    }
                    memcpy(&sW[c][g * 32 + l * 8 + 0], &a0, 4);
                    memcpy(&sW[c][g * 32 + l * 8 + 4], &a1, 4);
                    if (l == 0)
                        sSc[c][g] = dx * (float)(((g & 1) ? (2 * (xb->scales[g >> 1] >> 4))
                                                        : (2 * (xb->scales[g >> 1] & 0x0f))) + 1);
                }
            } else {
                memset(&sW[c][0], 0, IQ2S_MMA_PAD);
                for (int g = 0; g < 8; ++g) sSc[c][g] = 0.0f;
            }
        }
        __syncthreads();

        float dq0 = 0.0f, dq1 = 0.0f;
        {
            const int r0 = row0 + gid, r1 = r0 + 8;
            if (r0 < M) dq0 = xq8m[(int64_t)r0 * q8_row_blocks + b].d;
            if (r1 < M) dq1 = xq8m[(int64_t)r1 * q8_row_blocks + b].d;
        }
        const int cl0 = warp * 8 + tig * 2;

        for (int g = 0; g < 8; ++g) {
            const int r0 = row0 + gid, r1 = r0 + 8;
            uint32_t a[4] = {0, 0, 0, 0};
            if (r0 < M) {
                const int8_t *q0 = xq8m[(int64_t)r0 * q8_row_blocks + b].qs + (size_t)g * 32;
                memcpy(&a[0], q0 + tig * 4, 4);
                memcpy(&a[2], q0 + tig * 4 + 16, 4);
            }
            if (r1 < M) {
                const int8_t *q1 = xq8m[(int64_t)r1 * q8_row_blocks + b].qs + (size_t)g * 32;
                memcpy(&a[1], q1 + tig * 4, 4);
                memcpy(&a[3], q1 + tig * 4 + 16, 4);
            }
            uint32_t b2[2] = {0, 0};
            const int cl = warp * 8 + gid;
            memcpy(&b2[0], &sW[cl][g * 32 + tig * 4], 4);
            memcpy(&b2[1], &sW[cl][g * 32 + tig * 4 + 16], 4);

            int32_t acc[4] = {0, 0, 0, 0};
            mma_i8_s32(acc, a, b2);
            const float w0 = sSc[cl0][g], w1 = sSc[cl0 + 1][g];
            accf[0] += (float)acc[0] * w0 * dq0;
            accf[1] += (float)acc[1] * w1 * dq0;
            accf[2] += (float)acc[2] * w0 * dq1;
            accf[3] += (float)acc[3] * w1 * dq1;
        }
        __syncthreads();
    }

    const int oc = col0 + warp * 8 + tig * 2;
    const int orow = row0 + gid;
    if (orow < M) {
        if (oc < n_cols) y[(int64_t)orow * n_cols + oc] = accf[0];
        if (oc + 1 < n_cols) y[(int64_t)orow * n_cols + oc + 1] = accf[1];
    }
    if (orow + 8 < M) {
        if (oc < n_cols) y[(int64_t)(orow + 8) * n_cols + oc] = accf[2];
        if (oc + 1 < n_cols) y[(int64_t)(orow + 8) * n_cols + oc + 1] = accf[3];
    }
}

WUBU_EXPORT int wubu_cuda_matvec_iq3s_mrow(const void *d_w, const float *h_x, int K,
                                           int n_cols, long long col_stride, int M, float *h_y)
{
    if (!d_w || !h_x || !h_y || K <= 0 || n_cols <= 0 || M <= 0 || (K % QK_K) != 0) return -1;
    const int nblocks = K / QK_K;
    const int need = nblocks * M;
    if (g_q8m_cap < need) {
        if (g_q8m) cudaFree(g_q8m);
        g_q8m = NULL;
        if (cudaMalloc((void **)&g_q8m, (size_t)need * sizeof(c_q8_k)) != cudaSuccess) {
            g_q8m_cap = 0;
            return -2;
        }
        g_q8m_cap = need;
    }
    const size_t xneed = (size_t)K * (size_t)M * sizeof(float);
    const size_t yneed = (size_t)n_cols * (size_t)M * sizeof(float);
    if (g_mrow_xcap < xneed) {
        if (g_mrow_x) cudaFree(g_mrow_x);
        g_mrow_x = NULL;
        if (cudaMalloc((void **)&g_mrow_x, xneed) != cudaSuccess) { g_mrow_xcap = 0; return -3; }
        g_mrow_xcap = xneed;
    }
    if (g_mrow_ycap < yneed) {
        if (g_mrow_y) cudaFree(g_mrow_y);
        g_mrow_y = NULL;
        if (cudaMalloc((void **)&g_mrow_y, yneed) != cudaSuccess) { g_mrow_ycap = 0; return -3; }
        g_mrow_ycap = yneed;
    }
    if (cudaMemcpy(g_mrow_x, h_x, xneed, cudaMemcpyHostToDevice) != cudaSuccess) return -4;
    for (int m = 0; m < M; ++m)
        quantize_q8k_kernel<<<nblocks, 32>>>(g_mrow_x + (size_t)m * K,
                                             g_q8m + (size_t)m * nblocks, K, 0, 0);
    const int warps_per_block = 8;
    const int blocks = (n_cols + warps_per_block - 1) / warps_per_block;
    if (M >= 16 && !getenv("G4_NO_MMA")) {
        dim3 mgrid((n_cols + IQ2S_MMA_COLS - 1) / IQ2S_MMA_COLS, (M + 15) / 16);
        iq3s_matvec_mma_kernel<<<mgrid, IQ2S_MMA_WPB * 32>>>(
            (const uint8_t *)d_w, g_q8m, g_mrow_y, col_stride, nblocks, n_cols, M, nblocks);
        if (cudaGetLastError() != cudaSuccess) return -5;
        return cudaMemcpy(h_y, g_mrow_y, yneed, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -6;
    }
    iq3s_matvec_mrow_kernel<<<blocks, warps_per_block * 32>>>(
        (const uint8_t *)d_w, g_q8m, g_mrow_y, col_stride, nblocks, n_cols, M, nblocks);
    if (cudaGetLastError() != cudaSuccess) return -5;
    return cudaMemcpy(h_y, g_mrow_y, yneed, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -6;
}


/* ---------------- batched causal attention ----------------
 *
 * At 1717 tokens the CPU pass is ~60% of prefill: 48 layers x 1717^2/2 x 16 heads x 512 dims is
 * 1.2 TFLOP of MACs, and it is the only quadratic term left after the projections moved to the
 * M-row kernels. This is the same arithmetic, one block per (query token, q head):
 *
 *   scores  - warp per key, lane-strided dot then shuffle reduce
 *   softmax - max and exp in double, exactly as the CPU does, then 1/sum
 *   output  - one thread per output dim, sequential over keys, so the per-element accumulation
 *             order is the CPU's order and not a tree that happens to be nearby
 *
 * cfg: N rows, n_heads q heads, kv_heads with GQA grouping, head_dim, q row stride (qd),
 * kv row stride (kvd), cache size, sliding window (0 = full), and the position of each row. */
#define ATTN_MAXK 8192
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif
__global__ void attn_causal_kernel(const float * __restrict__ q, const float * __restrict__ k,
                                   const float * __restrict__ v, float * __restrict__ out,
                                   const int * __restrict__ positions,
                                   int N, int n_heads, int kv_heads, int hd, int qd, int kvd,
                                   int kv_size, int window)
{
    extern __shared__ float sc[];   /* one score per cached position */
    const int th = blockIdx.x;
    const int t = th / n_heads;
    const int hq = th % n_heads;
    if (t >= N) return;

    const int pos = positions[t];
    int kv_start = 0;
    if (window > 0 && pos >= window) kv_start = pos - window + 1;
    int kv_end = kv_size - 1;
    if (kv_end > pos) kv_end = pos;
    const int len = kv_end - kv_start + 1;
    if (len <= 0) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;
    /* GQA: consecutive q heads share a kv head. */
    const int kv_h = hq / (n_heads / kv_heads);
    const float *q_vec = q + (size_t)t * qd + (size_t)hq * hd;

    /* scores: warp per key, lane-strided over the head dim */
    for (int kp = kv_start + warp; kp <= kv_end; kp += nwarps) {
        const float *kv = k + (size_t)kp * kvd + (size_t)kv_h * hd;
        float a = 0.0f;
        for (int d = lane; d < hd; d += 32) a += q_vec[d] * kv[d];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0) sc[kp - kv_start] = a;
    }
    __syncthreads();

    /* softmax: double exp, as the CPU does, single thread (len <= 8192) */
    __shared__ float s_inv;
    if (tid == 0) {
        float max_s = -FLT_MAX;
        for (int i = 0; i < len; ++i) if (sc[i] > max_s) max_s = sc[i];
        float sum_e = 0.0f;
        for (int i = 0; i < len; ++i) {
            float e = (float)exp((double)(sc[i] - max_s));
            sc[i] = e;
            sum_e += e;
        }
        s_inv = 1.0f / (sum_e + 1e-10f);
    }
    __syncthreads();

    /* output: one thread per dim, sequential over keys in the CPU's order */
    const float inv = s_inv;
    float *o_vec = out + (size_t)t * qd + (size_t)hq * hd;
    for (int d = tid; d < hd; d += blockDim.x) {
        float val = 0.0f;
        for (int i = 0; i < len; ++i)
            val += sc[i] * v[(size_t)(kv_start + i) * kvd + (size_t)kv_h * hd + d];
        o_vec[d] = val * inv;
    }
}

WUBU_EXPORT int wubu_cuda_attn_causal(const float *h_q, const float *h_k, const float *h_v,
                                      float *h_out, const int *h_pos, int N, int n_heads,
                                      int kv_heads, int hd, int qd, int kvd, int kv_size,
                                      int window)
{
    if (!h_q || !h_k || !h_v || !h_out || !h_pos || N <= 0 || n_heads <= 0 || kv_heads <= 0) return -1;
    if (hd > 1024 || window > ATTN_MAXK) return -1;
    const size_t qn = (size_t)N * qd * sizeof(float);
    const size_t kvn = (size_t)kv_size * kvd * sizeof(float);
    const size_t on = (size_t)N * qd * sizeof(float);
    float *dq = (float *)pool_get("attn.q", qn);
    float *dk = (float *)pool_get("attn.k", kvn);
    float *dv = (float *)pool_get("attn.v", kvn);
    float *do_ = (float *)pool_get("attn.out", on);
    int *dp = (int *)pool_get("attn.pos", (size_t)N * sizeof(int));
    if (!dq || !dk || !dv || !do_ || !dp) return -2;
    if (cudaMemcpy(dq, h_q, qn, cudaMemcpyHostToDevice) != cudaSuccess) return -3;
    if (cudaMemcpy(dk, h_k, kvn, cudaMemcpyHostToDevice) != cudaSuccess) return -3;
    if (cudaMemcpy(dv, h_v, kvn, cudaMemcpyHostToDevice) != cudaSuccess) return -3;
    if (cudaMemcpy(dp, h_pos, (size_t)N * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return -3;
    const size_t shmem = (size_t)(window > 0 ? window : (kv_size < ATTN_MAXK ? kv_size : ATTN_MAXK)) *
                         sizeof(float);
    attn_causal_kernel<<<N * n_heads, 256, shmem>>>(dq, dk, dv, do_, dp, N, n_heads, kv_heads,
                                                   hd, qd, kvd, kv_size, window);
    if (cudaGetLastError() != cudaSuccess) return -4;
    return cudaMemcpy(h_out, do_, on, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -5;
}

/* Narrow matvecs are occupancy-limited (a 3072-column projection is 3072 warps for
 * 20 SMs, each still walking K), so split K four ways below this width. The FFN
 * shapes (15360 columns) stay on the warp-per-column v3 kernel. */
WUBU_EXPORT int wubu_cuda_matvec_iq2s_v3(const void *d_w, const void *d_q8, float *d_y,
                                         int K, int n_cols, long long col_stride);


WUBU_EXPORT int wubu_cuda_matvec_iq2s_v3(const void *d_w, const void *d_q8, float *d_y,
                                         int K, int n_cols, long long col_stride)
{
    if (!d_w || !d_q8 || !d_y || K <= 0 || n_cols <= 0 || (K % QK_K) != 0) return -1;
    const int nblocks = K / QK_K;
    /* Occupancy tweak for the FFN shape, tunable so a sweep does not need a rebuild per value. */
    int warps_per_block = 8;
    { const char *w = getenv("G4_MMA_WPB"); if (w) { const int v = atoi(w); if (v >= 1 && v <= 16) warps_per_block = v; } }
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
    /* 16 or fewer blocks per column (the LM head at 15, the q/k/v projections at 15): two columns
     * per warp, because the generic warp-per-column mapping leaves more than half the lanes idle at
     * this width. The FFN's 60-block columns keep the warp-per-column kernel. */
    if ((K / QK_K) <= 16 && (type == 11 || type == 22 || type == 21)) {
        const int warps = (n_cols + 1) / 2;
        if (type == 11)
            q3k_matvec_w16_kernel<<<(warps + 3) / 4, 128>>>((const uint8_t *)d_w, (const c_q8_k *)d_xq8,
                                                            d_y, col_stride_bytes, K / QK_K, n_cols);
        else if (type == 22)
            iq2s_matvec_w16_kernel<<<(warps + 3) / 4, 128>>>((const uint8_t *)d_w, (const c_q8_k *)d_xq8,
                                                            d_y, col_stride_bytes, K / QK_K, n_cols);
        else
            iq3s_matvec_w16_kernel<<<(warps + 3) / 4, 128>>>((const uint8_t *)d_w, (const c_q8_k *)d_xq8,
                                                            d_y, col_stride_bytes, K / QK_K, n_cols);
        return cudaGetLastError() == cudaSuccess ? 0 : -4;
    }
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

WUBU_EXPORT int wubu_cuda_upload_tables_iq2xs(const void *grid_512_u64)
{
    if (grid_512_u64 &&
        cudaMemcpyToSymbol(c_iq2xs_grid, grid_512_u64, 512 * sizeof(uint64_t)) != cudaSuccess)
        return -1;
    return 0;
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
        sum += 0.25f * f16_fast(d_bits) * q8b->d * (float)(sumi * (int)ls);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) y[col] = sum;
}

/* IQ2_XS x Q8_K, warp per column. Block: fp16 d, 32 uint16 grid/sign words
 * (4 per 32-element group: low 9 bits = grid index, bits 9-15 = sign mask), and 8
 * scale bytes (low nibble for entries 0-1, high for 2-3; value 1+2*scale). Lane ->
 * (group = lane>>2, entry = lane&3) covers one 256-element block with 32 lanes. */
__global__ void iq2xs_matvec_kernel(const uint8_t * __restrict__ w,
                                    const c_q8_k * __restrict__ xq8,
                                    float * __restrict__ y,
                                    int64_t col_stride_bytes, int nblocks, int n_cols)
{
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (col >= n_cols) return;

    const int g = lane >> 2;
    const int l = lane & 3;
    float sum = 0.0f;
    const uint8_t *wcol = w + (int64_t)col * col_stride_bytes;

    for (int b = 0; b < nblocks; ++b) {
        const uint8_t *blk = wcol + (int64_t)b * 74;
        const uint16_t d_bits = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        const uint8_t *scales = blk + 2 + 64;
        uint16_t q;
        memcpy(&q, blk + 2 + 2 * (4 * g + l), 2);

        const uint8_t *grid = (const uint8_t *)(c_iq2xs_grid + (q & 511));
        const uint8_t signs = c_ksigns_iq2xs[(q >> 9) & 127];
        const int8_t *q8 = xq8[b].qs + g * 32 + l * 8;

        /* The grid holds UNSIGNED magnitudes (ggml: y = db * grid value, sign from ksigns),
         * so casting those bytes to int8 flipped every magnitude >= 128. This is the only
         * type-17 consumer that was wrong: the fused FFN's type-17 path uses the same kernel
         * with n_cols=15360 and measured correct, while the composite's attn_k (n_cols=2048,
         * same K and stride) produced 9.7e+00 - 1.7e+01 absolute error on exactly the five
         * layers whose attn_k is IQ2_XS (16, 21, 22, 24, 25). Scalar rather than dp4a because
         * a 255 magnitude does not fit int8. */
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int qv = (int)q8[j];
            const int gv = (int)grid[j];
            sumi += ((signs >> j) & 1) ? -(qv * gv) : (qv * gv);
        }
        const int ls = (l < 2) ? (1 + 2 * (scales[g] & 0x0f)) : (1 + 2 * (scales[g] >> 4));
        sum += 0.125f * f16_fast(d_bits) * xq8[b].d * (float)(sumi * ls);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) y[col] = sum;
}

WUBU_EXPORT int wubu_cuda_matvec_iq2xs(const void *d_w, const void *d_q8, float *d_y,
                                       int K, int n_cols, long long col_stride)
{
    if (!d_w || !d_q8 || !d_y || K <= 0 || n_cols <= 0 || (K % QK_K) != 0) return -1;
    const int wpb = 8;
    const int blocks = (n_cols + wpb - 1) / wpb;
    iq2xs_matvec_kernel<<<blocks, wpb * 32>>>(
        (const uint8_t *)d_w, (const c_q8_k *)d_q8, d_y, col_stride, K / QK_K, n_cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
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
        const float d4d8 = f16_fast(d_bits) * q8b->d;
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
__global__ void quantize_q8k_kernel(const float * __restrict__ x, c_q8_k * __restrict__ y, int K, int xrow_stride, int yrow_stride)
{
    const int blk = blockIdx.x;
    const int row = blockIdx.y;                 /* 0 on the 1-D launches */
    const float * __restrict__ xr = x + (size_t)row * (size_t)xrow_stride;
    c_q8_k * __restrict__ yr = y + (size_t)row * (size_t)yrow_stride;
    const int lane = threadIdx.x;
    const int base = blk * QK_K;
    __shared__ float amax_sh;
    __shared__ float d_sh;

    /* Must mirror quantize_row_q8_K() exactly: max_val is the SIGNED element with
     * the largest magnitude, iscale = -127/max_val, and the stored block scale is
     * -max_val/127 (not the reciprocal). */
    float local_abs = 0.0f, local_val = 0.0f;
    for (int i = lane; i < QK_K; i += blockDim.x) {
        const float v = xr[base + i];
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
        for (int i = lane; i < QK_K; i += blockDim.x) yr[blk].qs[i] = 0;
        if (lane == 0) yr[blk].d = 0.0f;
        __syncthreads();
        return;
    }
    const float iscale = -127.0f / max_val;
    for (int i = lane; i < QK_K; i += blockDim.x) {
        /* Mirror quantize_row_q8_K() exactly:
         *   int v = (int)(iscale * x + (x >= 0 ? 0.5f : -0.5f));
         * The cast truncates toward zero. Using floorf/ceilf here rounded the other way
         * whenever iscale*x was negative (iscale carries the sign of max_val), which made
         * the device quantiser differ from the CPU one by +-1 on many elements - measured
         * as a 2.5e-02 disagreement on q/k/v between the two paths (EG-124). */
        const float xv = xr[base + i];
        float v = iscale * xv + (xv >= 0.0f ? 0.5f : -0.5f);
        int q = (int)v;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        yr[blk].qs[i] = (int8_t)q;
    }
    if (lane == 0) yr[blk].d = -max_val / 127.0f;
}

/* Gate hook: run the device Q8_K quantiser on a host array and download the bytes, so a
 * test can compare them against quantize_row_q8_K() element by element. */
WUBU_EXPORT int wubu_cuda_q8k_host(const float *h_x, void *h_out, int n)
{
    if (!h_x || !h_out || n <= 0 || (n % QK_K) != 0) return -1;
    const int nb = n / QK_K;
    float *dx = NULL;
    void *dq = NULL;
    int rc = -2;
    if (cudaMalloc(&dx, (size_t)n * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&dq, (size_t)nb * 292) != cudaSuccess) goto done;
    if (cudaMemcpy(dx, h_x, (size_t)n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    quantize_q8k_kernel<<<nb, 32>>>(dx, (c_q8_k *)dq, n, 0, 0);
    if (cudaGetLastError() != cudaSuccess) goto done;
    if (cudaMemcpy(h_out, dq, (size_t)nb * 292, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    if (dx) cudaFree(dx);
    if (dq) cudaFree(dq);
    return rc;
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
    quantize_q8k_kernel<<<K / QK_K, 32>>>(d_x, (c_q8_k *)d_q8, K, 0, 0);
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

/* Shape sweep: the engine's o-projection (3840 columns, K=3072) reaches only 18 GB/s
 * while the FFN shape reaches 82-100. Measure the kernel alone at each shape used by the
 * 12B model, so "shape" and "serialisation" stop being guesses. */
WUBU_EXPORT int wubu_cuda_bench_shapes(void)
{
    static const struct { int n_cols, K; const char *name; } shapes[] = {
        { 15360, 3840, "ffn gate/up/down" },
        {  3840, 3072, "attn out (o proj)" },
        {  3072, 3840, "attn q" },
        {  1024, 3840, "attn k/v" },
    };
    for (int s = 0; s < 4; ++s) {
        const int N = shapes[s].n_cols, K = shapes[s].K;
        const int cbytes = 82 * (K / QK_K);
        const size_t wb = (size_t)N * cbytes;
        uint8_t *h_w = (uint8_t *)malloc(wb);
        float *h_x = (float *)malloc(sizeof(float) * K);
        float *h_y = (float *)malloc(sizeof(float) * N);
        if (!h_w || !h_x || !h_y) return -1;
        memset(h_w, 0x44, wb);
        for (int i = 0; i < K; ++i) h_x[i] = (float)(i % 9) * 0.02f;
        void *d_w[3] = { NULL, NULL, NULL }, *d_x = NULL, *d_q8 = NULL;
        float *d_y = NULL;
        int ok = 1;
        for (int b = 0; b < 3; ++b) {
            if (cudaMalloc(&d_w[b], wb) != cudaSuccess) { ok = 0; break; }
            cudaMemcpy(d_w[b], h_w, wb, cudaMemcpyHostToDevice);
        }
        if (ok && cudaMalloc((void **)&d_x, sizeof(float) * K) != cudaSuccess) ok = 0;
        if (ok && cudaMalloc(&d_q8, (size_t)(K / QK_K + 1) * 292) != cudaSuccess) ok = 0;
        if (ok && cudaMalloc((void **)&d_y, sizeof(float) * N) != cudaSuccess) ok = 0;
        if (ok) {
            cudaMemcpy(d_x, (const void *)h_x, sizeof(float) * K, cudaMemcpyHostToDevice);
            quantize_q8k_kernel<<<K / QK_K, 32>>>((const float *)d_x, (c_q8_k *)d_q8, K, 0, 0);
            const int wpb = 8;
            const int blocks = (N + wpb - 1) / wpb;
            cudaDeviceSynchronize();
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            const int iters = 50;
            cudaEventRecord(e0);
            for (int it = 0; it < iters; ++it)
                iq2s_matvec_v3_kernel<<<blocks, wpb * 32>>>((const uint8_t *)d_w[it % 3],
                    (const c_q8_k *)d_q8, d_y, cbytes, K / QK_K, N);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            const double per = ms / iters;
            printf("[SHAPE] %-18s N=%5d K=%4d  %.3f ms  %.1f GB/s  (%.1f MB)\n",
                   shapes[s].name, N, K, per, (double)wb / per / 1e6, (double)wb / 1e6);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        for (int b = 0; b < 3; ++b) if (d_w[b]) cudaFree(d_w[b]);
        if (d_x) cudaFree(d_x);
        if (d_q8) cudaFree(d_q8);
        if (d_y) cudaFree(d_y);
        free(h_w); free(h_x); free(h_y);
    }
    return 0;
}

/* Host<->device transfer cost for the per-matvec vectors (12 KB up, 15 KB down), pageable
 * versus pinned. The o-projection spends ~0.21 ms per layer in upload + q8k + submit +
 * download, so the fraction that is transfer setup decides whether pinned staging is
 * worth the extra code. */
WUBU_EXPORT int wubu_cuda_bench_xfer(int iters)
{
    const int nu = 3072;   /* floats uploaded per matvec (hidden)      */
    const int nd = 3840;   /* floats downloaded per matvec (out cols)  */
    float *h_u = (float *)malloc(sizeof(float) * nu);
    float *h_d = (float *)malloc(sizeof(float) * nd);
    float *h_u_pin = NULL, *h_d_pin = NULL;
    void *d_u = NULL, *d_d = NULL;
    if (!h_u || !h_d) return -1;
    for (int i = 0; i < nu; ++i) h_u[i] = (float)i;
    if (cudaMalloc(&d_u, sizeof(float) * nu) != cudaSuccess) return -2;
    if (cudaMalloc(&d_d, sizeof(float) * nd) != cudaSuccess) return -2;
    cudaHostAlloc((void **)&h_u_pin, sizeof(float) * nu, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_d_pin, sizeof(float) * nd, cudaHostAllocDefault);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    /* pageable, one upload + one download, as the engine does it today */
    cudaEventRecord(e0);
    for (int it = 0; it < iters; ++it) {
        cudaMemcpy(d_u, h_u, sizeof(float) * nu, cudaMemcpyHostToDevice);
        cudaMemcpy(h_d, d_d, sizeof(float) * nd, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    const double pageable = ms / iters;

    /* pinned + async on one stream, single sync at the end of the pair */
    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaEventRecord(e0);
    for (int it = 0; it < iters; ++it) {
        memcpy(h_u_pin, h_u, sizeof(float) * nu);
        cudaMemcpyAsync(d_u, h_u_pin, sizeof(float) * nu, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(h_d_pin, d_d, sizeof(float) * nd, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        memcpy(h_d, h_d_pin, sizeof(float) * nd);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    const double pinned = ms / iters;

    printf("[XFER] up %d B + down %d B: pageable %.3f ms, pinned+async %.3f ms (%.1fx)\n",
           nu * 4, nd * 4, pageable, pinned, pageable / (pinned > 0 ? pinned : 1e-9));
    printf("[XFER] at 202 matvecs/token that is %.1f ms vs %.1f ms\n",
           pageable * 202.0, pinned * 202.0);

    cudaStreamDestroy(st);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFreeHost(h_u_pin); cudaFreeHost(h_d_pin);
    cudaFree(d_u); cudaFree(d_d);
    free(h_u); free(h_d);
    return 0;
}

/* Launch-shape sweep for the IQ2_S warp-per-column kernel: the same matvec at
 * different warps-per-block, to see whether the 76 GB/s streaming rate is a
 * concurrency limit (fixable by launch shape) or a per-warp read pattern limit
 * (needs a different kernel). */
static double ev_ms(cudaEvent_t a, cudaEvent_t b)
{
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    return (double)ms;
}

/* Is the short-K wide-N shape (the o-projection: K=2560, N=3840) limited by the kernel or
 * by the round trip? Three rotating buffers so streaming is not masked by L2. */
WUBU_EXPORT int wubu_cuda_bench_oproj(int iters)
{
    const int K = 2560, N = 3840;
    const int nb = K / QK_K;
    const size_t row = (size_t)nb * 82;
    enum { NBUF = 48 };   /* 48 separate 3.1 MB allocations, like the engine's tensors */
    void *d_w[NBUF];
    float *h_x = NULL, *d_x = NULL, *d_y = NULL, *h_y = NULL;
    void *d_q8 = NULL;
    if (iters <= 0) iters = 20;
    for (int i = 0; i < NBUF; ++i) d_w[i] = NULL;
    for (int i = 0; i < NBUF; ++i)
        if (cudaMalloc(&d_w[i], row * N) != cudaSuccess) return -1;
    if (cudaHostAlloc((void **)&h_x, sizeof(float) * K, cudaHostAllocDefault) != cudaSuccess) return -1;
    if (cudaHostAlloc((void **)&h_y, sizeof(float) * N, cudaHostAllocDefault) != cudaSuccess) return -1;
    if (cudaMalloc(&d_x, sizeof(float) * K) != cudaSuccess) return -1;
    if (cudaMalloc(&d_y, sizeof(float) * N) != cudaSuccess) return -1;
    if (cudaMalloc(&d_q8, (size_t)(nb + 1) * sizeof(c_q8_k)) != cudaSuccess) return -1;
    for (int i = 0; i < K; ++i) h_x[i] = (float)i * 0.001f;
    for (int i = 0; i < NBUF; ++i) {
        uint8_t *blk = (uint8_t *)malloc(row * N);
        for (size_t b = 0; b < row * N; ++b) blk[b] = (uint8_t)(b * 2654435761u);
        cudaMemcpy(d_w[i], blk, row * N, cudaMemcpyHostToDevice);
        free(blk);
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    double up = 0, q8t = 0, kern = 0, down = 0;
    for (int it = 0; it < iters; ++it) {
        cudaEventRecord(e0);
        cudaMemcpyAsync(d_x, h_x, sizeof(float) * K, cudaMemcpyHostToDevice);
        cudaEventRecord(e1); cudaEventSynchronize(e1); up += ev_ms(e0, e1);
        cudaEventRecord(e0);
        quantize_q8k_kernel<<<nb, 32>>>(d_x, (c_q8_k *)d_q8, K, 0, 0);
        cudaEventRecord(e1); cudaEventSynchronize(e1); q8t += ev_ms(e0, e1);
        cudaEventRecord(e0);
        const int wpb = 8, blocks = (N + wpb - 1) / wpb;
        iq2s_matvec_v3_kernel<<<blocks, wpb * 32>>>((const uint8_t *)d_w[it % NBUF],
                                                    (const c_q8_k *)d_q8, d_y, (int64_t)row, nb, N);
        cudaEventRecord(e1); cudaEventSynchronize(e1); kern += ev_ms(e0, e1);
        cudaEventRecord(e0);
        cudaMemcpyAsync(h_y, d_y, sizeof(float) * N, cudaMemcpyDeviceToHost);
        cudaEventRecord(e1); cudaEventSynchronize(e1); down += ev_ms(e0, e1);
    }
    const double total = up + q8t + kern + down;
    printf("[BENCH] %d iters oproj %dx%d: upload %.3f, q8k %.3f, kernel %.3f, download %.3f ms;"
           " total %.3f ms/call; kernel rate %.1f GB/s; at 48 calls/token %.1f ms\n",
           iters, K, N, up / iters, q8t / iters, kern / iters, down / iters, total / iters,
           ((double)row * N / (kern / iters)) / 1e6, total / iters * 48.0);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    for (int i = 0; i < NBUF; ++i) cudaFree(d_w[i]);
    cudaFreeHost(h_x); cudaFreeHost(h_y); cudaFree(d_x); cudaFree(d_y); cudaFree(d_q8);
    return 0;
}

WUBU_EXPORT int wubu_cuda_bench_wpb(void)
{
    const int K = 3840, N = 15360;
    const int cbytes = 82 * (K / QK_K);
    const size_t wb = (size_t)N * cbytes;
    uint8_t *h_w = (uint8_t *)malloc(wb);
    float *h_x = (float *)malloc(sizeof(float) * K);
    float *h_y = (float *)malloc(sizeof(float) * N);
    if (!h_w || !h_x || !h_y) return -1;
    memset(h_w, 0x33, wb);
    for (int i = 0; i < K; ++i) h_x[i] = (float)(i % 11) * 0.02f;

    void *d_w[3] = { NULL, NULL, NULL };
    void *d_x = NULL, *d_q8 = NULL;
    float *d_y = NULL;
    for (int b = 0; b < 3; ++b) {
        if (cudaMalloc(&d_w[b], wb) != cudaSuccess) return -2;
        cudaMemcpy(d_w[b], h_w, wb, cudaMemcpyHostToDevice);
    }
    if (cudaMalloc((void **)&d_x, sizeof(float) * K) != cudaSuccess) return -2;
    if (cudaMalloc(&d_q8, (size_t)(K / QK_K + 1) * 292) != cudaSuccess) return -2;
    if (cudaMalloc((void **)&d_y, sizeof(float) * N) != cudaSuccess) return -2;
    cudaMemcpy(d_x, (const void *)h_x, sizeof(float) * K, cudaMemcpyHostToDevice);
    quantize_q8k_kernel<<<K / QK_K, 32>>>((const float *)d_x, (c_q8_k *)d_q8, K, 0, 0);


    const int wpbs[4] = { 2, 4, 8, 16 };
    for (int c = 0; c < 4; ++c) {
        const int wpb = wpbs[c];
        const int blocks = (N + wpb - 1) / wpb;
        iq2s_matvec_v3_kernel<<<blocks, wpb * 32>>>((const uint8_t *)d_w[0], (const c_q8_k *)d_q8,
                                                    d_y, cbytes, K / QK_K, N);
        cudaDeviceSynchronize();
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        const int iters = 30;
        cudaEventRecord(e0);
        for (int it = 0; it < iters; ++it)
            iq2s_matvec_v3_kernel<<<blocks, wpb * 32>>>((const uint8_t *)d_w[it % 3], (const c_q8_k *)d_q8,
                                                        d_y, cbytes, K / QK_K, N);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double per = ms / iters;
        printf("[WPB] warps/block=%2d blocks=%5d  %.3f ms/call  %.1f GB/s\n",
               wpb, blocks, per, (double)wb / per / 1e6);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    for (int b = 0; b < 3; ++b) cudaFree(d_w[b]);
    cudaFree(d_x); cudaFree(d_q8); cudaFree(d_y);
    free(h_w); free(h_x); free(h_y);
    return 0;
}

WUBU_EXPORT int wubu_cuda_bench_stages(int iters)
{
    const int K = 3840, N = 15360;
    const int cbytes = 82 * (K / QK_K);
    const size_t wb = (size_t)N * cbytes;
    uint8_t *h_w = (uint8_t *)malloc(wb);
    float *h_x = (float *)malloc(sizeof(float) * K);
    float *h_y = (float *)malloc(sizeof(float) * N);
    if (!h_w || !h_x || !h_y) { free(h_w); free(h_x); free(h_y); return -1; }
    memset(h_w, 0x21, wb);
    for (int i = 0; i < K; ++i) h_x[i] = (float)(i % 13) * 0.01f;

    /* Three distinct weight buffers, rotated, so consecutive iterations never hit
     * L2: a single buffer re-read 20 times reports ~240 GB/s and hides the real
     * streaming rate (which is what the engine actually gets). */
    void *d_w3[3] = { NULL, NULL, NULL };
    void *d_w = NULL, *d_x = NULL, *d_q8 = NULL;
    float *d_y = NULL;
    for (int b = 0; b < 3; ++b) {
        if (cudaMalloc(&d_w3[b], wb) != cudaSuccess) return -2;
        cudaMemcpy(d_w3[b], h_w, wb, cudaMemcpyHostToDevice);
    }
    if (cudaMalloc(&d_w, wb) != cudaSuccess) { free(h_w); free(h_x); free(h_y); return -2; }
    if (cudaMalloc((void **)&d_x, sizeof(float) * K) != cudaSuccess) return -2;
    if (cudaMalloc(&d_q8, (size_t)(K / QK_K + 1) * 292) != cudaSuccess) return -2;
    if (cudaMalloc(&d_y, sizeof(float) * N) != cudaSuccess) return -2;
    cudaMemcpy(d_w, h_w, wb, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    double t_up = 0, t_q8 = 0, t_k = 0, t_dn = 0;

    for (int it = 0; it < iters; ++it) {
        cudaEventRecord(e0);
        cudaMemcpy(d_x, (const void *)h_x, sizeof(float) * K, cudaMemcpyHostToDevice);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); t_up += ms;

        cudaEventRecord(e0);
        /* Same shape the engine uses (one warp per K-block); the earlier 1-block
         * launch made q8k look 10x more expensive than it is. */
        quantize_q8k_kernel<<<K / QK_K, 32>>>((const float *)d_x, (c_q8_k *)d_q8, K, 0, 0);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1); t_q8 += ms;

        cudaEventRecord(e0);
        iq2s_matvec_v3_kernel<<<(N + 7) / 8, 8 * 32>>>((const uint8_t *)d_w3[it % 3], (const c_q8_k *)d_q8,
                                                       d_y, cbytes, K / QK_K, N);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1); t_k += ms;

        cudaEventRecord(e0);
        cudaMemcpy(h_y, d_y, sizeof(float) * N, cudaMemcpyDeviceToHost);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1); t_dn += ms;
    }
    printf("[BENCH] %d iters of %dx%d IQ2_S: upload %.3f ms, q8k %.3f ms, kernel %.3f ms, download %.3f ms\n",
           iters, N, K, t_up / iters, t_q8 / iters, t_k / iters, t_dn / iters);
    {
        const double per = (t_up + t_q8 + t_k + t_dn) / iters;
        printf("[BENCH] total %.3f ms/call; at 336 calls/token that is %.1f ms\n", per, per * 336.0);
        printf("[BENCH] effective streaming rate %.1f GB/s (%.1f MB per call)\n",
               (double)wb / (t_k / iters) / 1e6, (double)wb / 1e6);
    }

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_q8); cudaFree(d_y);
    free(h_w); free(h_x); free(h_y);
    return 0;
}

/* ---- fused FFN for decode (M == 1) -------------------------------------------
 * The CPU path costs three device round trips per layer (upload x, q8k, kernel,
 * download; twice for gate/up; then again for down) plus 7.5 ms/token of tanh GELU on
 * one core. Keeping gate, up, gelu and down on the device leaves one upload of the
 * 3840-float input and one download of the 3840-float output per layer. */

static c_q8_k *g_q8_scratch;
static int g_q8_cap;

static int ensure_q8_scratch(int k)
{
    const int need = k / QK_K + 1;
    if (g_q8_cap >= need) return 0;
    if (g_q8_scratch) cudaFree(g_q8_scratch);
    g_q8_scratch = NULL;
    if (cudaMalloc((void **)&g_q8_scratch, (size_t)need * sizeof(c_q8_k)) != cudaSuccess) {
        g_q8_cap = 0;
        return -1;
    }
    g_q8_cap = need;
    return 0;
}

__global__ void gelu_mul_kernel(const float * __restrict__ g, float * __restrict__ u, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = g[i];
    /* g4_gelu_tanh() in the CPU engine uses the ERF form (llama.cpp LLM_FFN_GELU is
     * ggml_gelu), not the tanh approximation. The device must use the same expression or
     * the two paths differ by ~1e-3 per element and can eventually flip a token. */
    u[i] = u[i] * (0.5f * x * (1.0f + erff(x * 0.7071067811865476f)));
}

/* ---- device building blocks for a device-resident layer --------------------------
 * These mirror g4_rms_norm(), the V normalisation and g4_rope_apply() exactly, so moving
 * a layer onto the device does not silently change any arithmetic. */

/* out[i] = x[i] * inv_rms * w[i]; w may be NULL (Gemma 4 normalises V with no weight). */
__global__ void rms_norm_row_kernel(const float * __restrict__ x, const float * __restrict__ w,
                                    float * __restrict__ out, int n, float eps, int has_w)
{
    __shared__ float red[32];
    const int row = blockIdx.x;
    const float *xr = x + (size_t)row * n;
    float *orow = out + (size_t)row * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        const int nw = (blockDim.x + 31) / 32;
        for (int i = 0; i < nw; ++i) tot += red[i];
        {
            /* same expression as the CPU: double sqrt + divide, not rsqrtf */
            const double msv = (double)(tot / (float)n + eps);
            red[0] = (float)(1.0 / sqrt(msv));
        }
    }
    __syncthreads();
    const float inv = red[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        orow[i] = has_w ? xr[i] * inv * w[i] : xr[i] * inv;
}

__global__ void residual_add_kernel(float * __restrict__ x, const float * __restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

/* Mirrors g4_rope_apply: same theta_scale, same freqs guard, Q and K loops kept separate. */
__global__ void rope_kernel(float * __restrict__ q, float * __restrict__ k,
                            float pos, int d_head, int q_stride, int k_stride,
                            int n_rot, float theta_scale, const float * __restrict__ freqs,
                            int n_q_heads, int n_k_heads)
{
    const int half = n_rot / 2;
    /* The grid is dim3(1, n_q_heads, n_k_heads), so a Q head must be rotated once, not
     * once per (y,z) block: guard on z == 0. (K takes the symmetric guard on y == 0 below.)
     * Without these two guards the same head is rotated n_k_heads or n_q_heads times, which
     * measures as a max-abs error of ~12 on q while k and v stay exact. */
    if (blockIdx.z != 0) { }
    else for (int h = blockIdx.y; h < n_q_heads; h += gridDim.y) {
        float *qv = q + (size_t)blockIdx.x * q_stride + (size_t)h * d_head;
        for (int j = threadIdx.x; j < half; j += blockDim.x) {
            const float ff = freqs ? freqs[j] : 1.0f;
            if (ff > 1e20f) continue;
            /* same construction as the CPU: repeated multiply and one shared sin/cos */
            float tf = 1.0f;
            for (int kk = 0; kk < j; kk++) tf *= theta_scale;
            const float theta = pos * tf / ff;
            float st, ct;
            wubu_sincosf(theta, &st, &ct);
            const float q0 = qv[j], q1 = qv[j + half];
            qv[j] = q0 * ct - q1 * st;
            qv[j + half] = q0 * st + q1 * ct;
        }
    }
    /* K must be rotated ONCE per head, not once per (q head x k head) block: the grid is
     * dim3(1, n_q_heads, n_k_heads), so without this guard every block with the same z
     * re-rotated the same K head - measured as a 1.4 max-abs error on k. */
    if (!k || blockIdx.y != 0) return;
    for (int h = blockIdx.z; h < n_k_heads; h += gridDim.z) {
        float *kv = k + (size_t)blockIdx.x * k_stride + (size_t)h * d_head;
        for (int j = threadIdx.x; j < half; j += blockDim.x) {
            const float ff = freqs ? freqs[j] : 1.0f;
            if (ff > 1e20f) continue;
            /* same construction as the CPU: repeated multiply and one shared sin/cos */
            float tf = 1.0f;
            for (int kk = 0; kk < j; kk++) tf *= theta_scale;
            const float theta = pos * tf / ff;
            float st, ct;
            wubu_sincosf(theta, &st, &ct);
            const float k0 = kv[j], k1 = kv[j + half];
            kv[j] = k0 * ct - k1 * st;
            kv[j + half] = k0 * st + k1 * ct;
        }
    }
}

/* Decode attention for ONE query position, mirroring g4_sliding_attn_cpu: no kq scale
 * (Gemma 4 sets f_attention_scale = 1.0), softmax over [kv_start, kv_start+kv_len), GQA
 * mapping kv_h = hq / (n_heads / n_kv_heads). One block per head; scores in dynamic
 * shared memory (kv_len floats). */
__global__ void attn_decode_kernel(const float * __restrict__ q,
                                   const float * __restrict__ kc,
                                   const float * __restrict__ vc,
                                   float * __restrict__ out,
                                   int n_heads, int n_kv_heads, int hd,
                                   int kv_start, int kv_len, int kvd)
{
    /* Arithmetic deliberately mirrors g4_dotf and the CPU softmax in the engine, so the
     * attention row is bit-identical rather than merely close. g4_dotf is two 8-lane FMA
     * accumulators combined by add, then add(lo,hi), then two hadd steps; the softmax sum is
     * sequential over cached positions; the value loop is (e * inv) * v accumulated in kp
     * order. Measured with G4_ATTN_ADIFF: before this the row differed by 4.3e-07 relative,
     * which is enough to flip a token once 40 layers of it accumulate. */
    extern __shared__ float sh[];
    const int hq = blockIdx.x;
    const int kv_h = hq / (n_heads / n_kv_heads);
    const float *qvec = q + (size_t)hq * hd;
    const float *K = kc + (size_t)kv_h * hd;
    const float *V = vc + (size_t)kv_h * hd;

    float lmax = -3.4e38f;
    for (int kp = threadIdx.x; kp < kv_len; kp += blockDim.x) {
        const float *kvp = K + (size_t)(kv_start + kp) * kvd;
        float a[16];
        for (int j = 0; j < 16; ++j) a[j] = 0.0f;
        const int full = hd & ~15;
        for (int d = 0; d < full; d += 16) {
            for (int j = 0; j < 16; ++j)
                a[j] = __fmaf_rn(qvec[d + j], kvp[d + j], a[j]);
        }
        float l[8];
        for (int j = 0; j < 8; ++j) l[j] = a[j] + a[j + 8];
        float sc = ((l[0] + l[4]) + (l[1] + l[5])) + ((l[2] + l[6]) + (l[3] + l[7]));
        for (int d = full; d < hd; ++d) sc += qvec[d] * kvp[d];
        sh[kp] = sc;
        if (sc > lmax) lmax = sc;
    }
    __shared__ float red[32];
    for (int off = 16; off > 0; off >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = lmax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -3.4e38f;
        const int nw = (blockDim.x + 31) / 32;
        for (int i = 0; i < nw; ++i) m = fmaxf(m, red[i]);
        red[0] = m;
    }
    __syncthreads();
    const float max_s = red[0];

    for (int kp = threadIdx.x; kp < kv_len; kp += blockDim.x)
        sh[kp] = (float)exp((double)(sh[kp] - max_s));
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;                      /* sequential, like the CPU */
        for (int kp = 0; kp < kv_len; ++kp) s += sh[kp];
        red[0] = 1.0f / (s + 1e-10f);
    }
    __syncthreads();
    const float inv = red[0];

    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (int kp = 0; kp < kv_len; ++kp) {
            const float w = __fmul_rn(sh[kp], inv);
            acc = __fmaf_rn(w, V[(size_t)(kv_start + kp) * kvd + d], acc);
        }
        out[(size_t)hq * hd + d] = acc;
    }
}

WUBU_EXPORT int wubu_cuda_attn_dev(const float *d_q, const float *d_kc, const float *d_vc,
                                   float *d_out, int n_heads, int n_kv_heads, int hd,
                                   int kv_start, int kv_len, int kvd)
{
    if (!d_q || !d_kc || !d_vc || !d_out || n_heads <= 0 || hd <= 0 || kv_len <= 0) return -1;
    if ((size_t)kv_len * sizeof(float) > 48 * 1024) return -2;   /* shared-mem ceiling */
    attn_decode_kernel<<<n_heads, 128, (size_t)kv_len * sizeof(float)>>>(
        d_q, d_kc, d_vc, d_out, n_heads, n_kv_heads, hd, kv_start, kv_len, kvd);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

/* Append one token's k/v into the device KV cache (kvd floats each). */
__global__ void kv_append_kernel(float * __restrict__ kc, float * __restrict__ vc,
                                 const float * __restrict__ ksrc, const float * __restrict__ vsrc,
                                 int pos, int kvd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kvd) return;
    kc[(size_t)pos * kvd + i] = ksrc[i];
    vc[(size_t)pos * kvd + i] = vsrc[i];
}

WUBU_EXPORT int wubu_cuda_kv_append_dev(float *d_kc, float *d_vc, const float *d_ksrc,
                                        const float *d_vsrc, int pos, int kvd)
{
    if (!d_kc || !d_vc || !d_ksrc || !d_vsrc || pos < 0 || kvd <= 0) return -1;
    kv_append_kernel<<<(kvd + 255) / 256, 256>>>(d_kc, d_vc, d_ksrc, d_vsrc, pos, kvd);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

/* Host wrapper for the KV round trip: prime n rows, append one, download, so the
 * prime+append mechanics can be checked independently of the engine. */
WUBU_EXPORT int wubu_cuda_kv_roundtrip_host(const void *h_k, const void *h_v, int n,
                                            const void *h_ka, const void *h_va, int pos,
                                            int kvd, int cap, float *out_k, float *out_v)
{
    void *dk = NULL, *dv = NULL, *dka = NULL, *dva = NULL;
    int rc = -1;
    const size_t rowb = (size_t)kvd * sizeof(float);
    if (n < 0 || pos < 0 || pos >= cap || kvd <= 0) return -1;
    if (cudaMalloc(&dk, rowb * (size_t)cap) != cudaSuccess) goto done;
    if (cudaMalloc(&dv, rowb * (size_t)cap) != cudaSuccess) goto done;
    if (n > 0) {
        if (cudaMalloc(&dka, rowb * (size_t)n) != cudaSuccess) goto done;
        if (cudaMalloc(&dva, rowb * (size_t)n) != cudaSuccess) goto done;
        if (cudaMemcpy(dka, h_k, rowb * (size_t)n, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
        if (cudaMemcpy(dva, h_v, rowb * (size_t)n, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
        if (cudaMemcpy(dk, dka, rowb * (size_t)n, cudaMemcpyDeviceToDevice) != cudaSuccess) goto done;
        if (cudaMemcpy(dv, dva, rowb * (size_t)n, cudaMemcpyDeviceToDevice) != cudaSuccess) goto done;
    }
    {
        void *dkt = NULL, *dvt = NULL;
        if (cudaMalloc(&dkt, rowb) != cudaSuccess) goto done;
        if (cudaMalloc(&dvt, rowb) != cudaSuccess) goto done;
        if (cudaMemcpy(dkt, h_ka, rowb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
        if (cudaMemcpy(dvt, h_va, rowb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
        if (wubu_cuda_kv_append_dev((float *)dk, (float *)dv, (const float *)dkt,
                                    (const float *)dvt, pos, kvd) != 0) { cudaFree(dkt); cudaFree(dvt); goto done; }
        cudaFree(dkt); cudaFree(dvt);
    }
    if (cudaMemcpy(out_k, dk, rowb * (size_t)cap, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    if (cudaMemcpy(out_v, dv, rowb * (size_t)cap, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    if (dk) cudaFree(dk);
    if (dv) cudaFree(dv);
    if (dka) cudaFree(dka);
    if (dva) cudaFree(dva);
    return rc;
}

/* Host wrappers for the gate. */
WUBU_EXPORT int wubu_cuda_attn_host(const void *h_q, const void *h_kc, const void *h_vc,
                                    float *h_out, int n_heads, int n_kv_heads, int hd,
                                    int kv_start, int kv_len, int kvd, int kcap)
{
    void *dq = NULL, *dkc = NULL, *dvc = NULL, *dout = NULL;
    int rc = -1;
    const size_t qb = (size_t)n_heads * hd * sizeof(float);
    const size_t kvb = (size_t)kcap * kvd * sizeof(float);
    if (cudaMalloc(&dq, qb) != cudaSuccess) goto done;
    if (cudaMalloc(&dkc, kvb) != cudaSuccess) goto done;
    if (cudaMalloc(&dvc, kvb) != cudaSuccess) goto done;
    if (cudaMalloc(&dout, qb) != cudaSuccess) goto done;
    if (cudaMemcpy(dq, h_q, qb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(dkc, h_kc, kvb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(dvc, h_vc, kvb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (wubu_cuda_attn_dev((const float *)dq, (const float *)dkc, (const float *)dvc,
                           (float *)dout, n_heads, n_kv_heads, hd, kv_start, kv_len, kvd) != 0) goto done;
    if (cudaMemcpy(h_out, dout, qb, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    if (dq) cudaFree(dq);
    if (dkc) cudaFree(dkc);
    if (dvc) cudaFree(dvc);
    if (dout) cudaFree(dout);
    return rc;
}

WUBU_EXPORT int wubu_cuda_rms_norm_dev(const float *d_x, const float *d_w, float *d_out,
                                       int rows, int n, float eps, int has_w);
WUBU_EXPORT int wubu_cuda_residual_add_dev(float *d_x, const float *d_y, int n);
WUBU_EXPORT int wubu_cuda_rope_dev(float *d_q, float *d_k, float pos, int d_head,
                                   int q_stride, int k_stride, int n_rot, float theta_scale,
                                   const float *d_freqs, int n_q_heads, int n_k_heads);

/* Host wrappers for the gate: these three kernels are compared against the CPU
 * implementations (g4_rms_norm, the V normalisation, g4_rope_apply) element by element. */
WUBU_EXPORT int wubu_cuda_rms_norm_host(const void *h_x, const void *h_w, float *h_out,
                                        int rows, int n, float eps, int has_w)
{
    void *dx = NULL, *dw = NULL, *dout = NULL;
    int rc = -1;
    const size_t nb = (size_t)rows * n * sizeof(float);
    if (cudaMalloc(&dx, nb) != cudaSuccess) goto done;
    if (cudaMalloc(&dout, nb) != cudaSuccess) goto done;
    if (has_w) {
        if (cudaMalloc(&dw, (size_t)n * sizeof(float)) != cudaSuccess) goto done;
        if (cudaMemcpy(dw, h_w, (size_t)n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    }
    if (cudaMemcpy(dx, h_x, nb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (wubu_cuda_rms_norm_dev((const float *)dx, (const float *)dw, (float *)dout,
                               rows, n, eps, has_w) != 0) goto done;
    if (cudaMemcpy(h_out, dout, nb, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    if (dx) cudaFree(dx);
    if (dw) cudaFree(dw);
    if (dout) cudaFree(dout);
    return rc;
}

WUBU_EXPORT int wubu_cuda_residual_add_host(const void *h_x, const void *h_y, float *h_out, int n)
{
    void *dx = NULL, *dy = NULL;
    int rc = -1;
    const size_t nb = (size_t)n * sizeof(float);
    if (cudaMalloc(&dx, nb) != cudaSuccess) goto done;
    if (cudaMalloc(&dy, nb) != cudaSuccess) goto done;
    if (cudaMemcpy(dx, h_x, nb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(dy, h_y, nb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (wubu_cuda_residual_add_dev((float *)dx, (const float *)dy, n) != 0) goto done;
    if (cudaMemcpy(h_out, dx, nb, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    if (dx) cudaFree(dx);
    if (dy) cudaFree(dy);
    return rc;
}

WUBU_EXPORT int wubu_cuda_rope_host(const void *h_q, const void *h_k, float *h_qo, float *h_ko,
                                    float pos, int d_head, int q_stride, int k_stride,
                                    int n_rot, float theta_scale, const void *h_freqs,
                                    int n_heads, int n_k_heads, int has_freqs)
{
    void *dq = NULL, *dk = NULL, *df = NULL;
    int rc = -1;
    const size_t qb = (size_t)q_stride * sizeof(float);
    const size_t kb = (size_t)k_stride * sizeof(float);
    if (cudaMalloc(&dq, qb) != cudaSuccess) goto done;
    if (h_k && cudaMalloc(&dk, kb) != cudaSuccess) goto done;
    if (cudaMemcpy(dq, h_q, qb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (h_k && cudaMemcpy(dk, h_k, kb, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (has_freqs) {
        if (cudaMalloc(&df, (size_t)(n_rot / 2) * sizeof(float)) != cudaSuccess) goto done;
        if (cudaMemcpy(df, h_freqs, (size_t)(n_rot / 2) * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    }
    if (wubu_cuda_rope_dev((float *)dq, h_k ? (float *)dk : NULL, pos, d_head, q_stride, k_stride,
                           n_rot, theta_scale, (const float *)df, n_heads, n_k_heads) != 0) goto done;
    if (cudaMemcpy(h_qo, dq, qb, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    if (h_k && h_ko && cudaMemcpy(h_ko, dk, kb, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    if (dq) cudaFree(dq);
    if (dk) cudaFree(dk);
    if (df) cudaFree(df);
    return rc;
}

WUBU_EXPORT int wubu_cuda_rms_norm_dev(const float *d_x, const float *d_w, float *d_out,
                                       int rows, int n, float eps, int has_w)
{
    if (!d_x || !d_out || rows <= 0 || n <= 0) return -1;
    rms_norm_row_kernel<<<rows, 256>>>(d_x, d_w, d_out, n, eps, has_w);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

WUBU_EXPORT int wubu_cuda_residual_add_dev(float *d_x, const float *d_y, int n)
{
    if (!d_x || !d_y || n <= 0) return -1;
    residual_add_kernel<<<(n + 255) / 256, 256>>>(d_x, d_y, n);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

WUBU_EXPORT int wubu_cuda_rope_dev(float *d_q, float *d_k, float pos, int d_head,
                                   int q_stride, int k_stride, int n_rot, float theta_scale,
                                   const float *d_freqs, int n_q_heads, int n_k_heads)
{
    if (!d_q || d_head <= 0 || n_rot <= 0) return -1;
    rope_kernel<<<dim3(1, n_q_heads, n_k_heads > 0 ? n_k_heads : 1), 128>>>(
        d_q, d_k, pos, d_head, q_stride, k_stride, n_rot, theta_scale, d_freqs,
        n_q_heads, n_k_heads);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

/* One place that maps a ggml type to a kernel. Three separate hand-written if-chains used to do
 * this, each listing 22/18/23 and falling through to the generic kernel for everything else -
 * which does not implement 17 (IQ2_XS) and answers with a zero buffer and rc=0. That is why the
 * five layers with an IQ2_XS attn_k attended on an all-zero K: no error, no fallback, max|k|=0. */
static int mv_any(const void *d_w, int type, const void *d_q8, float *d_y,
                  int K, int n_cols, long long col_stride)
{
    if (type == 22) return wubu_cuda_matvec_iq2s_v3(d_w, d_q8, d_y, K, n_cols, col_stride);
    if (type == 18) return wubu_cuda_matvec_iq3xxs(d_w, d_q8, d_y, K, n_cols, col_stride);
    if (type == 23) return wubu_cuda_matvec_iq4xs(d_w, d_q8, d_y, K, n_cols, col_stride);
    if (type == 17) return wubu_cuda_matvec_iq2xs(d_w, d_q8, d_y, K, n_cols, col_stride);
    if (type == 11 && (K / QK_K) <= 16) {
        /* LM head shape: 15 blocks per column, so two columns fit one warp. */
        const int nblocks = K / QK_K;
        const int warps = (n_cols + 1) / 2;
        q3k_matvec_w16_kernel<<<(warps + 3) / 4, 128>>>((const uint8_t *)d_w, (const c_q8_k *)d_q8,
                                                       d_y, col_stride, nblocks, n_cols);
        return cudaGetLastError() == cudaSuccess ? 0 : -3;
    }
    return wubu_cuda_quant_matvec(d_w, type, d_q8, d_y, K, n_cols, col_stride);
}

/* q8k + matvec with everything already on the device (no download). */
static int matvec_devbuf(const void *d_w, int type, const float *d_x, float *d_y,
                         int K, int n_cols, long long stride)
{
    if (ensure_q8_scratch(K) != 0) return -1;
    quantize_q8k_kernel<<<K / QK_K, 32>>>(d_x, g_q8_scratch, K, 0, 0);
    return mv_any(d_w, type, g_q8_scratch, d_y, K, n_cols, stride);
}

/* q, k and v share one input, so run all three against a single staged Q8_K row and let
 * the caller download q|k|v in one copy. Removes two uploads, two quantisations and two
 * synchronising downloads per layer; the three kernels still run one after another. */
/* One column-strided matvec through whichever kernel owns the type. Used by the fused
 * q/k/v path so the three projections can carry different quant types (on the 12B model
 * attn_q is IQ2_S while attn_k is IQ3_S, which is why the same-type fused path only
 * covered part of the layers). */
/* One place that maps a ggml type to a kernel. The composite used to spell its own
 * if-chain for q/k/v, which listed 22/18/23 and fell through to the generic kernel for
 * everything else - including 17 (IQ2_XS), which the generic kernel does not implement and
 * answers with a zero buffer and rc=0. That is why exactly the five layers whose attn_k is
 * IQ2_XS attended on an all-zero K: no error, no fallback, max|k| = 0.0. */
WUBU_EXPORT int wubu_cuda_qkv_dev3(const void *d_wq, const void *d_wk, const void *d_wv,
                                   int tq, int tk, int tv,
                                   const float *d_x, float *d_q, float *d_k, float *d_v,
                                   int K, int qd, int kvd,
                                   long long sq, long long sk, long long sv);

WUBU_EXPORT int wubu_cuda_qkv_dev(const void *d_wq, const void *d_wk, const void *d_wv, int type,
                                  const float *d_x, float *d_q, float *d_k, float *d_v,
                                  int K, int qd, int kvd,
                                  long long sq, long long sk, long long sv)
{
    return wubu_cuda_qkv_dev3(d_wq, d_wk, d_wv, type, type, type, d_x, d_q, d_k, d_v,
                              K, qd, kvd, sq, sk, sv);
}

WUBU_EXPORT int wubu_cuda_qkv_dev3(const void *d_wq, const void *d_wk, const void *d_wv,
                                   int tq, int tk, int tv,
                                   const float *d_x, float *d_q, float *d_k, float *d_v,
                                   int K, int qd, int kvd,
                                   long long sq, long long sk, long long sv)
{
    if (!d_wq || !d_wk || !d_wv || !d_x || !d_q || !d_k || !d_v) return -1;
    if ((K % QK_K) != 0) return -1;
    if (ensure_q8_scratch(K) != 0) return -2;
    quantize_q8k_kernel<<<K / QK_K, 32>>>(d_x, (c_q8_k *)g_q8_scratch, K, 0, 0);
    if (mv_any(d_wq, tq, g_q8_scratch, d_q, K, qd, sq) != 0) return -3;
    if (mv_any(d_wk, tk, g_q8_scratch, d_k, K, kvd, sk) != 0) return -4;
    if (mv_any(d_wv, tv, g_q8_scratch, d_v, K, kvd, sv) != 0) return -5;
    return cudaGetLastError() == cudaSuccess ? 0 : -6;
}

/* Host wrapper so the gate can compare the fused qkv path against the CPU ports. */
WUBU_EXPORT int wubu_cuda_qkv_host(const void *h_wq, const void *h_wk, const void *h_wv, int type,
                                   const void *h_x, float *h_q, float *h_k, float *h_v,
                                   int K, int qd, int kvd,
                                   long long sq, long long sk, long long sv)
{
    void *dwq = NULL, *dwk = NULL, *dwv = NULL, *dx = NULL;
    float *dq = NULL, *dk = NULL, *dv = NULL;
    int rc = -1;
    if (cudaMalloc(&dwq, (size_t)qd * sq) != cudaSuccess) goto done;
    if (cudaMalloc(&dwk, (size_t)kvd * sk) != cudaSuccess) goto done;
    if (cudaMalloc(&dwv, (size_t)kvd * sv) != cudaSuccess) goto done;
    if (cudaMalloc(&dx, sizeof(float) * K) != cudaSuccess) goto done;
    if (cudaMalloc((void **)&dq, sizeof(float) * qd) != cudaSuccess) goto done;
    if (cudaMalloc((void **)&dk, sizeof(float) * kvd) != cudaSuccess) goto done;
    if (cudaMalloc((void **)&dv, sizeof(float) * kvd) != cudaSuccess) goto done;
    if (cudaMemcpy(dwq, h_wq, (size_t)qd * sq, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(dwk, h_wk, (size_t)kvd * sk, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(dwv, h_wv, (size_t)kvd * sv, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(dx, h_x, sizeof(float) * K, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (wubu_cuda_qkv_dev(dwq, dwk, dwv, type, (const float *)dx, dq, dk, dv, K, qd, kvd, sq, sk, sv) != 0) goto done;
    if (cudaMemcpy(h_q, dq, sizeof(float) * qd, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    if (cudaMemcpy(h_k, dk, sizeof(float) * kvd, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    if (cudaMemcpy(h_v, dv, sizeof(float) * kvd, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    if (dwq) cudaFree(dwq);
    if (dwk) cudaFree(dwk);
    if (dwv) cudaFree(dwv);
    if (dx) cudaFree(dx);
    if (dq) cudaFree(dq);
    if (dk) cudaFree(dk);
    if (dv) cudaFree(dv);
    return rc;
}

/* ---- composite attention half-layer, gated standalone ---------------------------
 * q,k,v -> per-head RMS norms -> RoPE -> KV append -> attention -> o-projection in one
 * submission. Compared against a CPU sequence of the same steps by EG-124 before anything
 * in the engine is allowed to call it. */
WUBU_EXPORT int wubu_cuda_attn_half_dev(
        const void *d_wq, const void *d_wk, const void *d_wv, const void *d_wo,
        int tq, int tk, int tv, int to,
        const float *d_x, float *d_q, float *d_k, float *d_v,
        float *d_kc, float *d_vc, float *d_attn, float *d_out,
        const float *d_qn, const float *d_kn,
        int hidden, int qd, int kvd, int hd, int n_heads, int n_kv_heads,
        int pos, int kv_start, int kv_len,
        long long sq, long long sk, long long sv, long long so,
        float pos_f, int n_rot, float theta_scale, const float *d_freqs)
{
    if (!d_wq || !d_wk || !d_wv || !d_wo || !d_x || !d_out) return -1;
    if (ensure_q8_scratch(hidden) != 0) return -2;

    quantize_q8k_kernel<<<hidden / QK_K, 32>>>(d_x, g_q8_scratch, hidden, 0, 0);
    int rc;
    if (mv_any(d_wq, tq, g_q8_scratch, d_q, hidden, qd, sq) != 0) return -3;
    if (mv_any(d_wk, tk, g_q8_scratch, d_k, hidden, kvd, sk) != 0) return -4;
    if (mv_any(d_wv, tv, g_q8_scratch, d_v, hidden, kvd, sv) != 0) return -5;

    rms_norm_row_kernel<<<n_heads, 256>>>(d_q, d_qn, d_q, hd, 1e-6f, 1);
    rms_norm_row_kernel<<<n_kv_heads, 256>>>(d_k, d_kn, d_k, hd, 1e-6f, 1);
    rms_norm_row_kernel<<<n_kv_heads, 256>>>(d_v, NULL, d_v, hd, 1e-6f, 0);

    rope_kernel<<<dim3(1, n_heads, n_kv_heads), 128>>>(
        d_q, d_k, pos_f, hd, qd, kvd, n_rot, theta_scale, d_freqs, n_heads, n_kv_heads);

    kv_append_kernel<<<(kvd + 255) / 256, 256>>>(d_kc, d_vc, d_k, d_v, pos, kvd);

    if (wubu_cuda_attn_dev(d_q, d_kc, d_vc, d_attn, n_heads, n_kv_heads, hd,
                           kv_start, kv_len, kvd) != 0) return -6;

    if (ensure_q8_scratch(qd) != 0) return -2;
    quantize_q8k_kernel<<<qd / QK_K, 32>>>(d_attn, g_q8_scratch, qd, 0, 0);
    if (mv_any(d_wo, to, g_q8_scratch, d_out, qd, hidden, so) != 0) return -7;
    return cudaGetLastError() == cudaSuccess ? 0 : -8;
}


/* Attention stage only: q,k,v -> norms -> RoPE -> KV append -> attention, left in d_attn.
 * Used by the G4_ATTN_ADIFF instrument to compare the device attention row against the CPU
 * one at the same position, and by the (removed) CPU-quantisation experiment. */
WUBU_EXPORT int wubu_cuda_attn_stage(
        const void *d_wq, const void *d_wk, const void *d_wv,
        int tq, int tk, int tv,
        const float *d_x, float *d_q, float *d_k, float *d_v,
        float *d_kc, float *d_vc, float *d_attn,
        const float *d_qn, const float *d_kn,
        int hidden, int qd, int kvd, int hd, int n_heads, int n_kv_heads,
        int pos, int kv_start, int kv_len,
        long long sq, long long sk, long long sv,
        float pos_f, int n_rot, float theta_scale, const float *d_freqs)
{
    if (!d_wq || !d_wk || !d_wv || !d_x) return -1;
    if (ensure_q8_scratch(hidden) != 0) return -2;

    quantize_q8k_kernel<<<hidden / QK_K, 32>>>(d_x, g_q8_scratch, hidden, 0, 0);
    int rc;
    if (mv_any(d_wq, tq, g_q8_scratch, d_q, hidden, qd, sq) != 0) return -3;
    if (mv_any(d_wk, tk, g_q8_scratch, d_k, hidden, kvd, sk) != 0) return -4;
    if (mv_any(d_wv, tv, g_q8_scratch, d_v, hidden, kvd, sv) != 0) return -5;

    rms_norm_row_kernel<<<n_heads, 256>>>(d_q, d_qn, d_q, hd, 1e-6f, 1);
    rms_norm_row_kernel<<<n_kv_heads, 256>>>(d_k, d_kn, d_k, hd, 1e-6f, 1);
    rms_norm_row_kernel<<<n_kv_heads, 256>>>(d_v, NULL, d_v, hd, 1e-6f, 0);

    rope_kernel<<<dim3(1, n_heads, n_kv_heads), 128>>>(
        d_q, d_k, pos_f, hd, qd, kvd, n_rot, theta_scale, d_freqs, n_heads, n_kv_heads);

    kv_append_kernel<<<(kvd + 255) / 256, 256>>>(d_kc, d_vc, d_k, d_v, pos, kvd);

    if (wubu_cuda_attn_dev(d_q, d_kc, d_vc, d_attn, n_heads, n_kv_heads, hd,
                           kv_start, kv_len, kvd) != 0) return -6;

    return 0;
}

/* Host wrapper: everything on the host side, device KV owned here, so EG-124 can compare
 * the composite against a CPU sequence without the engine in the loop. */
WUBU_EXPORT int wubu_cuda_attn_half_host(
        const void *h_wq, const void *h_wk, const void *h_wv, const void *h_wo, int type,
        const void *h_x, const void *h_qn, const void *h_kn, const void *h_freqs,
        const float *h_kc, const float *h_vc, int kvd, int cap,
        const void *h_kappend, const void *h_vappend,
        float *h_q, float *h_k, float *h_v, float *h_attn, float *h_out,
        int hidden, int qd, int hd, int n_heads, int n_kv_heads,
        int pos, int kv_start, int kv_len, int n_rot, float theta_scale)
{
    void *dwq = NULL, *dwk = NULL, *dwv = NULL, *dwo = NULL;
    void *dx = NULL, *dqn = NULL, *dkn = NULL, *df = NULL;
    void *dkc = NULL, *dvc = NULL, *dka = NULL, *dva = NULL;
    float *dq = NULL, *dk = NULL, *dv = NULL, *dattn = NULL, *dout = NULL;
    const long long sq = (long long)(hidden / QK_K) * 82;
    const long long sk = sq, sv = sq, so = sq;
    int rc = -1;
    const size_t wb = (size_t)qd * sq;              /* qd columns for q; k/v/o sized below */
    if (cudaMalloc(&dwq, (size_t)qd * sq) != cudaSuccess) goto done;
    if (cudaMalloc(&dwk, (size_t)kvd * sk) != cudaSuccess) goto done;
    if (cudaMalloc(&dwv, (size_t)kvd * sv) != cudaSuccess) goto done;
    if (cudaMalloc(&dwo, (size_t)hidden * (size_t)(qd / QK_K) * 82) != cudaSuccess) goto done;
    if (cudaMalloc(&dx, sizeof(float) * hidden) != cudaSuccess) goto done;
    if (cudaMalloc(&dq, sizeof(float) * qd) != cudaSuccess) goto done;
    if (cudaMalloc(&dk, sizeof(float) * kvd) != cudaSuccess) goto done;
    if (cudaMalloc(&dv, sizeof(float) * kvd) != cudaSuccess) goto done;
    if (cudaMalloc(&dattn, sizeof(float) * qd) != cudaSuccess) goto done;
    if (cudaMalloc(&dout, sizeof(float) * hidden) != cudaSuccess) goto done;
    if (cudaMalloc(&dkc, (size_t)cap * kvd * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&dvc, (size_t)cap * kvd * sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&dqn, sizeof(float) * hd) != cudaSuccess) goto done;
    if (cudaMalloc(&dkn, sizeof(float) * hd) != cudaSuccess) goto done;
    if (cudaMalloc(&df, sizeof(float) * (n_rot / 2)) != cudaSuccess) goto done;
    (void)wb;

    if (cudaMemcpy(dwq, h_wq, (size_t)qd * sq, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(dwk, h_wk, (size_t)kvd * sk, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(dwv, h_wv, (size_t)kvd * sv, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(dwo, h_wo, (size_t)hidden * (size_t)(qd / QK_K) * 82, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(dx, h_x, sizeof(float) * hidden, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(dqn, h_qn, sizeof(float) * hd, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(dkn, h_kn, sizeof(float) * hd, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(df, h_freqs, sizeof(float) * (n_rot / 2), cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    /* prime the device KV with the rows before `pos`, exactly as the engine does */
    if (pos > 0) {
        if (cudaMemcpy(dkc, h_kc, (size_t)pos * kvd * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) goto done;
        if (cudaMemcpy(dvc, h_vc, (size_t)pos * kvd * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    }
    (void)h_kappend; (void)h_vappend; (void)dka; (void)dva;

    if (wubu_cuda_attn_half_dev(dwq, dwk, dwv, dwo, type, type, type, type, (const float *)dx, dq, dk, dv,
                                (float *)dkc, (float *)dvc, dattn, dout, (const float *)dqn,
                                (const float *)dkn, hidden, qd, kvd, hd, n_heads, n_kv_heads,
                                pos, kv_start, kv_len, sq, sk, sv, so, (float)pos, n_rot,
                                theta_scale, (const float *)df) != 0) goto done;
    if (cudaMemcpy(h_q, dq, sizeof(float) * qd, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    if (cudaMemcpy(h_k, dk, sizeof(float) * kvd, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    if (cudaMemcpy(h_v, dv, sizeof(float) * kvd, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    if (cudaMemcpy(h_attn, dattn, sizeof(float) * qd, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    if (cudaMemcpy(h_out, dout, sizeof(float) * hidden, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    rc = 0;
done:
    if (dwq) cudaFree(dwq); if (dwk) cudaFree(dwk); if (dwv) cudaFree(dwv); if (dwo) cudaFree(dwo);
    if (dx) cudaFree(dx); if (dq) cudaFree(dq); if (dk) cudaFree(dk); if (dv) cudaFree(dv);
    if (dattn) cudaFree(dattn); if (dout) cudaFree(dout);
    if (dkc) cudaFree(dkc); if (dvc) cudaFree(dvc);
    if (dqn) cudaFree(dqn); if (dkn) cudaFree(dkn); if (df) cudaFree(df);
    return rc;
}

/* A type per FFN tensor: the 12B model does not use one quant type for gate, up and down,
 * so the single-type fused FFN only covered a fraction of the layers and the rest went back
 * to three round trips each. */
/* ---------------- FFN over M rows, all on the device ----------------
 *
 * Per-shape accounting puts 2.4 s of a 3.37 s prefill outside the kernels: 190 matvec calls each
 * uploading an input and blocking on a download. The FFN is the worst of it - gate and up come back
 * to the host only to be GELU'd and multiplied, then go back up for the down projection: 16 MB per
 * layer across PCIe for 5.3 MB of arithmetic.
 *
 * This keeps the chain resident: one upload of the normalised rows, gate and up through the tensor
 * cores into device buffers, GELU*mul on the device, the result quantised in place and fed straight
 * to the down projection, one download. Same arithmetic as wubu_cuda_ffn_dev3, which does this for a
 * single row. */
__global__ void iq_xxs_matvec_warp_kernel(const uint8_t * __restrict__ w,
                                          const c_q8_k * __restrict__ xq8,
                                          float * __restrict__ y, int64_t col_stride_bytes,
                                          int nblocks, int n_cols, int M, int q8_row_blocks,
                                          int type, int row_group);

static int mrow_into_dev(int type, const void *dw, const c_q8_k *q8m, float *d_y,
                         int64_t stride, int nblocks, int cols, int M)
{
    if (type == 22) {
        dim3 g((cols + IQ2S_MMA_COLS - 1) / IQ2S_MMA_COLS, (M + 16 * IQ2S_MMA_RH - 1) / (16 * IQ2S_MMA_RH));
        iq2s_matvec_mma_kernel<<<g, IQ2S_MMA_WPB * 32>>>((const uint8_t *)dw, q8m, d_y, stride,
                                                        nblocks, cols, M, nblocks);
    } else if (type == 21) {
        dim3 g((cols + IQ2S_MMA_COLS - 1) / IQ2S_MMA_COLS, (M + 15) / 16);
        iq3s_matvec_mma_kernel<<<g, IQ2S_MMA_WPB * 32>>>((const uint8_t *)dw, q8m, d_y, stride,
                                                        nblocks, cols, M, nblocks);
    } else if ((type == 18 || type == 23) && xxs_mrow_enabled()) {
        /* IQ3_XXS / IQ4_XS for M rows: one warp per column, one lane per row, cooperative decode
         * into shared memory then a plain integer dot. Verified exact against a CPU dequantiser, and
         * against the per-row kernels on the real tensors apart from a handful of cancellation cases
         * (type 18: 7 of 1.34M elements over 1%; type 23: 2 of 334k).
         *
         * ON BY DEFAULT: measured against the unbatched per-token reference over 8 prompts at 8 and 24
         * tokens, this reproduces the reference in 13 of 16 cases against 9 of 16 with the kernels off
         * - i.e. it is closer to the reference *and* 1.8x faster on prefill (87 tokens: 55.3 vs 31.0
         * tok/s). Disable with G4_XXS_MMA=0. */
        const int rg = 32;
        dim3 g(cols, (M + rg - 1) / rg);
        iq_xxs_matvec_warp_kernel<<<g, 32>>>((const uint8_t *)dw, q8m, d_y, stride, nblocks, cols,
                                             M, nblocks, type, rg);
    } else {
        return -1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

WUBU_EXPORT int wubu_cuda_check_xxs(const void *, int, const void *, int, int, long long, int);
WUBU_EXPORT int wubu_cuda_check_iq3s(const void *, const c_q8_k *, int, int, long long, int);

/* Same A/B for IQ3_S, which is the down projection on almost every layer. Uses the grid from
 * mrow_into_dev so what is tested is exactly what the fused path runs. */
WUBU_EXPORT int wubu_cuda_check_iq3s(const void *d_w, const c_q8_k *d_qx, int K, int n_cols,
                                     long long stride, int M)
{
    if (!d_w || !d_qx || M < 16) return -1;
    const int nblocks = K / QK_K;
    float *d_ref = (float *)pool_get("chk.ref3", (size_t)n_cols * sizeof(float));
    float *d_mma = (float *)pool_get("chk.mma3", (size_t)n_cols * (size_t)M * sizeof(float));
    if (!d_ref || !d_mma) return -2;
    if (mv_any(d_w, 21, d_qx, d_ref, K, n_cols, stride) != 0) return -3;
    dim3 g((n_cols + IQ2S_MMA_COLS - 1) / IQ2S_MMA_COLS, (M + 15) / 16);
    iq3s_matvec_mma_kernel<<<g, IQ2S_MMA_WPB * 32>>>((const uint8_t *)d_w, d_qx, d_mma, stride,
                                                    nblocks, n_cols, M, nblocks);
    if (cudaGetLastError() != cudaSuccess) return -4;
    float *hr = (float *)malloc((size_t)n_cols * sizeof(float));
    float *hm = (float *)malloc((size_t)n_cols * sizeof(float));
    if (!hr || !hm) return -5;
    cudaMemcpy(hr, d_ref, (size_t)n_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hm, d_mma, (size_t)n_cols * sizeof(float), cudaMemcpyDeviceToHost);
    double worst = 0.0, sc = 0.0; int over1 = 0, nonzero = 0;
    for (int i = 0; i < n_cols; ++i) if (fabs(hr[i]) > sc) sc = fabs(hr[i]);
    if (sc <= 0.0) sc = 1.0;
    for (int i = 0; i < n_cols; ++i) {
        const double d = fabs((double)hr[i] - (double)hm[i]);
        if (d > worst) worst = d;
        if (d > 0.01 * sc) ++over1;
        if (d > 0.0) ++nonzero;
    }
    fprintf(stderr, "[IQ3S-CHECK] K=%d cols=%d M=%d | max abs %.6g scale %.6g rel %.4g | >1%% of scale %d, differing %d of %d\n",
            K, n_cols, M, worst, sc, worst / sc, over1, nonzero, n_cols);
    free(hr); free(hm);
    return 0;
}


/* A/B the M-row (tensor-core) IQ2_S path against the per-row path on the SAME q8 input.
 * The per-row path is the one decode uses and decode reproduces the reference tokens exactly, so a
 * disagreement here means the M-row kernel is wrong and not the quantisation. */
WUBU_EXPORT int wubu_cuda_check_iq2s(const void *d_w, const c_q8_k *d_qx, int K, int n_cols,
                                     long long stride, int M)
{
    if (!d_w || !d_qx || M < 16) return -1;
    const int nblocks = K / QK_K;
    float *d_ref = (float *)pool_get("chk.ref", (size_t)n_cols * sizeof(float));
    float *d_mma = (float *)pool_get("chk.mma", (size_t)n_cols * (size_t)M * sizeof(float));
    if (!d_ref || !d_mma) return -2;
    /* proven path: one row through whichever kernel decode uses */
    if (mv_any(d_w, 22, d_qx, d_ref, K, n_cols, stride) != 0) return -3;
    dim3 g((n_cols + IQ2S_MMA_COLS - 1) / IQ2S_MMA_COLS, (M + 16 * IQ2S_MMA_RH - 1) / (16 * IQ2S_MMA_RH));
    iq2s_matvec_mma_kernel<<<g, IQ2S_MMA_WPB * 32>>>((const uint8_t *)d_w, d_qx, d_mma, stride,
                                                    nblocks, n_cols, M, nblocks);
    if (cudaGetLastError() != cudaSuccess) return -4;
    float *hr = (float *)malloc((size_t)n_cols * sizeof(float));
    float *hm = (float *)malloc((size_t)n_cols * sizeof(float));
    if (!hr || !hm) return -5;
    cudaMemcpy(hr, d_ref, (size_t)n_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hm, d_mma, (size_t)n_cols * sizeof(float), cudaMemcpyDeviceToHost);
    double worst = 0.0; int over1 = 0, nonzero = 0; double sc = 0.0;
    for (int i = 0; i < n_cols; ++i) if (fabs(hr[i]) > sc) sc = fabs(hr[i]);
    if (sc <= 0.0) sc = 1.0;
    for (int i = 0; i < n_cols; ++i) {
        const double d = fabs((double)hr[i] - (double)hm[i]);
        if (d > worst) worst = d;
        if (d > 0.01 * sc) ++over1;
        if (d > 0.0) ++nonzero;
    }
    fprintf(stderr, "[IQ2S-CHECK] K=%d cols=%d M=%d | max abs %.6g scale %.6g rel %.4g | >1%% of scale %d, differing %d of %d\n",
            K, n_cols, M, worst, sc, worst / sc, over1, nonzero, n_cols);
    free(hr); free(hm);
    return 0;
}


/* Stage accounting for the fused FFN. Every claim this session made about where the 36 ms per layer
 * goes was arithmetic on transfer sizes; this measures the six stages directly and prints the average
 * of one full layer sweep, so the next decision is made on numbers instead of on the same guess. */
static int   g_ffn_stage_on = -1;
static double g_ffn_acc[8];
static int   g_ffn_calls;
static cudaEvent_t g_ffn_ev[8];

static void ffn_stage_mark(int slot)
{
    if (g_ffn_stage_on != 1 || slot < 0 || slot > 6) return;
    cudaEventRecord(g_ffn_ev[slot], 0);
}

WUBU_EXPORT int wubu_cuda_ffn_dev3(const void *d_wg, const void *d_wu, const void *d_wd,
                                   int tg, int tu, int td,
                                   const float *d_x, float *d_gate, float *d_up, float *d_out,
                                   int hidden, int ff, long long sg, long long su, long long sd);

int ffn_mrow_impl(const void *d_wg, int tg, const void *d_wu, int tu,
                                     const void *d_wd, int td, const float *h_x, int K, int M,
                                     int hidden, int ff, long long sg, long long su, long long sd,
                                     float *h_out)
{
    if (!d_wg || !d_wu || !d_wd || !h_x || !h_out || M <= 0) return -1;
    if ((hidden % QK_K) || (K % QK_K)) return -2;
    if (g_ffn_stage_on < 0) {
        g_ffn_stage_on = getenv("G4_FFN_STAGE") ? 1 : 0;
        if (g_ffn_stage_on) for (int i = 0; i < 8; ++i) cudaEventCreate(&g_ffn_ev[i]);
    }

    const int nbx = K / QK_K, nbf = ff / QK_K;
    float *dx = (float *)pool_get("ffn.x", (size_t)K * M * sizeof(float));
    float *dg = (float *)pool_get("ffn.gate", (size_t)ff * M * sizeof(float));
    float *du = (float *)pool_get("ffn.up", (size_t)ff * M * sizeof(float));
    float *dout = (float *)pool_get("ffn.out", (size_t)hidden * M * sizeof(float));
    c_q8_k *qx = (c_q8_k *)pool_get("ffn.q8x", (size_t)nbx * M * sizeof(c_q8_k));
    c_q8_k *qf = (c_q8_k *)pool_get("ffn.q8f", (size_t)nbf * M * sizeof(c_q8_k));
    if (!dx || !dg || !du || !dout || !qx || !qf) return -3;

    ffn_stage_mark(0);
    if (cudaMemcpy(dx, h_x, (size_t)K * M * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        return -4;
    ffn_stage_mark(1);
    quantize_q8k_kernel<<<dim3(nbx, M), 32>>>(dx, qx, hidden == K ? K : K, K, nbx);
    ffn_stage_mark(2);
    if (getenv("G4_Q8_CHECK")) {
        static int done_q8;
        if (!done_q8) {
            done_q8 = 1;
            c_q8_k *ref = (c_q8_k *)pool_get("chk.q8", (size_t)nbx * sizeof(c_q8_k));
            if (ref) {
                quantize_q8k_kernel<<<nbx, 32>>>(dx, ref, K, 0, 0);   /* the M=1 launch, row 0 */
                c_q8_k *a = (c_q8_k *)malloc((size_t)nbx * sizeof(c_q8_k));
                c_q8_k *b = (c_q8_k *)malloc((size_t)nbx * sizeof(c_q8_k));
                if (a && b) {
                    cudaMemcpy(a, qx, (size_t)nbx * sizeof(c_q8_k), cudaMemcpyDeviceToHost);
                    cudaMemcpy(b, ref, (size_t)nbx * sizeof(c_q8_k), cudaMemcpyDeviceToHost);
                    int bad_blocks = 0, bad_vals = 0; double worst_d = 0.0;
                    for (int i = 0; i < nbx; ++i) {
                        const double dd = fabs((double)a[i].d - (double)b[i].d);
                        if (dd > worst_d) worst_d = dd;
                        int bad = 0;
                        for (int j = 0; j < QK_K; ++j) if (a[i].qs[j] != b[i].qs[j]) ++bad;
                        if (bad || a[i].d != b[i].d) ++bad_blocks;
                        bad_vals += bad;
                    }
                    fprintf(stderr, "[Q8-CHECK] blocks=%d differing blocks=%d differing values=%d worst d delta=%.6g\n",
                            nbx, bad_blocks, bad_vals, worst_d);
                    free(a); free(b);
                }
            }
        }
    }
    if (getenv("G4_IQ2S_CHECK") && tg == 22) {
        static int done_i2;
        if (!done_i2) { done_i2 = 1; wubu_cuda_check_iq2s(d_wg, qx, K, ff, sg, M); }
    }
    if (getenv("G4_XXS_CHECK") && (tg == 18 || tg == 23)) {
        static int done;
        if (!done) {
            done = 1;
            wubu_cuda_check_xxs(d_wg, tg, qx, K, ff, sg, M);
        }
    }
    if (mrow_into_dev(tg, d_wg, qx, dg, sg, nbx, ff, M) != 0) return -5;
    if (mrow_into_dev(tu, d_wu, qx, du, su, nbx, ff, M) != 0) return -6;
    ffn_stage_mark(3);
    const size_t elems = (size_t)ff * (size_t)M;
    gelu_mul_kernel<<<(elems + 255) / 256, 256>>>(dg, du, (int)elems);
    ffn_stage_mark(4);
    /* gelu_mul_kernel(g, u, n) writes its product into u, so the gated values are in du. This
     * quantised dg - the raw gate - and the down projection consumed it, which is why the batched
     * FFN disagreed with the per-token path by ~110% on every row while each matvec, the
     * quantisation and the gelu all verified correct in isolation. */
    quantize_q8k_kernel<<<dim3(nbf, M), 32>>>(du, qf, ff, ff, nbf);
    ffn_stage_mark(5);
    if (getenv("G4_IQ2S_CHECK") && td == 22) {
        static int done_i2d;
        if (!done_i2d) { done_i2d = 1; wubu_cuda_check_iq2s(d_wd, qf, ff, hidden, sd, M); }
    }
    if (getenv("G4_IQ2S_CHECK") && td == 21) {
        static int done_i3d;
        if (!done_i3d) { done_i3d = 1; wubu_cuda_check_iq3s(d_wd, qf, ff, hidden, sd, M); }
    }
    if (getenv("G4_XXS_CHECK") && (td == 18 || td == 23)) {
        static int done_d;
        if (!done_d) {
            done_d = 1;
            wubu_cuda_check_xxs(d_wd, td, qf, ff, hidden, sd, M);
        }
    }
    if (mrow_into_dev(td, d_wd, qf, dout, sd, nbf, hidden, M) != 0) return -9;
    if (getenv("G4_FFN_ROWCHECK")) {
        static int done_rc;
        if (!done_rc) {
            done_rc = 1;
            /* The M=1 fused path is the one that reproduces the reference tokens. Run it on row 0 and
             * compare against row 0 of the M-row result: same weights, same input row, same quantisation,
             * so any difference is the wiring of ffn_mrow_impl and nothing else. */
            float *tg_ = (float *)pool_get("chk.fg", (size_t)ff * sizeof(float));
            float *tu_ = (float *)pool_get("chk.fu", (size_t)ff * sizeof(float));
            float *to_ = (float *)pool_get("chk.fo", (size_t)hidden * sizeof(float));
            if (tg_ && tu_ && to_) {
                const int rc = wubu_cuda_ffn_dev3(d_wg, d_wu, d_wd, tg, tu, td, dx, tg_, tu_, to_,
                                                  hidden, ff, sg, su, sd);
                float *hr = (float *)malloc((size_t)hidden * sizeof(float));
                float *hm = (float *)malloc((size_t)hidden * sizeof(float));
                if (rc == 0 && hr && hm) {
                    cudaMemcpy(hr, to_, (size_t)hidden * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(hm, dout, (size_t)hidden * sizeof(float), cudaMemcpyDeviceToHost);
                    double worst = 0.0, sc = 0.0; int over1 = 0;
                    for (int i = 0; i < hidden; ++i) if (fabs(hr[i]) > sc) sc = fabs(hr[i]);
                    if (sc <= 0.0) sc = 1.0;
                    for (int i = 0; i < hidden; ++i) {
                        const double dv = fabs((double)hr[i] - (double)hm[i]);
                        if (dv > worst) worst = dv;
                        if (dv > 0.01 * sc) ++over1;
                    }
                    fprintf(stderr, "[FFN-ROWCHECK] row0 M-row vs M=1 fused: max abs %.6g scale %.6g rel %.4g ->1%% %d of %d (rc=%d)\n",
                            worst, sc, worst / sc, over1, hidden, rc);
                } else {
                    fprintf(stderr, "[FFN-ROWCHECK] M=1 path returned %d\n", rc);
                }
                free(hr); free(hm);
            }
        }
    }
    ffn_stage_mark(6);
    if (cudaMemcpy(h_out, dout, (size_t)hidden * M * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return -13;
    ffn_stage_mark(7);

    if (g_ffn_stage_on == 1) {
        cudaEventSynchronize(g_ffn_ev[7]);
        static const char *nm[8] = {"upload","q8(x)","gate+up","gelu","q8(gu)","down","download","total"};
        for (int i = 0; i < 7; ++i) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, g_ffn_ev[i], g_ffn_ev[i + 1]);
            g_ffn_acc[i] += (double)ms;
        }
        if (++g_ffn_calls % 43 == 0) {
            double tot = 0.0;
            for (int i = 0; i < 7; ++i) tot += g_ffn_acc[i];
            fprintf(stderr, "[FFN-STAGE] per layer call, ms:");
            for (int i = 0; i < 7; ++i)
                fprintf(stderr, " %s %.2f", nm[i], g_ffn_acc[i] / (43.0 * (double)(g_ffn_calls / 43)));
            fprintf(stderr, " | total %.2f\n", tot / (43.0 * (double)(g_ffn_calls / 43)));
            for (int i = 0; i < 7; ++i) g_ffn_acc[i] = 0.0;
        }
    }
    return 0;
}

WUBU_EXPORT int wubu_cuda_ffn_mrow_dev3(const void *d_wg, int tg, const void *d_wu, int tu,
                                         const void *d_wd, int td, const float *h_x, int K, int M,
                                         int hidden, int ff, long long sg, long long su, long long sd,
                                         float *h_out)
{
    return ffn_mrow_impl(d_wg, tg, d_wu, tu, d_wd, td, h_x, K, M, hidden, ff, sg, su, sd, h_out);
}

WUBU_EXPORT int wubu_cuda_ffn_dev3(const void *d_wg, const void *d_wu, const void *d_wd,
                                   int tg, int tu, int td,
                                   const float *d_x, float *d_gate, float *d_up, float *d_out,
                                   int hidden, int ff, long long sg, long long su, long long sd)
{
    if (!d_wg || !d_wu || !d_wd || !d_x || !d_gate || !d_up || !d_out) return -1;
    if ((hidden % QK_K) != 0 || (ff % QK_K) != 0) return -1;
    if (matvec_devbuf(d_wg, tg, d_x, d_gate, hidden, ff, sg) != 0) return -2;
    if (matvec_devbuf(d_wu, tu, d_x, d_up, hidden, ff, su) != 0) return -3;
    gelu_mul_kernel<<<(ff + 255) / 256, 256>>>(d_gate, d_up, ff);
    if (matvec_devbuf(d_wd, td, d_up, d_out, ff, hidden, sd) != 0) return -4;
    return cudaGetLastError() == cudaSuccess ? 0 : -5;
}

WUBU_EXPORT int wubu_cuda_ffn_dev(const void *d_wg, const void *d_wu, const void *d_wd, int type,
                                  const float *d_x, float *d_gate, float *d_up, float *d_out,
                                  int hidden, int ff, long long sg, long long su, long long sd)
{
    return wubu_cuda_ffn_dev3(d_wg, d_wu, d_wd, type, type, type, d_x, d_gate, d_up, d_out,
                              hidden, ff, sg, su, sd);
}

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
    if (type == 16)      mrc = wubu_cuda_matvec_iq2xxs(d_w, d_xq8, d_y, K, n_cols, col_stride_bytes, 1);
    else if (type == 8)  mrc = wubu_cuda_matvec_q8_0(d_w, d_xq8, d_y, K, n_cols, col_stride_bytes);
    else if (type == 17) mrc = wubu_cuda_matvec_iq2xs(d_w, d_xq8, d_y, K, n_cols, col_stride_bytes);
    else if (type == 18) mrc = wubu_cuda_matvec_iq3xxs(d_w, d_xq8, d_y, K, n_cols, col_stride_bytes);
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

/* ---------------- prefill attention block, entirely on the device ----------------
 *
 * wubu_cuda_attn_half_dev does the whole block for one row: q/k/v, head norms, RoPE, KV append,
 * attention, o-projection. Prefill had no equivalent, so per layer it downloaded q (2.8 MB), uploaded
 * q/k/v again for attention (5.6 MB) and uploaded the attention output (2.8 MB) - 11 MB of PCIe for
 * a block whose arithmetic is 5 MB - and ran the head norms and RoPE on the CPU (162 ms of a 3.5 s
 * prefill). This is that block for M rows, with the same norms and the same RoPE construction.
 *
 * Requires positions to start at 0, so the device KV slab indexes by absolute position; anything else
 * returns non-zero and the caller keeps its host path. */
__global__ void rope_kernel_mrow(float * __restrict__ q, float * __restrict__ k,
                                 const int * __restrict__ positions, int d_head, int q_stride,
                                 int k_stride, int n_rot, float theta_scale,
                                 const float * __restrict__ freqs, int n_q_heads, int n_k_heads)
{
    const int half = n_rot / 2;
    const size_t row = blockIdx.x;
    const float pos = (float)positions[row];
    if (blockIdx.z != 0) { }
    else for (int h = blockIdx.y; h < n_q_heads; h += gridDim.y) {
        float *qv = q + row * q_stride + (size_t)h * d_head;
        for (int j = threadIdx.x; j < half; j += blockDim.x) {
            const float ff = freqs ? freqs[j] : 1.0f;
            if (ff > 1e20f) continue;
            float tf = 1.0f;
            for (int kk = 0; kk < j; kk++) tf *= theta_scale;
            const float theta = pos * tf / ff;
            float st, ct;
            wubu_sincosf(theta, &st, &ct);
            const float q0 = qv[j], q1 = qv[j + half];
            qv[j] = q0 * ct - q1 * st;
            qv[j + half] = q0 * st + q1 * ct;
        }
    }
    if (!k || blockIdx.y != 0) return;
    for (int h = blockIdx.z; h < n_k_heads; h += gridDim.z) {
        float *kv = k + row * k_stride + (size_t)h * d_head;
        for (int j = threadIdx.x; j < half; j += blockDim.x) {
            const float ff = freqs ? freqs[j] : 1.0f;
            if (ff > 1e20f) continue;
            float tf = 1.0f;
            for (int kk = 0; kk < j; kk++) tf *= theta_scale;
            const float theta = pos * tf / ff;
            float st, ct;
            wubu_sincosf(theta, &st, &ct);
            const float k0 = kv[j], k1 = kv[j + half];
            kv[j] = k0 * ct - k1 * st;
            kv[j + half] = k0 * st + k1 * ct;
        }
    }
}

__global__ void kv_append_mrow_kernel(float * __restrict__ kc, float * __restrict__ vc,
                                      const float * __restrict__ k, const float * __restrict__ v,
                                      const int * __restrict__ positions, int kvd, int M)
{
    const int t = blockIdx.y;
    if (t >= M) return;
    const int dst = positions[t];
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < kvd; i += blockDim.x * gridDim.x) {
        kc[(size_t)dst * kvd + i] = k[(size_t)t * kvd + i];
        vc[(size_t)dst * kvd + i] = v[(size_t)t * kvd + i];
    }
}

WUBU_EXPORT int wubu_cuda_attn_block_mrow(const void *d_wq, int tq, const void *d_wk, int tk,
                                          const void *d_wv, int tv,
                                          const float *h_x, const int *h_pos, int N,
                                          int hidden, int qd, int kvd, int hd,
                                          int n_heads, int n_kv_heads, int n_rot, float theta_scale,
                                          const float *d_qn, const float *d_kn, const float *d_freqs,
                                          long long sq, long long sk, long long sv,
                                          float *h_out, float *h_k, float *h_v)
{
    if (!d_wq || !d_wk || !d_wv || !h_x || !h_pos || !h_out || N <= 0) return -1;
    if (h_pos[0] != 0) return -1;                       /* slab indexes by absolute position */
    if ((hidden % QK_K) || (qd % QK_K) || (kvd % 256)) return -1;

    const int nbx = hidden / QK_K, nbq = qd / QK_K;
    float *dx = (float *)pool_get("attn.x", (size_t)hidden * N * sizeof(float));
    float *dq = (float *)pool_get("attn.dq", (size_t)qd * N * sizeof(float));
    float *dk = (float *)pool_get("attn.dk", (size_t)kvd * N * sizeof(float));
    float *dv = (float *)pool_get("attn.dv", (size_t)kvd * N * sizeof(float));
    float *dattn = (float *)pool_get("attn.dout", (size_t)qd * N * sizeof(float));
    float *dout = (float *)pool_get("attn.dres", (size_t)hidden * N * sizeof(float));
    float *kc = (float *)pool_get("attn.kc", (size_t)kvd * N * sizeof(float));
    float *vc = (float *)pool_get("attn.vc", (size_t)kvd * N * sizeof(float));
    c_q8_k *qx = (c_q8_k *)pool_get("attn.q8x", (size_t)nbx * N * sizeof(c_q8_k));
    c_q8_k *qa = (c_q8_k *)pool_get("attn.q8a", (size_t)nbq * N * sizeof(c_q8_k));
    int *dp = (int *)pool_get("attn.pos", (size_t)N * sizeof(int));
    if (!dx || !dq || !dk || !dv || !dattn || !dout || !kc || !vc || !qx || !qa || !dp) return -2;

    if (cudaMemcpy(dx, h_x, (size_t)hidden * N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        return -3;
    if (cudaMemcpy(dp, h_pos, (size_t)N * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
        return -3;
    quantize_q8k_kernel<<<dim3(nbx, N), 32>>>(dx, qx, hidden, hidden, nbx);
    if (cudaGetLastError() != cudaSuccess) return -4;

    if (mrow_into_dev(tq, d_wq, qx, dq, sq, nbx, qd, N) != 0) return -5;
    if (mrow_into_dev(tk, d_wk, qx, dk, sk, nbx, kvd, N) != 0) return -6;
    if (mrow_into_dev(tv, d_wv, qx, dv, sv, nbx, kvd, N) != 0) return -7;

    /* head norms: rows are (token, head) pairs, head_dim innermost, so the row count is heads*N */
    rms_norm_row_kernel<<<n_heads * N, 256>>>(dq, d_qn, dq, hd, 1e-6f, 1);
    rms_norm_row_kernel<<<n_kv_heads * N, 256>>>(dk, d_kn, dk, hd, 1e-6f, 1);
    rms_norm_row_kernel<<<n_kv_heads * N, 256>>>(dv, NULL, dv, hd, 1e-6f, 0);
    if (cudaGetLastError() != cudaSuccess) return -8;

    rope_kernel_mrow<<<dim3(N, n_heads, n_kv_heads), 128>>>(dq, dk, dp, hd, qd, kvd, n_rot,
                                                            theta_scale, d_freqs, n_heads, n_kv_heads);
    if (cudaGetLastError() != cudaSuccess) return -9;

    kv_append_mrow_kernel<<<dim3((kvd + 255) / 256, N), 256>>>(kc, vc, dk, dv, dp, kvd, N);
    if (cudaGetLastError() != cudaSuccess) return -10;

    {
        const size_t shmem = (size_t)(N < ATTN_MAXK ? N : ATTN_MAXK) * sizeof(float);
        attn_causal_kernel<<<N * n_heads, 256, shmem>>>(dq, kc, vc, dattn, dp, N, n_heads, n_kv_heads,
                                                        hd, qd, kvd, N, 0);
        if (cudaGetLastError() != cudaSuccess) return -11;
    }

    quantize_q8k_kernel<<<dim3(nbq, N), 32>>>(dattn, qa, qd, qd, nbq);
    if (cudaGetLastError() != cudaSuccess) return -12;
    /* the o-projection is the caller's; hand back the normalised attention output and the KV rows */
    if (cudaMemcpy(h_out, dattn, (size_t)qd * N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return -13;
    if (h_k && cudaMemcpy(h_k, dk, (size_t)kvd * N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return -14;
    if (h_v && cudaMemcpy(h_v, dv, (size_t)kvd * N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return -15;
    return 0;
}



/* Diagnostic: prove the IQ3_XXS / IQ4_XS codebooks actually reached the device. A missing table makes
 * the mma kernel return zero for every column, which shows up as plausible-looking but wrong text. */
WUBU_EXPORT int wubu_cuda_dump_tables(void)
{
    uint32_t g[4] = {0, 0, 0, 0};
    int8_t v[4] = {0, 0, 0, 0};
    if (cudaMemcpyFromSymbol(g, c_iq3xxs_grid, sizeof(g)) != cudaSuccess) return -1;
    if (cudaMemcpyFromSymbol(v, c_iq4nl, sizeof(v)) != cudaSuccess) return -2;
    fprintf(stderr, "[TBL] iq3xxs %08x %08x %08x %08x | iq4nl %d %d %d %d\n",
            g[0], g[1], g[2], g[3], v[0], v[1], v[2], v[3]);
    return 0;
}

/* IQ3_XXS (18) and IQ4_XS (23) for M rows, warp per column. The single-row kernels exist and are
 * correct, but the prefill path calls them once per row and each row re-reads all 19.9 MB of weights -
 * that is the 205-253 ms the five layers using these types cost. Here one warp takes a column: the 32
 * lanes decode the block cooperatively (lane = group*4 + entry, eight values each, exactly the
 * verified decode), then every lane dots ITS row against shared memory with dp4a. */
__global__ void iq_xxs_matvec_warp_kernel(const uint8_t * __restrict__ w,
                                          const c_q8_k * __restrict__ xq8,
                                          float * __restrict__ y, int64_t col_stride_bytes,
                                          int nblocks, int n_cols, int M, int q8_row_blocks,
                                          int type, int row_group)
{
    __shared__ int8_t sW[256];
    __shared__ float  sSc[8];

    const int lane = threadIdx.x & 31;
    const int col = blockIdx.x;
    const int row = blockIdx.y * row_group + lane;
    if (col >= n_cols) return;
    const uint8_t *wcol = w + (int64_t)col * col_stride_bytes;
    const int g = lane >> 2, entry = lane & 3;
    const int wbytes = (type == 18) ? 98 : 136;

    float acc = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t *blk = wcol + (int64_t)b * wbytes;
        const float dsuper = f16_fast((uint16_t)(blk[0] | ((uint16_t)blk[1] << 8)));
        if (type == 18) {
            uint32_t a32;
            memcpy(&a32, blk + 66 + 4 * g, 4);
            const uint8_t signs = c_ksigns_iq2xs[(a32 >> (7 * entry)) & 127];
            int32_t a0, a1;
            memcpy(&a0, &c_iq3xxs_grid[blk[2 + 8 * g + 2 * entry + 0]], 4);
            memcpy(&a1, &c_iq3xxs_grid[blk[2 + 8 * g + 2 * entry + 1]], 4);
            const uint32_t ml = (((uint32_t)(signs & 0x0F) * 0x00204081u) & 0x01010101u) * 0xFFu;
            const uint32_t mh = (((uint32_t)(signs >> 4) * 0x00204081u) & 0x01010101u) * 0xFFu;
            a0 = (int32_t)__vsub4((uint32_t)a0 ^ ml, ml);
            a1 = (int32_t)__vsub4((uint32_t)a1 ^ mh, mh);
            memcpy(&sW[g * 32 + entry * 8 + 0], &a0, 4);
            memcpy(&sW[g * 32 + entry * 8 + 4], &a1, 4);
            sSc[g] = dsuper * (0.5f + (float)(a32 >> 28)) * 0.5f;
        } else {
            const uint16_t sh = (uint16_t)(blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t *sl = blk + 4, *qs = blk + 8;
            const int ls = ((sl[g / 2] >> (4 * (g % 2))) & 0xf) | (((sh >> (2 * g)) & 3) << 4);
            const uint8_t *qb = qs + 16 * g + (entry & 1) * 8;
            const int high = (entry >= 2);
            int32_t a0 = 0, a1 = 0;
            for (int j = 0; j < 4; ++j) {
                const int v0 = high ? (qb[j] >> 4) : (qb[j] & 0xf);
                const int v1 = high ? (qb[4 + j] >> 4) : (qb[4 + j] & 0xf);
                ((int8_t *)&a0)[j] = c_iq4nl[v0];
                ((int8_t *)&a1)[j] = c_iq4nl[v1];
            }
            memcpy(&sW[g * 32 + entry * 8 + 0], &a0, 4);
            memcpy(&sW[g * 32 + entry * 8 + 4], &a1, 4);
            sSc[g] = dsuper * (float)(ls - 32);
        }
        __syncwarp();

        if (row < M) {
            const c_q8_k *q8b = xq8 + (int64_t)row * q8_row_blocks + b;
            for (int gg = 0; gg < 8; ++gg) {
                const int8_t *q = q8b->qs + gg * 32;
                const int8_t *s = sW + gg * 32;
                int sumi = 0;
                for (int v = 0; v < 32; v += 4) {
                    int32_t qa, wa;
                    memcpy(&qa, q + v, 4);
                    memcpy(&wa, s + v, 4);
                    sumi = __dp4a(qa, wa, sumi);
                }
                acc += (float)sumi * sSc[gg] * q8b->d;
            }
        }
        __syncwarp();
    }
    if (row < M) y[(int64_t)row * n_cols + col] = acc;
}

/* A/B the two kernels on identical arguments: the M-row path against the per-row kernel that decode
 * and the fallback already use. Whichever side is wrong becomes visible as a per-row difference. */
WUBU_EXPORT int wubu_cuda_check_xxs(const void *d_w, int type, const void *d_q8, int K, int n_cols,
                                    long long stride, int M)
{
    const int nb = K / QK_K;
    const c_q8_k *q8 = (const c_q8_k *)d_q8;
    float *y1 = (float *)malloc((size_t)M * n_cols * sizeof(float));
    float *y2 = (float *)malloc((size_t)M * n_cols * sizeof(float));
    if (!y1 || !y2) return -1;
    float *dy = (float *)pool_get("chk.y", (size_t)M * (size_t)n_cols * sizeof(float));
    if (!dy) { free(y1); free(y2); return -2; }
    dim3 g(n_cols, (M + 31) / 32);
    iq_xxs_matvec_warp_kernel<<<g, 32>>>((const uint8_t *)d_w, q8, dy, stride, nb, n_cols, M, nb, type, 32);
    if (cudaGetLastError() != cudaSuccess) { free(y1); free(y2); return -3; }
    cudaDeviceSynchronize();
    if (cudaMemcpy(y1, dy, (size_t)M * n_cols * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return -4;
    for (int r = 0; r < M; ++r) {
        int rc = (type == 18)
            ? wubu_cuda_matvec_iq3xxs(d_w, (const char *)d_q8 + (size_t)r * nb * sizeof(c_q8_k),
                                      dy, K, n_cols, stride)
            : wubu_cuda_matvec_iq4xs(d_w, (const char *)d_q8 + (size_t)r * nb * sizeof(c_q8_k),
                                     dy, K, n_cols, stride);
        if (rc != 0) { free(y1); free(y2); return -5; }
        cudaDeviceSynchronize();
        if (cudaMemcpy(y2 + (size_t)r * n_cols, dy, (size_t)n_cols * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) { free(y1); free(y2); return -6; }
    }
    double worst = 0.0; int wi = -1, wj = -1;
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < n_cols; ++c) {
            const double a = y1[(size_t)r * n_cols + c], b2 = y2[(size_t)r * n_cols + c];
            const double den = fabs(b2) > 1e-6 ? fabs(b2) : 1.0;
            const double rel = fabs(a - b2) / den;
            if (rel > worst) { worst = rel; wi = r; wj = c; }
        }
    int over1 = 0, over01 = 0, nonzero = 0;
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < n_cols; ++c) {
            const double a = y1[(size_t)r * n_cols + c], b2 = y2[(size_t)r * n_cols + c];
            const double den = fabs(b2) > 1e-6 ? fabs(b2) : 1.0;
            const double rel = fabs(a - b2) / den;
            if (fabs(a - b2) > 0.0) ++nonzero;
            if (rel > 0.01) ++over1;
            if (rel > 0.001) ++over01;
        }
    fprintf(stderr, "[XXS] type %d K %d cols %d M %d | worst %.4g (r%d c%d: %.6g vs %.6g) | "
                    "elements differing: %d, >0.1%%: %d, >1%%: %d of %d\n",
            type, K, n_cols, M, worst, wi, wj,
            wi >= 0 ? (double)y1[(size_t)wi * n_cols + wj] : 0.0,
            wi >= 0 ? (double)y2[(size_t)wi * n_cols + wj] : 0.0,
            nonzero, over01, over1, M * n_cols);
    free(y1); free(y2);
    return 0;
}


/* Isolation test: run the same kernel the resident path calls, on bytes the CALLER supplies, and hand back y.
 * With the engine's own wubu_iq_rows run on the identical slab, this separates "the kernel is wrong" from "the slab it
 * was given was wrong" - the distinction that engine-level instrumentation could not make. */
WUBU_EXPORT int wubu_cuda_selftest_iq2_run(int type, const void *h_w, size_t wbytes, const void *h_q8, int q8_bytes,
                                           int K, int n_cols, int64_t row_bytes, float *h_y_out)
{
    if (!h_w || !h_q8 || !h_y_out || K <= 0 || n_cols <= 0 || row_bytes <= 0 || wbytes == 0) return -1;
    void *d_w = NULL, *d_q8 = NULL, *d_y = NULL;
    if (cudaMalloc(&d_w, wbytes) != cudaSuccess) return -3;
    if (cudaMalloc(&d_q8, (size_t) q8_bytes) != cudaSuccess) { cudaFree(d_w); return -3; }
    if (cudaMalloc(&d_y, (size_t) n_cols * sizeof(float)) != cudaSuccess) { cudaFree(d_w); cudaFree(d_q8); return -3; }
    cudaMemcpy(d_w, h_w, wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q8, h_q8, (size_t) q8_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, (size_t) n_cols * sizeof(float));
    const int rc = (type == 16)
        ? wubu_cuda_matvec_iq2xxs(d_w, d_q8, (float *) d_y, K, n_cols, row_bytes, 1)
        : (type == 22)
        ? wubu_cuda_matvec_iq2s_v2(d_w, d_q8, (float *) d_y, K, n_cols, row_bytes)
        : (type == 23)
        ? wubu_cuda_matvec_iq4xs(d_w, d_q8, (float *) d_y, K, n_cols, row_bytes)
        : -4;
    const cudaError_t sync = cudaDeviceSynchronize();
    cudaMemcpy(h_y_out, d_y, (size_t) n_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_w); cudaFree(d_q8); cudaFree(d_y);
    if (sync != cudaSuccess) return -5;
    return rc;
}

/* Exported so the engine can report what the resident tier actually did, instead of the tier being a black box with a
 * hit rate nobody measured. Called at exit under IN2_GPU_STATS. */
WUBU_EXPORT void wubu_cuda_res_stats(long *hit, long *miss, long *evict, long *slots, long *budget_mb)
{
    if (hit) *hit = g_res_hit;
    if (miss) *miss = g_res_miss;
    if (evict) *evict = g_res_evict;
    if (slots) *slots = g_res_n;
    if (budget_mb) *budget_mb = (long) (g_res_budget / (1024 * 1024));
}

/* Batched expert matvec: one launch for a whole layer's active experts.
 * The measured problem with per-expert calls is not bandwidth (92 % of slabs are already resident) but launch count:
 * 24 layers x 8 experts = 192 serialised small launches per token, each doing ~2.6 MMAC. Gathering the 8 slabs into one
 * contiguous device span and issuing a single matvec of n_exp x n_rows rows removes 7/8 of that overhead.
 * The gather is n_exp memcpys of ~1.4 MB, and the slabs are supplied already resident by the caller. */
WUBU_EXPORT int wubu_cuda_moe_batch(int type, const void *const *h_w, int n_exp, size_t wbytes_each,
                                   const void *h_act, int act_bytes, float *h_y, int K, int n_rows_each)
{
    if (type != 16 && type != 22) return -1;
    if (!h_w || n_exp <= 0 || wbytes_each == 0 || !h_act || !h_y || K <= 0 || n_rows_each <= 0) return -1;
    const int64_t row_bytes = (int64_t) (wbytes_each / (size_t) n_rows_each);
    if (row_bytes <= 0 || (size_t) row_bytes * (size_t) n_rows_each != wbytes_each) return -1;
    const int n_cols = n_exp * n_rows_each;
    const size_t wb = wbytes_each * (size_t) n_exp;

    static void *b_w, *b_act, *b_y;
    static size_t b_wc, b_ac, b_yc;
    static std::atomic<int> b_lock{0};
    int e = 0;
    while (!b_lock.compare_exchange_weak(e, 1, std::memory_order_acquire)) e = 0;

    if (wb > b_wc) { if (b_w) { cudaFree(b_w); b_w = NULL; } if (cudaMalloc(&b_w, wb) != cudaSuccess) { b_lock.store(0); return -3; } b_wc = wb; }
    if ((size_t) act_bytes > b_ac) { if (b_act) { cudaFree(b_act); b_act = NULL; } if (cudaMalloc(&b_act, (size_t) act_bytes) != cudaSuccess) { b_lock.store(0); return -3; } b_ac = (size_t) act_bytes; }
    const size_t yb = (size_t) n_cols * sizeof(float);
    if (yb > b_yc) { if (b_y) { cudaFree(b_y); b_y = NULL; } if (cudaMalloc(&b_y, yb) != cudaSuccess) { b_lock.store(0); return -3; } b_yc = yb; }

    for (int i = 0; i < n_exp; ++i) {
        if (cudaMemcpy((char *) b_w + (size_t) i * wbytes_each, h_w[i], wbytes_each, cudaMemcpyHostToDevice) != cudaSuccess) {
            b_lock.store(0); return -4;
        }
    }
    if (cudaMemcpy(b_act, h_act, (size_t) act_bytes, cudaMemcpyHostToDevice) != cudaSuccess) { b_lock.store(0); return -4; }

    const int rc = (type == 16)
        ? wubu_cuda_matvec_iq2xxs(b_w, b_act, (float *) b_y, K, n_cols, row_bytes, 1)
        : wubu_cuda_matvec_iq2s_v2(b_w, b_act, (float *) b_y, K, n_cols, row_bytes);
    if (cudaDeviceSynchronize() != cudaSuccess) { b_lock.store(0); return -5; }
    if (cudaMemcpy(h_y, b_y, yb, cudaMemcpyDeviceToHost) != cudaSuccess) { b_lock.store(0); return -6; }
    b_lock.store(0);
    return rc;
}

/* Batched expert matvec with a DEVICE-to-DEVICE gather from the resident tier.
 * The host-pointer version of this entry was measured SLOWER than per-expert calls (5804 vs 5120 ms on a 24-token
 * decode) because it gathered 8 x 1.31 MB x 24 layers = 250 MB/token over PCIe - more than the 192 launches cost. At a
 * 92 % hit rate the slabs are already resident, so gather D2D (~500 GB/s) and touch the bus not at all.
 * Returns -7 if any ident is not resident: the caller must fall back rather than guess. */
WUBU_EXPORT int wubu_cuda_moe_batch_res(int type, const int64_t *idents, const void *const *h_w, int n_exp,
                                       size_t wbytes_each, const void *h_act, int act_bytes, float *h_y,
                                       int K, int n_rows_each)
{
    if (type != 16 && type != 22) return -1;
    if (!idents || !h_w || n_exp <= 0 || wbytes_each == 0 || !h_act || !h_y || K <= 0 || n_rows_each <= 0) return -1;
    const int64_t row_bytes = (int64_t) (wbytes_each / (size_t) n_rows_each);
    if (row_bytes <= 0 || (size_t) row_bytes * (size_t) n_rows_each != wbytes_each) return -1;
    const int n_cols = n_exp * n_rows_each;
    const size_t wb = wbytes_each * (size_t) n_exp;

    static void *b_w2, *b_act2, *b_y2;
    static size_t b_w2c, b_act2c, b_y2c;
    static std::atomic<int> b2_lock{0};
    int e = 0;
    while (!b2_lock.compare_exchange_weak(e, 1, std::memory_order_acquire)) e = 0;
#define B2_FAIL(v) do { b2_lock.store(0); return (v); } while (0)

    if (wb > b_w2c) { if (b_w2) { cudaFree(b_w2); b_w2 = NULL; } if (cudaMalloc(&b_w2, wb) != cudaSuccess) B2_FAIL(-3); b_w2c = wb; }
    if ((size_t) act_bytes > b_act2c) { if (b_act2) { cudaFree(b_act2); b_act2 = NULL; } if (cudaMalloc(&b_act2, (size_t) act_bytes) != cudaSuccess) B2_FAIL(-3); b_act2c = (size_t) act_bytes; }
    const size_t yb = (size_t) n_cols * sizeof(float);
    if (yb > b_y2c) { if (b_y2) { cudaFree(b_y2); b_y2 = NULL; } if (cudaMalloc(&b_y2, yb) != cudaSuccess) B2_FAIL(-3); b_y2c = yb; }

    for (int i = 0; i < n_exp; ++i) {
        int slot = -1;
        for (int j = 0; j < g_res_n; ++j)
            if (g_res_d[j] && g_res_ident[j] == idents[i] && g_res_bytes[j] == wbytes_each) { slot = j; break; }
        if (slot < 0) {
            /* POPULATE on a miss. This is the piece whose absence made batching slower than the per-expert path: the
             * batched branch replaces the calls that used to fill this cache, so without filling here every call misses
             * and pays the full host gather - measured 5870 vs 5138 ms. Fill once, gather D2D forever after. */
            const int room = (g_res_n < WUBU_RES_SLOTS) &&
                             (!g_res_budget || g_res_used + wbytes_each <= g_res_budget);
            if (room) {
                slot = g_res_n;
                if (cudaMalloc(&g_res_d[slot], wbytes_each) == cudaSuccess) {
                    if (cudaMemcpy(g_res_d[slot], h_w[i], wbytes_each, cudaMemcpyHostToDevice) == cudaSuccess) {
                        g_res_n++;
                        g_res_key[slot] = h_w[i];
                        g_res_bytes[slot] = wbytes_each;
                        g_res_ident[slot] = idents[i];
                        g_res_use[slot] = ++g_res_clock;
                        g_res_used += wbytes_each;
                        g_res_miss++;
                    } else { cudaFree(g_res_d[slot]); g_res_d[slot] = NULL; slot = -1; }
                } else { g_res_d[slot] = NULL; slot = -1; }
            }
        }
        if (slot < 0) {
            if (cudaMemcpy((char *) b_w2 + (size_t) i * wbytes_each, h_w[i], wbytes_each, cudaMemcpyHostToDevice) != cudaSuccess) B2_FAIL(-4);
            continue;
        }
        if (cudaMemcpyAsync((char *) b_w2 + (size_t) i * wbytes_each, g_res_d[slot], wbytes_each,
                            cudaMemcpyDeviceToDevice, 0) != cudaSuccess) B2_FAIL(-4);
        g_res_use[slot] = ++g_res_clock;
    }
    if (cudaMemcpy(b_act2, h_act, (size_t) act_bytes, cudaMemcpyHostToDevice) != cudaSuccess) B2_FAIL(-4);
    const int rc = (type == 16)
        ? wubu_cuda_matvec_iq2xxs(b_w2, b_act2, (float *) b_y2, K, n_cols, row_bytes, 1)
        : wubu_cuda_matvec_iq2s_v2(b_w2, b_act2, (float *) b_y2, K, n_cols, row_bytes);
    if (cudaDeviceSynchronize() != cudaSuccess) B2_FAIL(-5);
    if (cudaMemcpy(h_y, b_y2, yb, cudaMemcpyDeviceToHost) != cudaSuccess) B2_FAIL(-6);
    b2_lock.store(0);
#undef B2_FAIL
    return rc;
}

/* ---------------- IQ2_XXS, eight columns per warp ----------------
 *
 * The shipping kernel gives ONE output column to each warp, which is the structural reason it cannot use tensor cores:
 * mma.sync.m16n8k32 wants a warp to own 8 columns to fill N=8. This kernel is the tile that instruction needs, built
 * first in plain dp4a so the lane mapping and the parity can be settled independently of the instruction swap.
 *
 * Lane mapping: lane -> (group = lane>>2, entry = lane&3) still selects which 8 of a 256-value block this lane owns, but
 * the warp now holds EIGHT columns, so the same activation bytes are loaded once and reused eight times instead of being
 * re-read per column. Accumulators are acc[col][t]; the reduction stays a 32-lane butterfly per (col, t).
 *
 * UNWIRED ON PURPOSE: nothing calls this yet, and the default dispatch is untouched, so building it cannot change any
 * engine output. Parity must be established against iq2xxs_matvec_mrow_kernel on identical bytes before wiring.
 */
#define IQ2X8_MT 4          /* token rows per pass: acc is IQ2X8_MT * 8 floats, kept small to stay in registers */

__global__ void iq2xxs_matvec_8col_kernel(const uint8_t * __restrict__ w,
                                          const c_q8_k * __restrict__ xq8m,
                                          float * __restrict__ y,
                                          int64_t col_stride_bytes, int nblocks,
                                          int n_cols, int M, int q8_row_blocks)
{
    const int warps_per_block = blockDim.x / 32;
    const int warp = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int col0 = warp * 8;
    if (col0 >= n_cols) return;

    const int group = lane >> 2;      /* 0..7  */
    const int entry = lane & 3;       /* 0..3  */
    const int ncol_here = (n_cols - col0) < 8 ? (n_cols - col0) : 8;

    for (int m0 = 0; m0 < M; m0 += IQ2X8_MT) {
        const int mt = (M - m0) < IQ2X8_MT ? (M - m0) : IQ2X8_MT;
        float acc[8][IQ2X8_MT];
        for (int c = 0; c < 8; ++c)
            for (int t = 0; t < IQ2X8_MT; ++t) acc[c][t] = 0.0f;

        for (int b = 0; b < nblocks; ++b) {
            /* activation for this block, once for all eight columns */
            int32_t a0[IQ2X8_MT], a1[IQ2X8_MT];
            float   ad[IQ2X8_MT];
            const c_q8_k *rowbase = xq8m + (int64_t) m0 * q8_row_blocks;
            for (int t = 0; t < mt; ++t) {
                const c_q8_k *q8b = rowbase + (int64_t) t * q8_row_blocks + b;
                const int8_t *q8 = q8b->qs + group * 32 + entry * 8;
                memcpy(&a0[t], q8 + 0, 4);
                memcpy(&a1[t], q8 + 4, 4);
                ad[t] = q8b->d;
            }

            for (int c = 0; c < ncol_here; ++c) {
                const uint8_t *blk = w + (int64_t)(col0 + c) * col_stride_bytes + (int64_t) b * 66;
                const uint16_t d_bits = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
                const uint8_t *aux = blk + 2 + 8 * group;
                uint32_t w0, w1;
                memcpy(&w0, aux, 4);
                memcpy(&w1, aux + 4, 4);
                const uint8_t gidx = ((const uint8_t *) &w0)[entry];
                const uint8_t signs = c_ksigns_iq2xs[(w1 >> (7 * entry)) & 127];
                const uint8_t *grid = (const uint8_t *)(c_iq2xxs_grid + gidx);
                int32_t g0, g1;
                memcpy(&g0, grid, 4);
                memcpy(&g1, grid + 4, 4);
                const uint32_t m_lo = (((uint32_t)(signs & 0x0F) * 0x00204081u) & 0x01010101u) * 0xFFu;
                const uint32_t m_hi = (((uint32_t)(signs >> 4) * 0x00204081u) & 0x01010101u) * 0xFFu;
                g0 = (int32_t)__vsub4((uint32_t)g0 ^ m_lo, m_lo);
                g1 = (int32_t)__vsub4((uint32_t)g1 ^ m_hi, m_hi);
                const float wd = f16_fast(d_bits) * (0.5f + (float)(w1 >> 28)) * 0.25f;
                for (int t = 0; t < mt; ++t) {
                    const int sumi = __dp4a(g0, a0[t], 0) + __dp4a(g1, a1[t], 0);
                    acc[c][t] += wd * ad[t] * (float) sumi;
                }
            }
        }

        for (int c = 0; c < ncol_here; ++c) {
            for (int t = 0; t < mt; ++t) {
                float v = acc[c][t];
                for (int off = 16; off > 0; off >>= 1)
                    v += __shfl_down_sync(0xffffffffu, v, off);
                if (lane == 0) y[(int64_t)(m0 + t) * n_cols + col0 + c] = v;
            }
        }
    }
}

/* Parity gate for the 8-column kernel: run the shipping mrow kernel and iq2xxs_matvec_8col_kernel on the SAME device
 * bytes, bring both results back and report the largest difference. Wired to nothing; this exists so the tile can be
 * proven equivalent (or not) before any dispatch decision is made.
 * Deterministic synthetic data with a finite reference: fp16 scales forced to 1.0h so the comparison is about values. */
WUBU_EXPORT int wubu_cuda_iq2x8_check(int K, int n_cols, int M, double *out_maxdiff, double *out_maxref,
                                     double *out_ms_mrow, double *out_ms_8col)
{
    if (K <= 0 || n_cols <= 0 || M <= 0 || (K % QK_K) != 0) return -1;
    const int nblocks = K / QK_K;
    const size_t row_bytes = (size_t) nblocks * 66;
    const size_t wbytes = row_bytes * (size_t) n_cols;
    const size_t nq8 = (size_t) M * (size_t) nblocks;

    uint8_t *h_w = (uint8_t *) malloc(wbytes);
    c_q8_k  *h_q8 = (c_q8_k *) malloc(nq8 * sizeof(c_q8_k));
    float   *h_a = (float *) malloc((size_t) M * (size_t) n_cols * sizeof(float));
    float   *h_b = (float *) malloc((size_t) M * (size_t) n_cols * sizeof(float));
    if (!h_w || !h_q8 || !h_a || !h_b) { free(h_w); free(h_q8); free(h_a); free(h_b); return -2; }

    uint32_t st = 987654321u;
    for (size_t i = 0; i < wbytes; ++i) { st = st * 1664525u + 1013904223u; h_w[i] = (uint8_t) (st >> 24); }
    for (size_t b = 0; b + 66 <= wbytes; b += 66) { h_w[b] = 0x00; h_w[b + 1] = 0x3C; }   /* fp16 d = 1.0 */
    for (size_t i = 0; i < nq8; ++i) {
        float d = 0.25f;
        memcpy(&h_q8[i].d, &d, 4);
        for (int k = 0; k < 256; ++k) h_q8[i].qs[k] = (int8_t) ((k % 17) - 8);
    }

    void *d_w = NULL, *d_q8 = NULL, *d_a = NULL, *d_b = NULL;
    int rc = -3;
    if (cudaMalloc(&d_w, wbytes) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_q8, nq8 * sizeof(c_q8_k)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_a, (size_t) M * n_cols * sizeof(float)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_b, (size_t) M * n_cols * sizeof(float)) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(d_w, h_w, wbytes, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(d_q8, h_q8, nq8 * sizeof(c_q8_k), cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;

    {
        const int threads = 256, wpb = threads / 32;
        const int b_mrow = (n_cols + wpb - 1) / wpb;
        int b_8c = (n_cols / 8 + wpb - 1) / wpb;
        if (b_8c < 1) b_8c = 1;

        /* one untimed pass each, so first-touch faults and module load are not charged to either */ 
        iq2xxs_matvec_mrow_kernel<<<b_mrow, threads>>>((const uint8_t *) d_w, (const c_q8_k *) d_q8, (float *) d_a,
                                                      (int64_t) row_bytes, nblocks, n_cols, M, nblocks);
        iq2xxs_matvec_8col_kernel<<<b_8c, threads>>>((const uint8_t *) d_w, (const c_q8_k *) d_q8, (float *) d_b,
                                                     (int64_t) row_bytes, nblocks, n_cols, M, nblocks);
        if (cudaDeviceSynchronize() != cudaSuccess) { rc = -4; goto cleanup; }

        cudaEvent_t e0, e1;
        if (cudaEventCreate(&e0) != cudaSuccess || cudaEventCreate(&e1) != cudaSuccess) { rc = -5; goto cleanup; }
        cudaEventRecord(e0, 0);
        iq2xxs_matvec_mrow_kernel<<<b_mrow, threads>>>((const uint8_t *) d_w, (const c_q8_k *) d_q8, (float *) d_a,
                                                      (int64_t) row_bytes, nblocks, n_cols, M, nblocks);
        cudaEventRecord(e1, 0);
        if (cudaEventSynchronize(e1) != cudaSuccess) { cudaEventDestroy(e0); cudaEventDestroy(e1); rc = -4; goto cleanup; }
        float ms_m = 0.0f;
        cudaEventElapsedTime(&ms_m, e0, e1);

        cudaEventRecord(e0, 0);
        iq2xxs_matvec_8col_kernel<<<b_8c, threads>>>((const uint8_t *) d_w, (const c_q8_k *) d_q8, (float *) d_b,
                                                    (int64_t) row_bytes, nblocks, n_cols, M, nblocks);
        cudaEventRecord(e1, 0);
        if (cudaEventSynchronize(e1) != cudaSuccess) { cudaEventDestroy(e0); cudaEventDestroy(e1); rc = -4; goto cleanup; }
        float ms_8 = 0.0f;
        cudaEventElapsedTime(&ms_8, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);

        if (out_ms_mrow) *out_ms_mrow = (double) ms_m;
        if (out_ms_8col) *out_ms_8col = (double) ms_8;
    }
    if (cudaMemcpy(h_a, d_a, (size_t) M * n_cols * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(h_b, d_b, (size_t) M * n_cols * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;

    {
        double mx = 0.0, mr = 0.0;
        const size_t n = (size_t) M * (size_t) n_cols;
        for (size_t i = 0; i < n; ++i) {
            const double d = (double) h_a[i] - (double) h_b[i];
            const double ad = d < 0 ? -d : d;
            const double ar = h_a[i] < 0 ? -(double) h_a[i] : (double) h_a[i];
            if (ad > mx) mx = ad;
            if (ar > mr) mr = ar;
        }
        if (out_maxdiff) *out_maxdiff = mx;
        if (out_maxref)  *out_maxref  = mr;
        rc = 0;
    }
cleanup:
    if (d_w) cudaFree(d_w);
    if (d_q8) cudaFree(d_q8);
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    free(h_w); free(h_q8); free(h_a); free(h_b);
    return rc;
}

/* Batched residency entry: n_tok activation rows against ONE resident weight slab.
 *
 * The single-row entry has no batch dimension, so every projection the engine issues arrives as M = 1 and no kernel that
 * needs a real batch (the 8-column tile at 1.73x, or any future mma.sync path) can ever be reached from it. This entry
 * exists to give the call sites somewhere to hand a token block: it stages n_tok contiguous q8_K rows into one device
 * buffer, runs the device matvec with M = n_tok - which is what selects the 8-column tile - and copies n_tok result rows
 * back. Slab residency, identity keying and eviction are unchanged: they are the same g_res_* tables, same ident key.
 *
 * Unwired on purpose. Nothing calls this yet, so the oracle cannot move because of it.
 * Returns -7 if the slab cannot be made resident, so a caller can fall back rather than silently compute garbage. */
WUBU_EXPORT int wubu_cuda_resident_matvec_nt(int type, int64_t ident, const void *h_w, size_t wbytes,
                                             const void *h_act, int act_row_bytes, float *h_y,
                                             int K, int n_cols, int64_t col_stride_bytes, int n_tok)
{
    if (!h_w || !h_act || !h_y || K <= 0 || n_cols <= 0 || n_tok <= 0 || wbytes == 0 || act_row_bytes <= 0) return -1;
    if (type != 16 && type != 22) return -1;                 /* the two types the batched device path covers */
    if (cudaSetDevice(0) != cudaSuccess) return -2;

    static std::atomic<int> nt_lock{0};
    int e = 0;
    while (!nt_lock.compare_exchange_weak(e, 1, std::memory_order_acquire)) e = 0;
#define NT_FAIL(v) do { nt_lock.store(0); return (v); } while (0)

    /* 1. make the slab resident (fill on miss, exactly like the single-row path) */
    int slot = -1;
    for (int j = 0; j < g_res_n; ++j)
        if (g_res_d[j] && g_res_ident[j] == ident && g_res_bytes[j] == wbytes) { slot = j; break; }
    if (slot < 0) {
        const int room = (g_res_n < WUBU_RES_SLOTS) && (!g_res_budget || g_res_used + wbytes <= g_res_budget);
        if (!room) NT_FAIL(-7);
        slot = g_res_n;
        if (cudaMalloc(&g_res_d[slot], wbytes) != cudaSuccess) { g_res_d[slot] = NULL; NT_FAIL(-7); }
        if (cudaMemcpy(g_res_d[slot], h_w, wbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(g_res_d[slot]); g_res_d[slot] = NULL; NT_FAIL(-7);
        }
        g_res_n++;
        g_res_key[slot] = h_w;
        g_res_bytes[slot] = wbytes;
        g_res_ident[slot] = ident;
        g_res_used += wbytes;
        g_res_miss++;
    }
    g_res_use[slot] = ++g_res_clock;
    g_res_hit++;

    /* 2. stage n_tok activation rows and run the device matvec with a real batch */
    static void *nt_a, *nt_y;
    static size_t nt_ac, nt_yc;
    const size_t ab = (size_t) act_row_bytes * (size_t) n_tok;
    const size_t yb = (size_t) n_cols * sizeof(float) * (size_t) n_tok;
    if (ab > nt_ac) { if (nt_a) { cudaFree(nt_a); nt_a = NULL; } if (cudaMalloc(&nt_a, ab) != cudaSuccess) NT_FAIL(-3); nt_ac = ab; }
    if (yb > nt_yc) { if (nt_y) { cudaFree(nt_y); nt_y = NULL; } if (cudaMalloc(&nt_y, yb) != cudaSuccess) NT_FAIL(-3); nt_yc = yb; }
    if (cudaMemcpy(nt_a, h_act, ab, cudaMemcpyHostToDevice) != cudaSuccess) NT_FAIL(-4);

    int rc;
    if (type == 16) {
        rc = wubu_cuda_matvec_iq2xxs(g_res_d[slot], nt_a, (float *) nt_y, K, n_cols, col_stride_bytes, n_tok);
    } else {
        rc = wubu_cuda_matvec_iq2s_v2(g_res_d[slot], nt_a, (float *) nt_y, K, n_cols, col_stride_bytes);
    }
    if (rc != 0) NT_FAIL(-5);
    if (cudaMemcpy(h_y, nt_y, yb, cudaMemcpyDeviceToHost) != cudaSuccess) NT_FAIL(-6);
    nt_lock.store(0);
#undef NT_FAIL
    return 0;
}

/* ---- f16 DENSE LANE: resident single-f16 matvec via cuBLAS (f16 x f16 -> f32 accum). ----
 * The fast lane: the engine hands us the PRE-dequantized f16 weight (from the materialized set), we hold it
 * resident keyed by ident and run a pure cuBLAS GEMM at M=n_cols, N=1, K. No per-token dequant. Its own cache
 * namespace (g_f16_*) so it never collides with the quantized resident slots. */
static void   *g_f16_d[4096];
static int64_t g_f16_id[4096];
static size_t  g_f16_b[4096];
static int     g_f16_n = 0;
static cublasHandle_t g_f16_h = NULL;
static __half *g_f16_sx = NULL; static int g_f16_sx_cap = 0;
static float  *g_f16_sy = NULL; static int g_f16_sy_cap = 0;
WUBU_EXPORT int wubu_cuda_resident_f16_matvec(int64_t ident, const void *h_w_f16, size_t wbytes,
    const float *h_act, int K, float *h_y, int n_cols)
{
    if (!h_w_f16 || !h_act || !h_y || wbytes == 0 || K <= 0 || n_cols <= 0) return -1;
    if (!g_f16_h) { if (cublasCreate(&g_f16_h) != CUBLAS_STATUS_SUCCESS) { g_f16_h = NULL; return -2; } }
    int slot = -1;
    for (int i = 0; i < g_f16_n; ++i) if (g_f16_id[i] == ident && g_f16_b[i] == wbytes) { slot = i; break; }
    if (slot < 0) {
        if (g_f16_n >= 4096) return -3;
        slot = g_f16_n;
        if (cudaMalloc(&g_f16_d[slot], wbytes) != cudaSuccess) { g_f16_d[slot] = NULL; return -4; }
        if (cudaMemcpy(g_f16_d[slot], h_w_f16, wbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(g_f16_d[slot]); g_f16_d[slot] = NULL; return -5;
        }
        g_f16_id[slot] = ident; g_f16_b[slot] = wbytes; g_f16_n++;
    }
    if (K > g_f16_sx_cap) { if (g_f16_sx) cudaFree(g_f16_sx); if (cudaMalloc(&g_f16_sx, (size_t)K * 2) != cudaSuccess) { g_f16_sx = NULL; g_f16_sx_cap = 0; return -6; } g_f16_sx_cap = K; }
    if (n_cols > g_f16_sy_cap) { if (g_f16_sy) cudaFree(g_f16_sy); if (cudaMalloc(&g_f16_sy, (size_t)n_cols * 4) != cudaSuccess) { g_f16_sy = NULL; g_f16_sy_cap = 0; return -7; } g_f16_sy_cap = n_cols; }
    /* convert the f32 activation to f16 (small: K elements) and upload */
    static __half *hx = NULL; static int hx_cap = 0;
    if (K > hx_cap) { free(hx); hx = (__half *) malloc((size_t)K * 2); hx_cap = K; }
    if (!hx) return -8;
    for (int i = 0; i < K; ++i) hx[i] = __float2half_rn(h_act[i]);
    if (cudaMemcpy(g_f16_sx, hx, (size_t)K * 2, cudaMemcpyHostToDevice) != cudaSuccess) return -9;
    /* CUBLAS_COMPUTE_32F requires alpha/beta as FLOAT pointers, not half - a half* here reads 4 bytes from 2
     * and yields NaN (caught by the standalone DLL test, y came back all-NaN). */
    float one = 1.0f, zero = 0.0f;
    /* W is [out][in] row-major (= col-major K x n_cols, lda=K); OP_T -> C = W * x. */
    cublasStatus_t st = cublasGemmEx(g_f16_h, CUBLAS_OP_T, CUBLAS_OP_N, n_cols, 1, K,
        &one, g_f16_d[slot], CUDA_R_16F, K, g_f16_sx, CUDA_R_16F, K,
        &zero, g_f16_sy, CUDA_R_32F, n_cols, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) return -10;
    cudaDeviceSynchronize();  /* cublas may run on its own stream; order the copy back explicitly */
    if (cudaMemcpy(h_y, g_f16_sy, (size_t)n_cols * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return -11;
    return 0;
}
