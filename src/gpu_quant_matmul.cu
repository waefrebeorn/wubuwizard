/**
 * gpu_quant_matmul.cu — Quantized GPU matmul for Q5_K/Q6_K/IQ1_M types.
 *
 * Weight layout: GGUF stores weights with dims[0]=C (output dim, innermost).
 * Each column of D input-dim elements is stored as ceil(D/256) blocks.
 * y[col] = sum_r x[r] * deq(W[col][r])  — column-major weight access.
 */
#include "gpu_quant_matmul.h"
#include "gguf_reader.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

/* nvcc with the MSVC host compiler has no __builtin_memcpy. */
#if defined(_MSC_VER) && !defined(__builtin_memcpy)
#define __builtin_memcpy memcpy
#endif

#define QK_K 256

// IQ1_S grid table (2048 × uint64) — needed for IQ1_M dequant
// Matches gguf_reader.c iq1s_grid exactly
#define NGRID_IQ1S 2048
#define IQ1S_DELTA 0.125f
__constant__ uint64_t d_iq1s_grid[NGRID_IQ1S];

static __device__ __inline__ float fp16_to_fp32_dev(uint16_t v) {
    int sign = (v >> 15) & 1, exp = (v >> 10) & 0x1F, mant = v & 0x03FF;
    uint32_t f32;
    if (exp == 0) {
        // F16 denormal: value = (-1)^sign * mant * 2^(-24).
        // Match CPU f16_to_f32() — normalize as F32 denorm+1, then subtract 2^-14.
        uint32_t norm = (sign << 31) | ((1 + 112) << 23) | (mant << 13);
        float nv; __builtin_memcpy(&nv, &norm, 4);
        return sign ? nv + 6.103515625e-5f : nv - 6.103515625e-5f;
    } else if (exp == 31) {
        f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        f32 = (sign << 31) | ((uint32_t)(127 - 15 + exp) << 23) | (mant << 13);
    }
    float r; __builtin_memcpy(&r, &f32, 4); return r;
}

static __device__ __inline__ void get_scale_min_k4_dev(int j, const uint8_t *q,
                                                       uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
           *m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }
}

// ================================================================
// Q5_K Matmul — fused dequant+dot, no bv[256] local array spill.
// y[col] = sum_r x[r] * deq(W[col][r])
// Each thread handles 1 column. Dequant in 32-element sub-blocks,
// accumulating dot product immediately (no large stack buffer).
// ================================================================
__global__ void quant_matmul_q5_k_kernel(const float *x, const uint8_t *W_q,
                                          float *y, int n_rows, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols) return;
    const int BPC = (n_rows + QK_K - 1) / QK_K;  // blocks per column
    const uint8_t *base = W_q + (int64_t)col * BPC * 176;
    double sum = 0.0;
    for (int b = 0; b < BPC; b++) {
        const uint8_t *block = base + (int64_t)b * 176;
        float d = fp16_to_fp32_dev(*(const uint16_t *)block);
        float dmin = fp16_to_fp32_dev(*(const uint16_t *)(block + 2));
        const uint8_t *sc = block + 4, *qh = block + 16, *qs = block + 48;
        int rem = n_rows - b * QK_K;
        if (rem > QK_K) rem = QK_K;
        int is = 0;
        int x_off = b * QK_K;
        for (int j = 0; j < rem; j += 64) {
            uint8_t s1, m1, s2, m2;
            get_scale_min_k4_dev(is+0, sc, &s1, &m1);
            get_scale_min_k4_dev(is+1, sc, &s2, &m2);
            float d1 = d*s1, ml1 = dmin*m1, d2 = d*s2, ml2 = dmin*m2;
            int qb = j/2, ci = j/64;
            int lim = (j + 32 < rem) ? 32 : rem - j;
            for (int l = 0; l < lim; l++) {
                uint8_t lo = qs[qb + l];
                uint8_t hi0 = (qh[l] >> (ci*2 + 0)) & 1;
                uint8_t hi1 = (qh[l] >> (ci*2 + 1)) & 1;
                float v0 = d1 * ((lo & 0x0F) + (hi0 ? 16 : 0)) - ml1;
                sum += (double)x[x_off + j + l] * (double)v0;
                if (j + 32 + l < rem) {
                    float v1 = d2 * ((lo >> 4)   + (hi1 ? 16 : 0)) - ml2;
                    sum += (double)x[x_off + j + 32 + l] * (double)v1;
                }
            }
            is += 2;
        }
    }
    y[col] = (float)sum;
}

// ================================================================
// Q6_K Matmul — fused dequant+dot, no bv[256] local array spill.
// Each thread handles 1 column. Dequant in 32-element sub-groups,
// accumulating dot product immediately.
// ================================================================
__global__ void quant_matmul_q6_k_kernel(const float *x, const uint8_t *W_q,
                                          float *y, int n_rows, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols) return;
    const int BPC = (n_rows + QK_K - 1) / QK_K;
    const uint8_t *base = W_q + (int64_t)col * BPC * 210;
    double sum = 0.0;
    for (int b = 0; b < BPC; b++) {
        const uint8_t *block = base + (int64_t)b * 210;
        float d = fp16_to_fp32_dev(*(const uint16_t *)(block + 208));
        const uint8_t *ql = block, *qh = block + 128;
        const int8_t *sc = (const int8_t *)(block + 192);
        int rem = n_rows - b * QK_K;
        if (rem > QK_K) rem = QK_K;
        for (int ni = 0; ni < rem; ni += 128) {
            // Process 128-element chunk in 32 iterations, each producing 4 vals
            for (int l = 0; l < 32; l++) {
                int is = l / 16;  // sc cycle: [0-3] for first half, [4-7] for second
                int idx0 = b * QK_K + ni + l;
                int idx1 = b * QK_K + ni + 32 + l;
                int idx2 = b * QK_K + ni + 64 + l;
                int idx3 = b * QK_K + ni + 96 + l;
                uint8_t l0 = ql[l + 0];
                uint8_t l32 = ql[l + 32];
                uint8_t h = qh[l];
                if (idx0 < b * QK_K + rem) {
                    int8_t v6 = (int8_t)((l0 & 0xF) | ((h >> 0) & 3) << 4);
                    sum += (double)x[idx0] * (double)d * (double)sc[is+0] * (double)(v6 - 32);
                }
                if (idx1 < b * QK_K + rem) {
                    int8_t v6 = (int8_t)((l32 & 0xF) | ((h >> 2) & 3) << 4);
                    sum += (double)x[idx1] * (double)d * (double)sc[is+2] * (double)(v6 - 32);
                }
                if (idx2 < b * QK_K + rem) {
                    int8_t v6 = (int8_t)((l0 >> 4) | ((h >> 4) & 3) << 4);
                    sum += (double)x[idx2] * (double)d * (double)sc[is+4] * (double)(v6 - 32);
                }
                if (idx3 < b * QK_K + rem) {
                    int8_t v6 = (int8_t)((l32 >> 4) | ((h >> 6) & 3) << 4);
                    sum += (double)x[idx3] * (double)d * (double)sc[is+6] * (double)(v6 - 32);
                }
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
    y[col] = (float)sum;
}

// ================================================================
// Q4_K Matmul — fused dequant+dot, NO qh field.
// y[col] = sum_r x[r] * deq(W[col][r])
// Each thread handles 1 column. 256 elements/block, 144 bytes/block.
// ================================================================
__global__ void quant_matmul_q4_k_kernel(const float *x, const uint8_t *W_q,
                                          float *y, int n_rows, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols) return;
    const int BPC = (n_rows + QK_K - 1) / QK_K;
    const uint8_t *base = W_q + (int64_t)col * BPC * 144;
    double sum = 0.0;
    for (int b = 0; b < BPC; b++) {
        const uint8_t *block = base + (int64_t)b * 144;
        float d = fp16_to_fp32_dev(*(const uint16_t *)block);
        float dmin = fp16_to_fp32_dev(*(const uint16_t *)(block + 2));
        const uint8_t *sc = block + 4;
        const uint8_t *qs = block + 16;  // no qh field
        int rem = n_rows - b * QK_K;
        if (rem > QK_K) rem = QK_K;
        int is = 0;
        int x_off = b * QK_K;
        for (int j = 0; j < rem; j += 64) {
            uint8_t s1, m1, s2, m2;
            get_scale_min_k4_dev(is+0, sc, &s1, &m1);
            get_scale_min_k4_dev(is+1, sc, &s2, &m2);
            float d1 = d*s1, ml1 = dmin*m1, d2 = d*s2, ml2 = dmin*m2;
            int qb = j/2;
            int lim = (j + 32 < rem) ? 32 : rem - j;
            for (int l = 0; l < lim; l++) {
                uint8_t lo = qs[qb + l];
                float v0 = d1 * (lo & 0x0F) - ml1;
                sum += (double)x[x_off + j + l] * (double)v0;
                if (j + 32 + l < rem) {
                    float v1 = d2 * (lo >> 4) - ml2;
                    sum += (double)x[x_off + j + 32 + l] * (double)v1;
                }
            }
            is += 2;
        }
    }
    y[col] = (float)sum;
}

// ================================================================
// IQ1_M Matmul — fused dequant+dot using grid lookup.
// Block: 56 bytes per 256 elements. qs[32] + qh[16] + scales[8].
// Each thread handles 1 column. Dequant in 32-element sub-blocks
// via 8-element grid lookups, accumulating dot product immediately.
// ================================================================
__global__ void quant_matmul_iq1_m_kernel(const float *x, const uint8_t *W_q,
                                           float *y, int n_rows, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols) return;
    const int BPC = (n_rows + QK_K - 1) / QK_K;
    const uint8_t *base = W_q + (int64_t)col * BPC * 56;
    double sum = 0.0;
    for (int b = 0; b < BPC; b++) {
        const uint8_t *block = base + (int64_t)b * 56;
        const uint8_t *qs = block;                  // 32 bytes
        const uint8_t *qh_b = block + 32;           // 16 bytes
        const uint16_t *sc = (const uint16_t *)(block + 48); // 4 uint16_t = 8 bytes
        // Global fp16 scale from high nibbles
        uint16_t scale_bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                              ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        float d = fp16_to_fp32_dev(scale_bits);
        int rem = n_rows - b * QK_K;
        if (rem > QK_K) rem = QK_K;
        int x_off = b * QK_K;
        for (int ib = 0; ib < 8; ib++) {
            // Two 3-bit sub-scales per ib
            float dl1 = d * (2.0f * (float)((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1.0f);
            float dl2 = d * (2.0f * (float)((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1.0f);
            // 11-bit grid indices
            int qs_off = ib * 4;
            int qh_off = ib;
            uint8_t qh0 = qh_b[qh_off * 2 + 0];
            uint8_t qh1 = qh_b[qh_off * 2 + 1];
            uint16_t idx0 = qs[qs_off + 0] | ((uint16_t)(qh0 << 8) & 0x700);
            uint16_t idx1 = qs[qs_off + 1] | ((uint16_t)(qh0 << 4) & 0x700);
            uint16_t idx2 = qs[qs_off + 2] | ((uint16_t)(qh1 << 8) & 0x700);
            uint16_t idx3 = qs[qs_off + 3] | ((uint16_t)(qh1 << 4) & 0x700);
            // Delta signs
            float delta0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            float delta1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            float delta2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            float delta3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            // First two 8-element groups: dl1
            int eb_off = x_off + ib * 32;
            if (eb_off + 0 * 8 < b * QK_K + rem) {
                uint64_t g0 = d_iq1s_grid[idx0], g1 = d_iq1s_grid[idx1];
                const int8_t *gv0 = (const int8_t *)&g0;
                const int8_t *gv1 = (const int8_t *)&g1;
                for (int j = 0; j < 8; j++) {
                    int idx_off0 = eb_off + j;
                    if (idx_off0 < b * QK_K + rem)
                        sum += (double)x[idx_off0] * (double)(dl1 * ((float)gv0[j] + delta0));
                    int idx_off1 = eb_off + 8 + j;
                    if (idx_off1 < b * QK_K + rem)
                        sum += (double)x[idx_off1] * (double)(dl1 * ((float)gv1[j] + delta1));
                }
            }
            // Last two 8-element groups: dl2
            if (eb_off + 2 * 8 < b * QK_K + rem) {
                uint64_t g2 = d_iq1s_grid[idx2], g3 = d_iq1s_grid[idx3];
                const int8_t *gv2 = (const int8_t *)&g2;
                const int8_t *gv3 = (const int8_t *)&g3;
                for (int j = 0; j < 8; j++) {
                    int idx_off2 = eb_off + 16 + j;
                    if (idx_off2 < b * QK_K + rem)
                        sum += (double)x[idx_off2] * (double)(dl2 * ((float)gv2[j] + delta2));
                    int idx_off3 = eb_off + 24 + j;
                    if (idx_off3 < b * QK_K + rem)
                        sum += (double)x[idx_off3] * (double)(dl2 * ((float)gv3[j] + delta3));
                }
            }
        }
    }
    y[col] = (float)sum;
}
// ================================================================
// Batched quant matmul — processes multiple tokens in parallel
// x: [C, n_rows], y: [C, n_cols]
__global__ void quant_matmul_q5_k_batched(const float *x, const uint8_t *W_q,
                                          float *y, int C, int n_rows, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tok = blockIdx.y;
    if (col >= n_cols || tok >= C) return;

    const float *x_tok = x + (int64_t)tok * n_rows;
    float *y_tok = y + (int64_t)tok * n_cols;
    const int BPC = (n_rows + QK_K - 1) / QK_K;
    const uint8_t *base = W_q + (int64_t)col * BPC * 176;

    double sum = 0.0;
    for (int b = 0; b < BPC; b++) {
        const uint8_t *block = base + (int64_t)b * 176;
        float d = fp16_to_fp32_dev(*(const uint16_t *)block);
        float dmin = fp16_to_fp32_dev(*(const uint16_t *)(block + 2));
        const uint8_t *sc = block + 4, *qh = block + 16, *qs = block + 48;
        int rem = n_rows - b * QK_K;
        if (rem > QK_K) rem = QK_K;
        int is = 0;
        int x_off = b * QK_K;
        for (int j = 0; j < rem; j += 64) {
            uint8_t s1, m1, s2, m2;
            get_scale_min_k4_dev(is+0, sc, &s1, &m1);
            get_scale_min_k4_dev(is+1, sc, &s2, &m2);
            float d1 = d*s1, ml1 = dmin*m1, d2 = d*s2, ml2 = dmin*m2;
            int qb = j/2, ci = j/64;
            int lim = (j + 32 < rem) ? 32 : rem - j;
            for (int l = 0; l < lim; l++) {
                uint8_t lo = qs[qb + l];
                uint8_t hi0 = (qh[l] >> (ci*2 + 0)) & 1;
                uint8_t hi1 = (qh[l] >> (ci*2 + 1)) & 1;
                float v0 = d1 * ((lo & 0x0F) + (hi0 ? 16 : 0)) - ml1;
                sum += (double)x_tok[x_off + j + l] * (double)v0;
                if (j + 32 + l < rem) {
                    float v1 = d2 * ((lo >> 4)   + (hi1 ? 16 : 0)) - ml2;
                    sum += (double)x_tok[x_off + j + 32 + l] * (double)v1;
                }
            }
            is += 2;
        }
    }
    y_tok[col] = (float)sum;
}

__global__ void quant_matmul_q6_k_batched(const float *x, const uint8_t *W_q,
                                          float *y, int C, int n_rows, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tok = blockIdx.y;
    if (col >= n_cols || tok >= C) return;

    const float *x_tok = x + (int64_t)tok * n_rows;
    float *y_tok = y + (int64_t)tok * n_cols;
    const int BPC = (n_rows + QK_K - 1) / QK_K;
    const uint8_t *base = W_q + (int64_t)col * BPC * 210;

    double sum = 0.0;
    for (int b = 0; b < BPC; b++) {
        const uint8_t *block = base + (int64_t)b * 210;
        float d = fp16_to_fp32_dev(*(const uint16_t *)(block + 208));
        const uint8_t *ql = block, *qh = block + 128;
        const int8_t *sc = (const int8_t *)(block + 192);
        int rem = n_rows - b * QK_K;
        if (rem > QK_K) rem = QK_K;
        for (int ni = 0; ni < rem; ni += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int idx0 = b * QK_K + ni + l;
                int idx1 = b * QK_K + ni + 32 + l;
                int idx2 = b * QK_K + ni + 64 + l;
                int idx3 = b * QK_K + ni + 96 + l;
                uint8_t l0 = ql[l + 0];
                uint8_t l32 = ql[l + 32];
                uint8_t h = qh[l];
                if (idx0 < b * QK_K + rem) {
                    int8_t v6 = (int8_t)((l0 & 0xF) | ((h >> 0) & 3) << 4);
                    sum += (double)x_tok[idx0] * (double)d * (double)sc[is+0] * (double)(v6 - 32);
                }
                if (idx1 < b * QK_K + rem) {
                    int8_t v6 = (int8_t)((l32 & 0xF) | ((h >> 2) & 3) << 4);
                    sum += (double)x_tok[idx1] * (double)d * (double)sc[is+2] * (double)(v6 - 32);
                }
                if (idx2 < b * QK_K + rem) {
                    int8_t v6 = (int8_t)((l0 >> 4) | ((h >> 4) & 3) << 4);
                    sum += (double)x_tok[idx2] * (double)d * (double)sc[is+4] * (double)(v6 - 32);
                }
                if (idx3 < b * QK_K + rem) {
                    int8_t v6 = (int8_t)((l32 >> 4) | ((h >> 6) & 3) << 4);
                    sum += (double)x_tok[idx3] * (double)d * (double)sc[is+6] * (double)(v6 - 32);
                }
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
    y_tok[col] = (float)sum;
}

// Batched Q4_K — processes multiple tokens in parallel
__global__ void quant_matmul_q4_k_batched(const float *x, const uint8_t *W_q,
                                           float *y, int C, int n_rows, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tok = blockIdx.y;
    if (col >= n_cols || tok >= C) return;
    const float *x_tok = x + (int64_t)tok * n_rows;
    float *y_tok = y + (int64_t)tok * n_cols;
    const int BPC = (n_rows + QK_K - 1) / QK_K;
    const uint8_t *base = W_q + (int64_t)col * BPC * 144;
    double sum = 0.0;
    for (int b = 0; b < BPC; b++) {
        const uint8_t *block = base + (int64_t)b * 144;
        float d = fp16_to_fp32_dev(*(const uint16_t *)block);
        float dmin = fp16_to_fp32_dev(*(const uint16_t *)(block + 2));
        const uint8_t *sc = block + 4;
        const uint8_t *qs = block + 16;
        int rem = n_rows - b * QK_K;
        if (rem > QK_K) rem = QK_K;
        int is = 0;
        int x_off = b * QK_K;
        for (int j = 0; j < rem; j += 64) {
            uint8_t s1, m1, s2, m2;
            get_scale_min_k4_dev(is+0, sc, &s1, &m1);
            get_scale_min_k4_dev(is+1, sc, &s2, &m2);
            float d1 = d*s1, ml1 = dmin*m1, d2 = d*s2, ml2 = dmin*m2;
            int qb = j/2;
            int lim = (j + 32 < rem) ? 32 : rem - j;
            for (int l = 0; l < lim; l++) {
                uint8_t lo = qs[qb + l];
                float v0 = d1 * (lo & 0x0F) - ml1;
                sum += (double)x_tok[x_off + j + l] * (double)v0;
                if (j + 32 + l < rem) {
                    float v1 = d2 * (lo >> 4) - ml2;
                    sum += (double)x_tok[x_off + j + 32 + l] * (double)v1;
                }
            }
            is += 2;
        }
    }
    y_tok[col] = (float)sum;
}

// Batched IQ1_M grid: dim.x = n_cols threads, dim.y = C batch
__global__ void quant_matmul_iq1_m_batched(const float *x, const uint8_t *W_q,
                                            float *y, int C, int n_rows, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tok = blockIdx.y;
    if (col >= n_cols || tok >= C) return;
    const float *x_tok = x + (int64_t)tok * n_rows;
    float *y_tok = y + (int64_t)tok * n_cols;
    const int BPC = (n_rows + QK_K - 1) / QK_K;
    const uint8_t *base = W_q + (int64_t)col * BPC * 56;
    double sum = 0.0;
    for (int b = 0; b < BPC; b++) {
        const uint8_t *block = base + (int64_t)b * 56;
        const uint8_t *qs = block;
        const uint8_t *qh_b = block + 32;
        const uint16_t *sc = (const uint16_t *)(block + 48);
        uint16_t scale_bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                              ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        float d = fp16_to_fp32_dev(scale_bits);
        int rem = n_rows - b * QK_K;
        if (rem > QK_K) rem = QK_K;
        int x_off = b * QK_K;
        for (int ib = 0; ib < 8; ib++) {
            float dl1 = d * (2.0f * (float)((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1.0f);
            float dl2 = d * (2.0f * (float)((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1.0f);
            int qs_off = ib * 4;
            int qh_off = ib;
            uint8_t qh0 = qh_b[qh_off * 2 + 0];
            uint8_t qh1 = qh_b[qh_off * 2 + 1];
            uint16_t idx0 = qs[qs_off + 0] | ((uint16_t)(qh0 << 8) & 0x700);
            uint16_t idx1 = qs[qs_off + 1] | ((uint16_t)(qh0 << 4) & 0x700);
            uint16_t idx2 = qs[qs_off + 2] | ((uint16_t)(qh1 << 8) & 0x700);
            uint16_t idx3 = qs[qs_off + 3] | ((uint16_t)(qh1 << 4) & 0x700);
            float delta0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            float delta1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            float delta2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            float delta3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            int eb_off = x_off + ib * 32;
            if (eb_off + 0 * 8 < b * QK_K + rem) {
                uint64_t g0 = d_iq1s_grid[idx0], g1 = d_iq1s_grid[idx1];
                const int8_t *gv0 = (const int8_t *)&g0;
                const int8_t *gv1 = (const int8_t *)&g1;
                for (int j = 0; j < 8; j++) {
                    int idx_off0 = eb_off + j;
                    if (idx_off0 < b * QK_K + rem)
                        sum += (double)x_tok[idx_off0] * (double)(dl1 * ((float)gv0[j] + delta0));
                    int idx_off1 = eb_off + 8 + j;
                    if (idx_off1 < b * QK_K + rem)
                        sum += (double)x_tok[idx_off1] * (double)(dl1 * ((float)gv1[j] + delta1));
                }
            }
            if (eb_off + 2 * 8 < b * QK_K + rem) {
                uint64_t g2 = d_iq1s_grid[idx2], g3 = d_iq1s_grid[idx3];
                const int8_t *gv2 = (const int8_t *)&g2;
                const int8_t *gv3 = (const int8_t *)&g3;
                for (int j = 0; j < 8; j++) {
                    int idx_off2 = eb_off + 16 + j;
                    if (idx_off2 < b * QK_K + rem)
                        sum += (double)x_tok[idx_off2] * (double)(dl2 * ((float)gv2[j] + delta2));
                    int idx_off3 = eb_off + 24 + j;
                    if (idx_off3 < b * QK_K + rem)
                        sum += (double)x_tok[idx_off3] * (double)(dl2 * ((float)gv3[j] + delta3));
                }
            }
        }
    }
    y_tok[col] = (float)sum;
}

extern "C"
int wubu_cuda_quant_matmul_batched(const float *x, int C,
    const uint8_t *W_q, int qt,
    int nr, int nc,
    float *y, cudaStream_t st) {
    int block = 256;
    dim3 grid((nc + block - 1) / block, C);
    switch (qt) {
    case GGML_TYPE_Q5_K: quant_matmul_q5_k_batched<<<grid,block,0,st>>>(x,W_q,y,C,nr,nc); return 1;
    case GGML_TYPE_Q6_K: quant_matmul_q6_k_batched<<<grid,block,0,st>>>(x,W_q,y,C,nr,nc); return 1;
    case GGML_TYPE_Q4_K: quant_matmul_q4_k_batched<<<grid,block,0,st>>>(x,W_q,y,C,nr,nc); return 1;
    case GGML_TYPE_IQ1_M: quant_matmul_iq1_m_batched<<<grid,block,0,st>>>(x,W_q,y,C,nr,nc); return 1;
    default: return 0;
    }
}

// Grid table upload — must be called once before IQ1_M kernels
extern "C" void wubu_cuda_quant_matmul_set_iq1s_grid(const uint64_t *grid) {
    cudaMemcpyToSymbol(d_iq1s_grid, grid, NGRID_IQ1S * sizeof(uint64_t));
}

// ================================================================
// Host dispatch — original single-token
// ================================================================
extern "C"
size_t wubu_cuda_quant_matmul_scratch(int nr, int nc, int qt) {
    (void)nr; (void)nc; (void)qt; return 0;
}

extern "C"
int wubu_cuda_quant_matmul(const float *x, const uint8_t *W_q, int qt,
    int nr, int nc, float *y, float *scr, size_t ss, cudaStream_t st) {
    (void)scr; (void)ss;
    int block = 256;
    int grid = (nc + block - 1) / block;
    switch (qt) {
    case GGML_TYPE_Q5_K: quant_matmul_q5_k_kernel<<<grid,block,0,st>>>(x,W_q,y,nr,nc); return 1;
    case GGML_TYPE_Q6_K: quant_matmul_q6_k_kernel<<<grid,block,0,st>>>(x,W_q,y,nr,nc); return 1;
    case GGML_TYPE_Q4_K: quant_matmul_q4_k_kernel<<<grid,block,0,st>>>(x,W_q,y,nr,nc); return 1;
    case GGML_TYPE_IQ1_M: quant_matmul_iq1_m_kernel<<<grid,block,0,st>>>(x,W_q,y,nr,nc); return 1;
    default: return 0;
    }
}
