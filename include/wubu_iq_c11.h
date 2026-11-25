/*
 * wubu_iq_c11.h -- C11 API for the AVX2 multi-token dot products of the ggml
 * i-quant expert formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS)
 * and the K-quants Q4_K / Q5_K against Q8_K activations, plus IQ4_NL against
 * Q8_0. Each token's result is bit-identical to ggml-cpu's AVX2 dot for that
 * pair: the same exact integer lanes per block and ggml's float steps in ggml's
 * order.
 *
 * Translated 1:1 from maya's src/kernels/cpu/iq_avx2.cpp (C++ -> C11). Plain C,
 * no C++, no name mangling. Integer format codes are ggml's ggml_type values:
 *
 *   12 Q4_K   13 Q5_K   16 IQ2_XXS   17 IQ2_XS   18 IQ3_XXS
 *   20 IQ4_NL (IQ4_NL down-path only)  21 IQ3_S   22 IQ2_S   23 IQ4_XS   29 IQ1_M
 *
 * AVX2 (with FMA and F16C) is required at run time: every entry point checks
 * __builtin_cpu_supports("avx2") before touching a vector register. When the CPU
 * cannot run the kernels the void entry points are no-ops that write 0.0f to
 * the requested output elements, and wubu_iq_supported()/wubu_iq_rows1_supported()
 * return 0 so callers never get that far.
 *
 * `n` is the number of ELEMENTS in a row (not blocks); the kernels compute
 * nb = n / QK_K (256) internally -- IQ4_NL uses n / QK4_NL (32).
 */
#ifndef WUBU_IQ_C11_H
#define WUBU_IQ_C11_H

#include <stddef.h>

/* 1 when ggml_type is handled AND the CPU can run the AVX2 kernels, else 0. */
int wubu_iq_supported(int ggml_type);

/* 1 when the single-token "rows1" path handles ggml_type (and the CPU is able). */
int wubu_iq_rows1_supported(int ggml_type);

/*
 * Each format's dot product consumes ONE specific activation layout -- quantise
 * the activations with ggml's matching *_ref function and nothing else:
 *
 *   ggml type  format   activation block      ggml quantiser
 *   ---------  ------   ----------------      --------------
 *    8         Q8_0     block_q8_0 (32 elem)  quantize_row_q8_0_ref
 *   12         Q4_K     block_q8_K (256)      quantize_row_q8_K_ref
 *   13         Q5_K     block_q8_K (256)      quantize_row_q8_K_ref
 *   14         Q6_K     block_q8_K (256)      quantize_row_q8_K_ref
 *   16         IQ2_XXS  block_q8_K (256)      quantize_row_q8_K_ref
 *   17         IQ2_XS   block_q8_K (256)      quantize_row_q8_K_ref
 *   18         IQ3_XXS  block_q8_K (256)      quantize_row_q8_K_ref
 *   19         IQ1_S    block_q8_K (256)      quantize_row_q8_K_ref
 *   20         IQ4_NL   block_q8_0 (32)       quantize_row_q8_0_ref   [IQ4_NL path
 *                                                only: wubu_iq4nl_down_rows; type 20
 *                                                is not a wubu_iq_rows type]
 *   21         IQ3_S    block_q8_K (256)      quantize_row_q8_K_ref
 *   22         IQ2_S    block_q8_K (256)      quantize_row_q8_K_ref
 *   23         IQ4_XS   block_q8_K (256)      quantize_row_q8_K_ref
 *   29         IQ1_M    block_q8_K (256)      quantize_row_q8_K_ref
 *   40         NVFP4    block_q8_0 (32)       quantize_row_q8_0_ref
 *   42         Q2_0     block_q8_0 (32)       quantize_row_q8_0_ref
 *
 * One NVFP4 / Q2_0 weight block spans 64 elements, i.e. two block_q8_0 blocks;
 * one Q8_0 weight block spans 32 elements, i.e. one. The activation buffer for a
 * row of n elements therefore holds n / wubu_iq_block_elems(type) blocks.
 */
int wubu_iq_block_elems(int ggml_type);

/* "q8_K" or "q8_0": which activation struct ggml_type's dot takes (matching
 * wubu_iq_block_elems), or NULL when the type is not handled here. */
const char * wubu_iq_act_kind(int ggml_type);

/*
 * out[t][r] = w_r . a[t] for rows [r0, r1); w_r lives at w + r * row_bytes,
 * a[t] is a block_q8_K array of n/256 blocks.
 */
void wubu_iq_rows(int ggml_type, const void *w, size_t row_bytes, int n,
                  const void *const *act, int nt, float *const *out, int r0, int r1);

/* Same arguments and exactness, decoding the weights with the cheaper decode
 * step (IQ2_XS, IQ3_XXS, IQ3_S, IQ1_M; IQ2_XXS opt-in via MAYA_IQ2XXS_1T=1). */
void wubu_iq_rows1(int ggml_type, const void *w, size_t row_bytes, int n,
                   const void *const *act, int nt, float *const *out, int r0, int r1);

/*
 * ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows [r0, r1); the gate
 * rows are at blob, the up rows at blob + up_off, both strided by gu_row bytes.
 */
void wubu_iq_gu_rows(int ggml_type, const void *blob, size_t gu_row, size_t up_off,
                     int n, const void *const *act, int nt, float *const *ff,
                     int r0, int r1);

/*
 * out[t][r] = w_r . h[t] for IQ4_NL (type 20) rows against Q8_0 activations
 * (ggml's block_q8_0); n is a multiple of QK4_NL (32).
 */
void wubu_iq4nl_down_rows(const void *w, size_t row_bytes, int n,
                          const void *const *hq, int nt, float *const *out,
                          int r0, int r1);

/*
 * Q5_K's min term: fused (fmaf) when 1, the two rounding steps when 0. Both are
 * exact forms; the default is fused (see iq_avx512.cpp in maya).
 */
void wubu_iq_set_q5k_fused(int fused);

#endif /* WUBU_IQ_C11_H */
