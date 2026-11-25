/* int8 activation blocks shared by the IQ/i-quant row kernels (C11 + AVX2).
 *
 * One 32-element group carries its own scale, matching the 32-element sub-blocks that IQ2_XS applies its
 * sub-scales to, so a weight group and an activation group line up without re-scaling per element.
 * Quantise ONCE per matvec and pass the same array to every row of the expert: that amortisation over 2304 rows
 * is where the speedup lives, not in the kernel itself.
 */
#ifndef WUBU_IQ2Q8_H
#define WUBU_IQ2Q8_H

#include <stdint.h>

typedef struct {
    int8_t q[32];
    float  scale;
} in2_act_q8;

/* block_q8_K: what the K-quants and the IQ2/IQ3/IQ1 kernels consume. 256 elements, one float scale. */
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } in2_q8k_blk;

/* block_q8_0: what Q8_0, Q2_0 and NVFP4 consume. 32 elements, FP16 scale. The two are NOT interchangeable and
 * feeding the wrong one yields plausible numbers rather than a crash, so the kind is explicit at the call site. */
typedef struct { uint16_t d; int8_t qs[32]; } in2_q8_0_blk;

uint16_t in2_f32_to_f16(float f);
float    in2_f16_to_f32(uint16_t h);

void  in2_act_quantise(const float *x, int64_t n_groups, in2_act_q8 *out);
void  in2_q8k_quantise(const float *x, int64_t n_groups, in2_q8k_blk *out);
void  in2_q8_0_quantise(const float *x, int64_t n_groups, in2_q8_0_blk *out);
float dot_row_iq2_xs_q8(const uint8_t *data, const in2_act_q8 *act, int64_t n_groups);

#endif
