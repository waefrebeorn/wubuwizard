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

void  in2_act_quantise(const float *x, int64_t n_groups, in2_act_q8 *out);
float dot_row_iq2_xs_q8(const uint8_t *data, const in2_act_q8 *act, int64_t n_groups);

#endif
