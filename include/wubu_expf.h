/* wubu_expf.h -- one deterministic expf, shared by the CPU attention and the CUDA kernel.
 *
 * Why: EG-128 measured that CUDA's expf and glibc's disagree on ~30% of the values the
 * softmax feeds them, by 1-2 ulp. That difference is what kept the fused attention row at
 * 3.5e-07 relative after every reduction order had been matched, and it cannot be closed by
 * ordering work because the operation itself is implemented differently per toolchain.
 *
 * Both sides call this function. Every step is either an IEEE-defined operation (fmaf,
 * floorf) or integer work, so the two produce the same bits.
 */
#ifndef WUBU_EXPF_H
#define WUBU_EXPF_H

#include <math.h>
#include <string.h>
#include <stdint.h>

#ifdef __CUDACC__
#define WUBU_EXPF_HD __host__ __device__ static inline
#else
#define WUBU_EXPF_HD static inline
#endif

WUBU_EXPF_HD float wubu_expf(float x)
{
    /* exp(x) = 2^k * exp(r), r in [-ln2/2, ln2/2]; Horner polynomial in fmaf so both
     * toolchains perform the same sequence of single-rounded operations. */
    if (x > 88.72283935546875f) x = 88.72283935546875f;
    if (x < -87.33654785156250f) return 0.0f;

    const float log2e  = 1.4426950408889634f;
    const float ln2_hi = 0.693145751953125f;
    const float ln2_lo = 1.428606765330187045e-06f;

    const float kf = floorf(fmaf(x, log2e, 0.5f));
    float r = fmaf(-kf, ln2_hi, x);
    r = fmaf(-kf, ln2_lo, r);

    float p = 1.0f / 720.0f;
    p = fmaf(p, r, 1.0f / 120.0f);
    p = fmaf(p, r, 1.0f / 24.0f);
    p = fmaf(p, r, 1.0f / 6.0f);
    p = fmaf(p, r, 0.5f);
    p = fmaf(p, r, 1.0f);
    p = fmaf(p, r, 1.0f);

    const int32_t k = (int32_t)kf;
    const uint32_t bits = (uint32_t)(k + 127) << 23;
    float sc;
    memcpy(&sc, &bits, sizeof(sc));
    return p * sc;
}

#endif
