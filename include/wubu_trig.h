/* wubu_trig.h -- one deterministic sinf/cosf, shared by the CPU and CUDA attention paths.
 *
 * Why: EG-123 reports the device RoPE differing from the CPU one by 2.384e-06, and that is
 * the whole residual in the fused attention path (q 1.4e-06, attention 3.0e-07). The causes
 * are powf(theta_scale, j) and cosf/sinf, each a different implementation per toolchain.
 * Replacing them with one implementation built from fmaf and integer work makes the rotation
 * bit-identical on both sides.
 */
#ifndef WUBU_TRIG_H
#define WUBU_TRIG_H

#include <math.h>
#include <string.h>
#include <stdint.h>

#ifdef __CUDACC__
#define WUBU_TRIG_HD __host__ __device__ static inline
#else
#define WUBU_TRIG_HD static inline
#endif

/* Cody-Waite reduction: n = round(x * 2/pi), r = x - n*(pi/2) in three parts. */
WUBU_TRIG_HD void wubu_sincosf(float x, float *s, float *c)
{
    /* Reduction in double: theta reaches ~1024, and single-precision reduction loses the
     * fractional part long before that (measured: 4e-06 against libc with a float
     * reduction). Double is IEEE-defined identically on both sides, so this stays
     * deterministic; rope is small enough that the double rate does not matter. */
    const double two_over_pi = 6.36619772367581382433e-01;
    const double pio2_1      = 1.57079632673412561417e+00;
    const double pio2_2      = 6.07710050650619224932e-11;
    const double pio2_3      = 2.02226624879595063154e-21;

    const double xd = (double)x;
    const double fn = floor(xd * two_over_pi + 0.5);
    double rd = xd - fn * pio2_1;
    rd = rd - fn * pio2_2;
    rd = rd - fn * pio2_3;
    const float r = (float)rd;

    const float r2 = r * r;

    /* sin(r) = r*(1 + r2*(-1/6 + r2*(1/120 + r2*(-1/5040 + r2/362880)))) */
    float sp = 1.0f / 362880.0f;
    sp = fmaf(sp, r2, -1.0f / 5040.0f);
    sp = fmaf(sp, r2, 1.0f / 120.0f);
    sp = fmaf(sp, r2, -1.0f / 6.0f);
    sp = fmaf(sp, r2, 1.0f);
    const float sr = fmaf(sp, r, 0.0f);

    /* cos(r) = 1 + r2*(-1/2 + r2*(1/24 + r2*(-1/720 + r2/40320))) */
    float cp = 1.0f / 40320.0f;
    cp = fmaf(cp, r2, -1.0f / 720.0f);
    cp = fmaf(cp, r2, 1.0f / 24.0f);
    cp = fmaf(cp, r2, -0.5f);
    cp = fmaf(cp, r2, 1.0f);

    const int32_t n = (int32_t)fn;
    const unsigned q = (unsigned)(n & 3);
    float sv, cv;
    if (q == 0)      { sv = sr;  cv = cp;  }
    else if (q == 1) { sv = cp;  cv = -sr; }
    else if (q == 2) { sv = -sr; cv = -cp; }
    else             { sv = -cp; cv = sr;  }
    if (s) *s = sv;
    if (c) *c = cv;
}

WUBU_TRIG_HD float wubu_sinf(float x) { float s, c; wubu_sincosf(x, &s, &c); return s; }
WUBU_TRIG_HD float wubu_cosf(float x) { float s, c; wubu_sincosf(x, NULL, &c); return c; }

#endif
