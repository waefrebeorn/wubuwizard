/*
 * wubu_foldmath.h -- the folded sin/cos, Silas Lock's algorithm as
 * presented in Kaze Emanuar's "The Folded Polynomial" (N64, 2023):
 * define ONE small polynomial on the first quarter [0, pi/4] and cover
 * the whole circle with the sine-wave symmetries (the quadrant fold).
 * On the N64 the sqrt gave the missing value cheaply; on Zen 4 the
 * odd/even polynomial PAIR is cheaper than the sqrt, so this version
 * computes both sin(r) and cos(r) as tiny polynomials -- branchless,
 * no sqrt, no table, no libm: the "compute" arm of the compute-vs-
 * fetch roofline (the N64 sine-table mis-optimization lesson).
 *
 * HEADER-ONLY (static inline): the fold must be INLINE in the caller
 * so the vectorizer sees the whole thing -- an opaque call cannot be
 * SIMD'd, and libm's loop rides __svml_sinf8. Compiled inline, the
 * fold vectorizes to the same width and beats the sinf/cosf loop.
 *
 * The fold: x = n*pi/2 + r, n = round(x*2/pi), r in [-pi/4, pi/4].
 *   sin(r) = r*P(r^2)  (odd, Taylor to r^9: err < 5e-7)
 *   cos(r) = Q(r^2)    (even, Taylor to r^8: err < 1.2e-6)
 *   q = n mod 4 (branchless bitwise selects):
 *     q0: s=+sin r   c=+cos r      q2: s=-sin r   c=-cos r
 *     q1: s=+cos r   c=-sin r      q3: s=-cos r   c=+sin r
 * Float Cody-Waite reduction (accurate for |x| < ~1e4 -- the RoPE +
 * game ranges; degrades gracefully beyond).
 */
#ifndef WUBU_FOLDMATH_H
#define WUBU_FOLDMATH_H

#include <math.h>

/* the Taylor coefficients for sin on [-pi/4, pi/4] (odd in r) */
#define FMS1  1.0f
#define FMS3 -0.166666666666666666f
#define FMS5  0.008333333333333333f
#define FMS7 -0.000198412698412698f
#define FMS9  0.000002755731922399f

/* the Taylor coefficients for cos (even in r) */
#define FMC2 -0.5f
#define FMC4  0.041666666666666667f
#define FMC6 -0.001388888888888889f
#define FMC8  0.000024801587301587f

/* the Cody-Waite two-part pi/2 (float) */
#define FMPI2_HI 1.57079637050628662109375f
#define FMPI2_LO -4.37113900018624283e-8f
#define FMTWO_OVER_PI 0.6366197723675814f

static inline float wubu_fold_sin_poly(float r)
{
    float r2 = r * r;
    return r * (FMS1 + r2 * (FMS3 + r2 * (FMS5 + r2 * (FMS7 + r2 * FMS9))));
}

static inline float wubu_fold_cos_poly(float r)
{
    float r2 = r * r;
    return 1.0f + r2 * (FMC2 + r2 * (FMC4 + r2 * (FMC6 + r2 * FMC8)));
}

/* both values in one call: one reduction, two tiny polynomials, no
 * sqrt, no branches -- bitwise &/| selects (no short-circuit) so the
 * whole fold vectorizes to the SIMD width. */
static inline void wubu_fold_sincos(float x, float *s, float *c)
{
    float nf = roundf(x * FMTWO_OVER_PI);
    /* r = x - n*pi/2, Cody-Waite */
    float r = fmaf(-nf, FMPI2_HI, x);
    r = fmaf(-nf, FMPI2_LO, r);
    float vs = wubu_fold_sin_poly(r);   /* sin(r), signed */
    float vc = wubu_fold_cos_poly(r);   /* cos(r), even */
    /* the quadrant as a FLOAT in [0,4) -- no int conversions */
    float q = nf - 4.0f * floorf(nf * 0.25f);
    float odd  = ((q >= 1.0f) & (q < 2.0f)) | (q >= 3.0f) ? 1.0f : 0.0f;
    float s_mag = fmaf(odd, vc - vs, vs);                  /* odd ? vc : vs */
    float c_mag = fmaf(odd, vs - vc, vc);                  /* odd ? vs : vc */
    float s_neg = (q >= 2.0f) ? -1.0f : 1.0f;
    float c_neg = (q >= 1.0f) & (q < 3.0f) ? -1.0f : 1.0f;
    *s = s_mag * s_neg;
    *c = c_mag * c_neg;
}

static inline float wubu_fold_sin(float x)
{
    float s, c;
    wubu_fold_sincos(x, &s, &c);
    return s;
}

static inline float wubu_fold_cos(float x)
{
    float s, c;
    wubu_fold_sincos(x, &s, &c);
    return c;
}

/* ---- the AVX2 8-wide batch (ported from wuburvc src/wubu_math.c,
 * the sibling repo's CPU-optimization research, 2026-08-09).
 * Computes 8 sin/cos pairs per iteration with FMA + branchless
 * quadrant blending. Falls back to the scalar fold on non-AVX2. */
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
static inline void wubu_fold_sincos8(const float *x, float *s, float *c, int n)
{
    const __m256 k2opi  = _mm256_set1_ps(FMTWO_OVER_PI);
    const __m256 khalfpi = _mm256_set1_ps(FMPI2_HI);
    const __m256 khalflo = _mm256_set1_ps(FMPI2_LO);
    /* our scalar coefficients (the higher-order 9-term sin / 8-term cos) */
    const __m256 ks9 = _mm256_set1_ps(FMS9);
    const __m256 ks7 = _mm256_set1_ps(FMS7);
    const __m256 ks5 = _mm256_set1_ps(FMS5);
    const __m256 ks3 = _mm256_set1_ps(FMS3);
    const __m256 ks1 = _mm256_set1_ps(FMS1);
    const __m256 kc8 = _mm256_set1_ps(FMC8);
    const __m256 kc6 = _mm256_set1_ps(FMC6);
    const __m256 kc4 = _mm256_set1_ps(FMC4);
    const __m256 kc2 = _mm256_set1_ps(FMC2);
    const __m256 kone = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 q = _mm256_mul_ps(xv, k2opi);
        __m256 nf = _mm256_round_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 r = _mm256_fnmadd_ps(nf, khalfpi, xv);   /* Cody-Waite */
        r = _mm256_fnmadd_ps(nf, khalflo, r);
        /* the quadrant as a float in [0,4) */
        __m256 qd = _mm256_sub_ps(nf, _mm256_mul_ps(_mm256_set1_ps(4.0f),
                              _mm256_floor_ps(_mm256_mul_ps(nf, _mm256_set1_ps(0.25f)))));
        __m256 sgn = _mm256_blendv_ps(_mm256_set1_ps(1.0f), _mm256_set1_ps(-1.0f),
                                      _mm256_cmp_ps(r, _mm256_setzero_ps(), _CMP_LT_OQ));
        __m256 ar = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), r);
        __m256 r2 = _mm256_mul_ps(ar, ar);
        /* sin poly: r * (s1 + r2*(s3 + r2*(s5 + r2*(s7 + r2*s9)))) */
        __m256 ps = _mm256_fmadd_ps(ks9, r2, ks7);
        ps = _mm256_fmadd_ps(ps, r2, ks5);
        ps = _mm256_fmadd_ps(ps, r2, ks3);
        ps = _mm256_fmadd_ps(ps, r2, ks1);
        __m256 sa = _mm256_mul_ps(ar, ps);
        /* cos poly: 1 + r2*(c2 + r2*(c4 + r2*(c6 + r2*c8))) */
        __m256 pc = _mm256_fmadd_ps(kc8, r2, kc6);
        pc = _mm256_fmadd_ps(pc, r2, kc4);
        pc = _mm256_fmadd_ps(pc, r2, kc2);
        __m256 ca = _mm256_fmadd_ps(pc, r2, kone);
        /* odd quadrant ? swap : keep */
        __m256 m1 = _mm256_cmp_ps(qd, _mm256_set1_ps(1.0f), _CMP_LT_OQ);
        __m256 m2 = _mm256_cmp_ps(qd, _mm256_set1_ps(2.0f), _CMP_LT_OQ);
        __m256 m3 = _mm256_cmp_ps(qd, _mm256_set1_ps(3.0f), _CMP_LT_OQ);
        __m256 m4b = _mm256_cmp_ps(qd, _mm256_set1_ps(3.0f), _CMP_GE_OQ);
        __m256 m2b = _mm256_andnot_ps(m1, m2);
        __m256 m3b = _mm256_andnot_ps(m2, m3);
        __m256 sa_sgn = _mm256_mul_ps(sa, sgn);
        __m256 sa_neg = _mm256_xor_ps(sa_sgn, _mm256_set1_ps(-0.0f));
        __m256 ca_neg = _mm256_xor_ps(ca, _mm256_set1_ps(-0.0f));
        /* sin: q0 sgn*sa | q1 ca | q2 -sgn*sa | q3 -ca */
        __m256 sv = _mm256_blendv_ps(_mm256_setzero_ps(), sa_sgn, m1);
        sv = _mm256_blendv_ps(sv, ca, m2b);
        sv = _mm256_blendv_ps(sv, sa_neg, m3b);
        sv = _mm256_blendv_ps(sv, ca_neg, m4b);
        /* cos: q0 ca | q1 -sgn*sa | q2 -ca | q3 sgn*sa */
        __m256 cv = _mm256_blendv_ps(_mm256_setzero_ps(), ca, m1);
        cv = _mm256_blendv_ps(cv, sa_neg, m2b);
        cv = _mm256_blendv_ps(cv, ca_neg, m3b);
        cv = _mm256_blendv_ps(cv, sa_sgn, m4b);
        _mm256_storeu_ps(s + i, sv);
        _mm256_storeu_ps(c + i, cv);
    }
    for (; i < n; i++) {
        float ss, cc;
        wubu_fold_sincos(x[i], &ss, &cc);
        s[i] = ss;
        c[i] = cc;
    }
}
#else
static inline void wubu_fold_sincos8(const float *x, float *s, float *c, int n)
{
    for (int i = 0; i < n; i++) {
        float ss, cc;
        wubu_fold_sincos(x[i], &ss, &cc);
        s[i] = ss;
        c[i] = cc;
    }
}
#endif

#endif
