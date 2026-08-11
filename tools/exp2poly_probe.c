/* exp2poly_probe.c — measure NEON polynomial exp2 vs libm exp2f */
#include <stdio.h>
#include <math.h>
#include <arm_neon.h>

static inline float32x4_t exp2q_f32(float32x4_t x) {
    float32x4_t i = vrndnq_f32(x);
    float32x4_t f = vsubq_f32(x, i);
    int32x4_t ie = vcvtq_s32_f32(i);
    ie = vshlq_n_s32(vaddq_s32(ie, vdupq_n_s32(127)), 23);
    float32x4_t pow2i = vreinterpretq_f32_s32(ie);
    float32x4_t u = vmulq_f32(f, vdupq_n_f32(0.6931471805599453f));
    float32x4_t p = vdupq_n_f32(2.4801587301587302e-05f);
    p = vfmaq_f32(vdupq_n_f32(1.984126984126984e-04f), p, u);
    p = vfmaq_f32(vdupq_n_f32(1.3888888888888888e-03f), p, u);
    p = vfmaq_f32(vdupq_n_f32(8.3333333333333332e-03f), p, u);
    p = vfmaq_f32(vdupq_n_f32(4.1666666666666666e-02f), p, u);
    p = vfmaq_f32(vdupq_n_f32(1.6666666666666666e-01f), p, u);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, u);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, u);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, u);
    return vmulq_f32(pow2i, p);
}

int main(void) {
    double max_rel = 0; double sum = 0; int n = 0, big = 0;
    for (float x = -30.0f; x <= 30.0f; x += 0.0005f) {
        float ref = exp2f(x);
        float fast = vgetq_lane_f32(exp2q_f32(vdupq_n_f32(x)), 0);
        float rel = fabsf(fast - ref) / (fabsf(ref) + 1e-30f);
        float absd = fabsf(fast - ref);
        if (rel > max_rel) max_rel = rel;
        if (absd > 1e-6f) big++;
        sum += rel; n++;
    }
    printf("poly-exp2 vs libm exp2f over [-30,30]:\n");
    printf("  max relative err = %.3e\n", max_rel);
    printf("  mean rel err     = %.3e\n  samples>1e-6 abs = %d / %d\n", sum/n, big, n);
    /* silu output diff */
    double silu_max = 0; int over = 0; n = 0;
    const float log2e = 1.4426950408889634f;
    for (float s = -10.0f; s <= 10.0f; s += 0.0005f) {
        float r1 = s / (1.0f + exp2f(-s * log2e));
        float f = vgetq_lane_f32(exp2q_f32(vdupq_n_f32(-s * log2e)), 0);
        float r2 = s / (1.0f + f);
        float d = fabsf(r1 - r2);
        if (d > silu_max) silu_max = d;
        if (d > 1e-6f) over++;
        n++;
    }
    printf("  silu out max abs diff = %.3e (over 1e-6: %d)\n", silu_max, over);
    return 0;
}
