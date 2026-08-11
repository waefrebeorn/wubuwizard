/* fexpa_probe.c — measure vexpq_f32 (FEXPA) vs exp2f on silu's input range */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <arm_neon.h>

static inline float32x4_t fexpaq(float32x4_t a) {
    float32x4_t r;
    __asm__("fexpa %0.4s, %1.4s" : "=w"(r) : "w"(a));
    return r;
}

int main(void) {
    const float log2e = 1.4426950408889634f;
    double max_rel = 0, max_abs = 0;
    double sum = 0; int n = 0, big = 0;
    /* silu inputs across UNET/VAE: values in ~[-10, 10] */
    for (float s = -10.0f; s <= 10.0f; s += 0.0001f) {
        float x = -s * log2e;
        float ref = exp2f(x);
        float fast = vgetq_lane_f32(fexpaq(vdupq_n_f32(x)), 0);
        float rel = fabsf(fast - ref) / (fabsf(ref) + 1e-30f);
        float absd = fabsf(fast - ref);
        if (rel > max_rel) max_rel = rel;
        if (absd > max_abs) max_abs = absd;
        if (absd > 1e-6f) big++;
        sum += rel; n++;
    }
    printf("FEXPA vs exp2f over [-10,10] step 1e-4:\n");
    printf("  max relative err = %.3e\n  max abs err      = %.3e\n", max_rel, max_abs);
    printf("  mean rel err     = %.3e\n  samples>1e-6 abs = %d / %d\n", sum / n, big, n);
    /* also check the silu output diff */
    double silu_max = 0; int silu_over_1ulp = 0; n = 0;
    for (float s = -10.0f; s <= 10.0f; s += 0.0001f) {
        float r1 = s / (1.0f + exp2f(-s * log2e));
        float f = vgetq_lane_f32(fexpaq(vdupq_n_f32(-s * log2e)), 0);
        float r2 = s / (1.0f + f);
        float d = fabsf(r1 - r2);
        if (d > silu_max) silu_max = d;
        if (d > 1e-7f) silu_over_1ulp++;
        n++;
    }
    printf("  silu out max abs diff = %.3e (over 1e-7: %d)\n", silu_max, silu_over_1ulp);
    return 0;
}
