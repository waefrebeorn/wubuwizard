/* test_eg30_fp8attn.c -- EG-30 gate: FP8 Q/K/V, F32 accumulate attention. */
#include <stdio.h>
#include <math.h>

#include "wubu_fp8_attn.h"
#include "wubu_fp8.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define NQ 2
#define NK 3
#define D 4

static float ref_q[NQ*D] = { 1,0,1,0,  0,1,0,1 };
static float ref_k[NK*D] = { 1,0,0,0,  0,1,0,0,  1,1,0,0 };
static float ref_v[NK*D] = { 1,0,0,0,  0,1,0,0,  0,0,1,0 };

int main(void) {
    uint8_t q[NQ*D], k[NK*D], v[NK*D];
    float out[NQ*D], ref[NQ*D], s[NK], p[NK], maxerr = 0.0f;
    const float qs = 0.5f, ks = 1.0f, vs = 2.0f;
    size_t fp8_bytes, f32_bytes;
    int i, t, j;

    for (i = 0; i < NQ*D; i++) q[i] = wubu_fp8_e4m3_from_f32(ref_q[i]);
    for (i = 0; i < NK*D; i++) {
        k[i] = wubu_fp8_e4m3_from_f32(ref_k[i]);
        v[i] = wubu_fp8_e4m3_from_f32(ref_v[i]);
    }

    CHECK(wubu_fp8_attn(q, NQ, D, qs, k, NK, ks, v, vs, out) == 0);

    /* F32 reference with the same scales. */
    for (i = 0; i < NQ; i++) {
        float mx = -INFINITY, sum = 0.0f;
        for (t = 0; t < NK; t++) {
            float dot = 0.0f;
            for (j = 0; j < D; j++) dot += ref_q[i*D+j]*qs * ref_k[t*D+j]*ks;
            s[t] = dot / sqrtf((float)D);
            if (s[t] > mx) mx = s[t];
        }
        for (t = 0; t < NK; t++) { p[t] = expf(s[t]-mx); sum += p[t]; }
        for (t = 0; t < NK; t++) p[t] /= sum;
        for (j = 0; j < D; j++) {
            float a = 0.0f;
            for (t = 0; t < NK; t++) a += p[t] * ref_v[t*D+j] * vs;
            ref[i*D+j] = a;
        }
    }
    for (i = 0; i < NQ*D; i++) {
        float e = fabsf(out[i] - ref[i]);
        if (e > maxerr) maxerr = e;
    }
    printf("EG-30 fp8_attn out=[%.3f %.3f %.3f %.3f] maxerr=%.6f\n",
           out[0], out[1], out[2], out[3], maxerr);
    CHECK(maxerr < 0.01f);

    fp8_bytes = sizeof(q) + sizeof(k) + sizeof(v);
    f32_bytes = sizeof(ref_q) + sizeof(ref_k) + sizeof(ref_v);
    printf("EG-30 operand bytes fp8=%u f32=%u\n",
           (unsigned)fp8_bytes, (unsigned)f32_bytes);
    CHECK(fp8_bytes * 4 == f32_bytes);

    CHECK(wubu_fp8_attn(NULL, NQ, D, qs, k, NK, ks, v, vs, out) == -1);
    CHECK(wubu_fp8_attn(q, 0, D, qs, k, NK, ks, v, vs, out) == -1);
    CHECK(wubu_fp8_attn(q, NQ, D, 0.0f, k, NK, ks, v, vs, out) == -1);
    CHECK(wubu_fp8_attn(q, NQ, D, qs, k, NK, ks, v, vs, NULL) == -1);

    if (failures) {
        fprintf(stderr, "test_eg30_fp8attn: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg30_fp8attn: PASS\n");
    return 0;
}
