/* test_eg109_fp8_shift.c -- EG-109 gate: table-free E4M3 arithmetic decode. */
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "wubu_fp8_shift.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

static float reference(uint8_t b)
{
    uint32_t mag = b & 0x7Fu;
    uint32_t sign = ((uint32_t)b & 0x80u) << 24;
    uint32_t bits;
    if (mag < 8u) {
        float v = (float)mag / 512.0f;
        uint32_t sub;
        memcpy(&sub, &v, sizeof(sub));
        return (b & 0x80u) ? -v : v;
    }
    bits = (mag << 20) + 0x3c000000u;
    if (mag == 127u) bits = 0;
    bits |= sign;
    {
        float v;
        memcpy(&v, &bits, sizeof(v));
        return v;
    }
}

int main(void) {
    uint8_t codes[256];
    float got[256];
    int mismatch = 0;
    for (int i = 0; i < 256; ++i) codes[i] = (uint8_t)i;
    CHECK(wubu_fp8_shift_decode(codes, got, 256) == 0);
    for (int i = 0; i < 256; ++i) {
        float want = reference(codes[i]);
        if (isnan(got[i]) || isnan(want) || fabs(got[i] - want) > 0.0f) ++mismatch;
    }
    CHECK(mismatch == 0);
    CHECK(fabsf(wubu_fp8_shift_one(0x38) - 1.0f) < 1e-9);
    CHECK(fabsf(wubu_fp8_shift_one(0x7E) - 448.0f) < 1e-6);
    CHECK(fabsf(wubu_fp8_shift_one(0x01) - 1.0f / 512.0f) < 1e-12);
    CHECK(wubu_fp8_shift_one(0x7F) == 0.0f);       /* NaN-to-zero contract */
    CHECK(wubu_fp8_shift_one(0xFF) == 0.0f);       /* sign preserved on zero */
    printf("EG-109 exhaustive 256-code mismatch=%d 0x38=%.1f 0x7E=%.1f 0x01=%.6f\n",
           mismatch, wubu_fp8_shift_one(0x38), wubu_fp8_shift_one(0x7E),
           wubu_fp8_shift_one(0x01));
    CHECK(wubu_fp8_shift_decode(NULL, got, 4) == -1);
    CHECK(wubu_fp8_shift_decode(codes, NULL, 4) == -1);
    CHECK(wubu_fp8_shift_decode(codes, got, 0) == -1);
    if (failures) { fprintf(stderr, "test_eg109_fp8_shift: FAIL (%d)\n", failures); return 1; }
    printf("test_eg109_fp8_shift: PASS\n");
    return 0;
}
