/* wubu_fp8_shift.c -- EG-109 arithmetic E4M3 decode. Strict C11. */
#include "wubu_fp8_shift.h"

#include <string.h>

float wubu_fp8_shift_one(uint8_t code)
{
    uint32_t mag = (uint32_t)code & 0x7Fu;
    uint32_t sign = ((uint32_t)code & 0x80u) << 24;
    uint32_t bits;
    float value;
    if (mag < 8u) {
        value = (float)mag / 512.0f;
        return (code & 0x80u) ? -value : value;
    }
    bits = (mag << 20) + 0x3c000000u;
    if (mag == 127u) bits = 0;
    bits |= sign;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

int wubu_fp8_shift_decode(const uint8_t *codes, float *out, int n)
{
    if (!codes || !out || n < 1 || n > WUBU_FP8_SHIFT_MAX) return -1;
    for (int i = 0; i < n; ++i) out[i] = wubu_fp8_shift_one(codes[i]);
    return 0;
}
