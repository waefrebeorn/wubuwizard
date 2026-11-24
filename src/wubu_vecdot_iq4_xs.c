/*
 * wubu_vecdot_iq4_xs.c -- fused IQ4_XS x Q8_K dot product (EG-121).
 *
 * IQ4_XS has no AVX2 kernel in this engine: iq4_xs_vec_dot falls through to ggml's
 * scalar generic path, so its 125 MB (4 tensors in the 12B Gemma file) cost ~25 ms
 * per decode step on the CPU. This is the same kernel restructured so the compiler
 * can vectorise it, plus the table accessor the CUDA bridge needs.
 *
 * Block (136 bytes per 256 elements, ggml block_iq4_xs):
 *   fp16 d | uint16 scales_h | uint8 scales_l[4] | uint8 qs[128]
 * Per 32 elements: 16 bytes of qs hold 32 4-bit codes (low nibble -> element j,
 * high nibble -> element j+16) and a signed 6-bit scale (ls - 32) taken from
 * scales_l[ib/2] interleaved with two bits of scales_h.
 *
 * Ported from ggml's ggml_vec_dot_iq4_xs_q8_K_generic (MIT).
 */
#include "wubu_vecdot_iq4_xs.h"
#include "gguf_reader.h"

#include <stdint.h>
#include <string.h>

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

const void *wubu_vecdot_iq4_xs_table(void) { return (const void *)kvalues_iq4nl; }

float wubu_vec_dot_iq4_xs_q8_K(const void *vx, const void *vy, int n)
{
    const uint8_t *x = (const uint8_t *)vx;
    const uint8_t *y = (const uint8_t *)vy;
    const int nb = n / 256;
    float sumf = 0.0f;

    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t *xb = x + (size_t)ibl * 136;
        const uint8_t *yb = y + (size_t)ibl * 292;

        uint16_t d_bits;
        memcpy(&d_bits, xb, 2);
        const float d4d8 = gguf_f16_to_f32(d_bits) * *(const float *)yb;

        uint16_t h;
        memcpy(&h, xb + 2, 2);
        const uint8_t *scales_l = xb + 4;
        const uint8_t *qs = xb + 8;
        const int8_t *q8 = (const int8_t *)(yb + 4);

        for (int ib = 0; ib < 8; ib += 2) {
            const uint8_t ls1 = (uint8_t)((scales_l[ib / 2] & 0x0f) | ((h << 4) & 0x30));
            const uint8_t ls2 = (uint8_t)((scales_l[ib / 2] >> 4) | ((h << 2) & 0x30));
            h = (uint16_t)(h >> 4);
            const float d1 = d4d8 * (float)((int)ls1 - 32);
            const float d2 = d4d8 * (float)((int)ls2 - 32);
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j + 0] * kvalues_iq4nl[qs[j] & 0x0f];
                sumi2 += q8[j + 16] * kvalues_iq4nl[qs[j] >> 4];
            }
            sumf += d1 * (float)(sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j + 0] * kvalues_iq4nl[qs[j] & 0x0f];
                sumi2 += q8[j + 16] * kvalues_iq4nl[qs[j] >> 4];
            }
            sumf += d2 * (float)(sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    return sumf;
}
