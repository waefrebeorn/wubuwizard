/* wubu_vecdot_iq3_xxs.c -- fused IQ3_XXS x Q8_K dot product (EG-120).
 *
 * IQ3_XXS is 6.7%% of the Gemma-4-12B file (282 MB across 20 tensors) and had no fused
 * dot, so every decode step dequantised it to F32 (1.1 GB of conversion) before the
 * multiply. The kernel below is ggml's scalar reference (MIT) ported to strict C11:
 * block = fp16 d + 64 bytes of 3-bit grid indices + 32 bytes of signs/scales, and the
 * final result carries ggml's 0.25 scale factor.
 */
#include "wubu_vecdot_iq3_xxs.h"
#include "gguf_reader.h"

#include <stdint.h>
#include <string.h>

static const uint8_t kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128,
};

static const uint8_t ksigns_iq2xs[128] = {
    0, 129, 130, 3, 132, 5, 6, 135,
    136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23,
    24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39,
    40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183,
    184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71,
    72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215,
    216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231,
    232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119,
    120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};


const void *wubu_vecdot_iq3_xxs_grid(void) { return (const void *)iq3xxs_grid; }
const void *wubu_vecdot_iq3_xxs_signs(void) { return (const void *)ksigns_iq2xs; }

/* x: IQ3_XXS blocks (98 bytes per 256 elements), y: Q8_K blocks (292 bytes per 256). */
float wubu_vec_dot_iq3_xxs_q8_K(const void *vx, const void *vy, int n)
{
    const uint8_t *x = (const uint8_t *)vx;
    const uint8_t *y = (const uint8_t *)vy;
    const int nb = n / 256;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const uint8_t *xb = x + (size_t)i * 98;
        const uint8_t *yb = y + (size_t)i * 292;

        uint16_t d_bits;
        memcpy(&d_bits, xb, 2);
        const float dx = gguf_f16_to_f32(d_bits);
        float dy;
        memcpy(&dy, yb, 4);                       /* Q8_K stores d as f32 */

        const uint8_t *q3 = xb + 2;
        const uint8_t *gas = xb + 2 + 64;
        const int8_t *q8 = (const int8_t *)(yb + 4);
        int32_t bsum = 0;

        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32;
            memcpy(&aux32, gas, 4);
            gas += 4;
            const uint32_t ls = 2u * (aux32 >> 28) + 1u;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = (const uint8_t *)&iq3xxs_grid[q3[2 * l + 0]];
                const uint8_t *grid2 = (const uint8_t *)&iq3xxs_grid[q3[2 * l + 1]];
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j + 0] * ((signs & kmask_iq2xs[j + 0]) ? -1 : 1);
                    sumi += grid2[j] * q8[j + 4] * ((signs & kmask_iq2xs[j + 4]) ? -1 : 1);
                }
                q8 += 8;
            }
            q3 += 8;
            bsum += sumi * (int32_t)ls;
        }
        sumf += dx * dy * (float)bsum;
    }
    return 0.25f * sumf;
}
