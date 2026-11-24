/* wubu_vecdot_iq2xs.c -- fused IQ2_XS x Q8_K dot product. Strict C11.
 *
 * The 5 IQ2_XS tensors in the 12B Gemma-4 IQ2 GGUF cost 7.2 ms per 3840x3840
 * matvec (WUBU_GPU_STATS) because the type had no vec_dot at all: the engine
 * dequantised the whole tensor per call. This kernel reads the 2.3125 bpw blocks
 * directly against Q8_K activations.
 *
 * Layout: block_iq2_xs = fp16 d(2) + uint16 qs[32] + uint8 scales[8]
 *         = 74 bytes / 256 elements.
 *   qs[g] : low 9 bits = iq2xs_grid index, top 7 bits = sign pattern for the
 *           8 values of that 32-element group (expanded through ksigns_iq2xs)
 *   scales[g/4]: two 4-bit scales, low nibble for group g, high for g+1
 *
 * Reference: ggml_vec_dot_iq2_xs_q8_K(_generic).
 */
#include "wubu_vecdot_iq2xs.h"
#include "gguf_reader.h"

#include <stdint.h>
#include <string.h>

#define QK_K 256

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];      /* 32 */
    uint8_t  scales[QK_K / 32]; /* 8  */
} wubu_block_iq2_xs;

static const uint64_t iq2xs_grid[512] = {
#include "iq2xs_grid.inc"
};

static const uint8_t wubu_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

int wubu_vec_dot_iq2_xs_q8_k_scalar(int n, float *s, const void *w, const void *q8v)
{
    const wubu_block_iq2_xs *x = (const wubu_block_iq2_xs *)w;
    const block_q8_K        *y = (const block_q8_K *)q8v;
    const int nb = n / QK_K;

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d = gguf_f16_to_f32(x[i].d) * y[i].d;
        const int8_t *q8 = y[i].qs;
        /* Per 32-element group there are FOUR grid entries (8 values each); the
         * low nibble of scales[g] covers entries 0-1, the high nibble 2-3, and
         * the dequant carries the IQ2 family's 0.125 * (1 + 2*scale) factor. */
        int bsum = 0;
        for (int g = 0; g < QK_K / 32; ++g) {
            const int ls_lo = 1 + 2 * (x[i].scales[g] & 0x0f);
            const int ls_hi = 1 + 2 * (x[i].scales[g] >> 4);
            for (int l = 0; l < 4; ++l) {
                const uint16_t q = x[i].qs[4 * g + l];
                const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (q & 511));
                const uint8_t signs = wubu_ksigns_iq2xs[(q >> 9) & 127];
                int sumi = 0;
                for (int j = 0; j < 8; ++j)
                    sumi += q8[g * 32 + l * 8 + j] * grid[j] * ((signs >> j) & 1 ? -1 : 1);
                bsum += sumi * ((l < 2) ? ls_lo : ls_hi);
            }
        }
        sumf += 0.125f * d * (float)bsum;
    }
    *s = sumf;
    return 0;
}

int wubu_vec_dot_iq2_xs_q8_k(int n, float *s, const void *w, const void *q8)
{
    if (!s || !w || !q8 || n <= 0 || (n % QK_K) != 0) return -1;
    return wubu_vec_dot_iq2_xs_q8_k_scalar(n, s, w, q8);
}

void wubu_vec_dot_iq2_xs_q8_k_vd(int n, float *s, size_t bs, const void *vx,
                                 size_t bx, const void *vy, size_t by, int nrc)
{
    (void)bs; (void)bx; (void)by;
    if (nrc != 1) { if (s) *s = 0.0f; return; }
    if (wubu_vec_dot_iq2_xs_q8_k(n, s, vx, vy) != 0 && s) *s = 0.0f;
}
