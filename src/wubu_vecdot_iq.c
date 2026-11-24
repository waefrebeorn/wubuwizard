/* wubu_vecdot_iq.c -- fused IQ2_S / IQ3_S x Q8_K dot products. Strict C11.
 *
 * Reference: ggml_vec_dot_iq2_s_q8_K_generic / ggml_vec_dot_iq3_s_q8_K_generic
 * in llama/ggml-cpu/quants.c (MIT), with the block layouts from
 * llama/ggml-common.h and the grid tables already used by src/gguf_reader.c.
 *
 * Grid semantics (must match gguf_reader.c dequantisers, which are the
 * independent reference the gate compares against):
 *   iq2s_grid[1024] : uint64, 8 int8 magnitudes per entry
 *   iq3s_grid[512]  : uint32, 4 int8 magnitudes per entry
 */
#include "wubu_vecdot_iq.h"
#include "gguf_reader.h"

#include <stdint.h>
#include <string.h>

#define QK_K 256

typedef struct {
    uint16_t d;
    uint8_t  qs[QK_K / 4];      /* 64 */
    uint8_t  qh[QK_K / 32];     /* 8  */
    uint8_t  scales[QK_K / 32]; /* 8  */
} wubu_block_iq2_s;

typedef struct {
    uint16_t d;
    uint8_t  qs[QK_K / 4];      /* 64 */
    uint8_t  qh[QK_K / 32];     /* 8  */
    uint8_t  signs[QK_K / 8];   /* 32 */
    uint8_t  scales[QK_K / 64]; /* 4  */
} wubu_block_iq3_s;

static const uint8_t ksign_mask[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static const uint64_t iq2s_grid[1024] = {
#include "iq2s_grid.inc"
};

static const uint32_t iq3s_grid[512] = {
#include "iq3s_grid.inc"
};

int wubu_vec_dot_iq2_s_q8_k(int n, float *s, const void *w, const void *q8)
{
    if (!s || !w || !q8 || n <= 0 || (n % QK_K) != 0) return -1;

    const wubu_block_iq2_s *x = (const wubu_block_iq2_s *)w;
    const block_q8_K       *y = (const block_q8_K *)q8;
    const int nb = n / QK_K;

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d = gguf_f16_to_f32(x[i].d) * y[i].d;
        const int8_t  *q8v = y[i].qs;
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = qs + QK_K / 8;

        int bsum = 0;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const int ls1 = 1 + 2 * (x[i].scales[ib32] & 0xf);
            const int ls2 = 1 + 2 * (x[i].scales[ib32] >> 4);
            int sumi1 = 0, sumi2 = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t *grid = (const uint8_t *)(iq2s_grid +
                    (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
                for (int j = 0; j < 8; ++j)
                    sumi1 += q8v[j] * grid[j] * ((signs[l] & ksign_mask[j]) ? -1 : 1);
                q8v += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const uint8_t *grid = (const uint8_t *)(iq2s_grid +
                    (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
                for (int j = 0; j < 8; ++j)
                    sumi2 += q8v[j] * grid[j] * ((signs[l] & ksign_mask[j]) ? -1 : 1);
                q8v += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs += 4;
            signs += 4;
        }
        sumf += d * (float)bsum;
    }
    /* IQ2_S dequant carries a 0.25 factor on (0.5 + scale), i.e. 0.125 * (1 + 2*scale). */
    *s = 0.125f * sumf;
    return 0;
}

int wubu_vec_dot_iq3_s_q8_k(int n, float *s, const void *w, const void *q8)
{
    if (!s || !w || !q8 || n <= 0 || (n % QK_K) != 0) return -1;

    const wubu_block_iq3_s *x = (const wubu_block_iq3_s *)w;
    const block_q8_K       *y = (const block_q8_K *)q8;
    const int nb = n / QK_K;

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d = gguf_f16_to_f32(x[i].d) * y[i].d;
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = x[i].signs;
        const int8_t  *q8v = y[i].qs;

        int bsum = 0;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const int ls1 = 2 * (x[i].scales[ib32 / 2] & 0xf) + 1;
            const int ls2 = 2 * (x[i].scales[ib32 / 2] >> 4) + 1;

            int sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = (const uint8_t *)(iq3s_grid +
                    (qs[2 * l + 0] | ((qh[ib32 + 0] << (8 - 2 * l)) & 256)));
                const uint8_t *grid2 = (const uint8_t *)(iq3s_grid +
                    (qs[2 * l + 1] | ((qh[ib32 + 0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8v[j + 0] * ((signs[l] & ksign_mask[j + 0]) ? -1 : 1);
                    sumi += grid2[j] * q8v[j + 4] * ((signs[l] & ksign_mask[j + 4]) ? -1 : 1);
                }
                q8v += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls1;

            sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = (const uint8_t *)(iq3s_grid +
                    (qs[2 * l + 0] | ((qh[ib32 + 1] << (8 - 2 * l)) & 256)));
                const uint8_t *grid2 = (const uint8_t *)(iq3s_grid +
                    (qs[2 * l + 1] | ((qh[ib32 + 1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8v[j + 0] * ((signs[l] & ksign_mask[j + 0]) ? -1 : 1);
                    sumi += grid2[j] * q8v[j + 4] * ((signs[l] & ksign_mask[j + 4]) ? -1 : 1);
                }
                q8v += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls2;
        }
        sumf += d * (float)bsum;
    }
    *s = sumf;
    return 0;
}

void wubu_vec_dot_iq2_s_q8_k_vd(int n, float *s, size_t bs, const void *vx,
                                size_t bx, const void *vy, size_t by, int nrc)
{
    (void)bs; (void)bx; (void)by;
    if (nrc != 1) { if (s) *s = 0.0f; return; }
    if (wubu_vec_dot_iq2_s_q8_k(n, s, vx, vy) != 0 && s) *s = 0.0f;
}

void wubu_vec_dot_iq3_s_q8_k_vd(int n, float *s, size_t bs, const void *vx,
                                size_t bx, const void *vy, size_t by, int nrc)
{
    (void)bs; (void)bx; (void)by;
    if (nrc != 1) { if (s) *s = 0.0f; return; }
    if (wubu_vec_dot_iq3_s_q8_k(n, s, vx, vy) != 0 && s) *s = 0.0f;
}
