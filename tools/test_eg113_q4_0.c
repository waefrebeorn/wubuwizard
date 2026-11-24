/* test_eg113_q4_0.c -- EG-113 gate: fused Q4_0 x Q8_K dot.
 *
 * Reference is the repo's independent dequantiser: gguf_dequantize() to F32,
 * dotted against the Q8_K activations (d * qs). The AVX2 and scalar ports must
 * both reproduce it; the AVX2 port must also agree with the scalar port.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gguf_reader.h"
#include "wubu_vecdot_q4_0.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define QK_K 256
#define NB_32 (QK_K / 32)

static uint64_t rng_state = 0xC0FFEE123456789ULL;
static uint32_t rng_u32(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

static int close_enough(float a, float b)
{
    const float diff = fabsf(a - b);
    const float scale = fabsf(a) > fabsf(b) ? fabsf(a) : fabsf(b);
    return diff <= 1e-3f + 1e-4f * scale;
}

int main(void)
{
    enum { NBLK = 64 };
    static uint8_t block[NBLK][NB_32 * WUBU_Q4_0_BLOCK_SIZE];
    static float w_f32[NBLK][QK_K];
    static float x[QK_K];
    static block_q8_K q8[2];

    for (int b = 0; b < NBLK; ++b) {
        for (int s = 0; s < NB_32; ++s) {
            uint8_t *blk = block[b] + s * WUBU_Q4_0_BLOCK_SIZE;
            const uint16_t d16 = (uint16_t)(0x2C00u + (rng_u32() & 0x3FFu)); /* fp16 ~0.01..0.05 */
            memcpy(&blk[0], &d16, 2);
            for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t)rng_u32();
        }
    }

    float worst_avx = 0.0f;
    for (int b = 0; b < NBLK; ++b) {
        for (int j = 0; j < QK_K; ++j)
            x[j] = ((float)(int32_t)(rng_u32() & 0xFFFF) - 32768.0f) / 4096.0f;
        quantize_row_q8_K(x, q8, QK_K);

        float got = 0.0f;
        int rc = wubu_vec_dot_q4_0_q8_k(QK_K, &got, block[b], q8);
        CHECK(rc == 0);

        float sc = 0.0f;
        rc = wubu_vec_dot_q4_0_q8_k_scalar(QK_K, &sc, block[b], q8);
        CHECK(rc == 0);
        CHECK(close_enough(got, sc));

        gguf_dequantize(block[b], GGML_TYPE_Q4_0, QK_K, w_f32[b]);
        double acc = 0.0;
        for (int j = 0; j < QK_K; ++j) acc += (double)w_f32[b][j] * (double)q8[0].d * (double)q8[0].qs[j];
        const float want = (float)acc;
        const float rel = fabsf(got - want) / (1.0f + fabsf(want));
        if (rel > worst_avx) worst_avx = rel;
        CHECK(close_enough(got, want));
    }

    /* Fail-closed contract. */
    {
        float s = 7.0f;
        CHECK(wubu_vec_dot_q4_0_q8_k(QK_K, NULL, block[0], q8) == -1);
        CHECK(wubu_vec_dot_q4_0_q8_k(QK_K, &s, NULL, q8) == -1);
        CHECK(wubu_vec_dot_q4_0_q8_k(QK_K, &s, block[0], NULL) == -1);
        CHECK(wubu_vec_dot_q4_0_q8_k(128, &s, block[0], q8) == -1);
        CHECK(s == 7.0f);
    }

    printf("EG-113 q4_0_fused blocks=%d worst_rel=%.3g\n", NBLK, worst_avx);
    if (failures) { fprintf(stderr, "test_eg113_q4_0: FAIL (%d)\n", failures); return 1; }
    printf("test_eg113_q4_0: PASS\n");
    return 0;
}
