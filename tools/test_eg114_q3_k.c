/* test_eg114_q3_k.c -- EG-114 gate: fused Q3_K x Q8_K dot.
 *
 * Reference: gguf_dequantize() to F32 (the repo's independent Q3_K dequantiser),
 * dotted against the dequantised Q8_K row. AVX2 and scalar ports must both match
 * it, and each other.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gguf_reader.h"
#include "wubu_vecdot_q3_k.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define QK_K 256

static uint64_t rng_state = 0x5EEDBEEF1234ULL;
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
    return diff <= 2e-3f + 2e-4f * scale;
}

int main(void)
{
    enum { NBLK = 64 };
    static uint8_t blk[NBLK][WUBU_Q3_K_BLOCK_SIZE];
    static float w_f32[NBLK][QK_K];
    static float x[QK_K];
    static block_q8_K q8[2];

    for (int b = 0; b < NBLK; ++b) {
        const uint16_t d16 = (uint16_t)(0x2C00u + (rng_u32() & 0x3FFu));
        memcpy(&blk[b][0], &d16, 2);
        for (int i = 2; i < WUBU_Q3_K_BLOCK_SIZE; ++i) blk[b][i] = (uint8_t)rng_u32();
    }

    float worst = 0.0f;
    for (int b = 0; b < NBLK; ++b) {
        for (int j = 0; j < QK_K; ++j)
            x[j] = ((float)(int32_t)(rng_u32() & 0xFFFF) - 32768.0f) / 4096.0f;
        quantize_row_q8_K(x, q8, QK_K);

        float got = 0.0f;
        CHECK(wubu_vec_dot_q3_k_q8_k(QK_K, &got, blk[b], q8) == 0);

        float sc = 0.0f;
        CHECK(wubu_vec_dot_q3_k_q8_k_scalar(QK_K, &sc, blk[b], q8) == 0);
        CHECK(close_enough(got, sc));

        gguf_dequantize(blk[b], GGML_TYPE_Q3_K, QK_K, w_f32[b]);
        double acc = 0.0;
        for (int j = 0; j < QK_K; ++j)
            acc += (double)w_f32[b][j] * (double)q8[0].d * (double)q8[0].qs[j];
        const float want = (float)acc;
        const float rel = fabsf(got - want) / (1.0f + fabsf(want));
        if (rel > worst) worst = rel;
        CHECK(close_enough(got, want));
    }

    {
        float s = 3.0f;
        CHECK(wubu_vec_dot_q3_k_q8_k(QK_K, NULL, blk[0], q8) == -1);
        CHECK(wubu_vec_dot_q3_k_q8_k(QK_K, &s, NULL, q8) == -1);
        CHECK(wubu_vec_dot_q3_k_q8_k(QK_K, &s, blk[0], NULL) == -1);
        CHECK(wubu_vec_dot_q3_k_q8_k(64, &s, blk[0], q8) == -1);
        CHECK(s == 3.0f);
    }

    printf("EG-114 q3_k_fused blocks=%d worst_rel=%.3g\n", NBLK, worst);
    if (failures) { fprintf(stderr, "test_eg114_q3_k: FAIL (%d)\n", failures); return 1; }
    printf("test_eg114_q3_k: PASS\n");
    return 0;
}
