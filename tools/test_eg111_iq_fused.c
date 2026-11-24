/* test_eg111_iq_fused.c -- EG-111 gate: fused IQ2_S/IQ3_S x Q8_K dot products.
 *
 * Independent reference: gguf_dequantize() (src/gguf_reader.c, written against
 * the llama.cpp dequantisers) turned into F32, dotted against the Q8_K
 * activations dequantised as d*qs[j]. The fused kernel must agree with that to
 * float rounding; agreement proves the block indexing, grid lookups, sign masks,
 * per-32 scales and block scale are all wired the same way.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gguf_reader.h"
#include "wubu_vecdot_iq.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define QK_K 256

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint32_t rng_u32(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

/* `near` is a legacy Windows macro (windef.h), so it cannot be a function name. */
static int close_enough(float a, float b)
{
    const float diff = fabsf(a - b);
    const float scale = fabsf(a) > fabsf(b) ? fabsf(a) : fabsf(b);
    return diff <= 1e-3f + 1e-4f * scale;
}

/* Reference dot: dequantised weights (F32) against dequantised Q8_K row. */
static float ref_dot(const float *w_f32, const block_q8_K *q8, int n)
{
    const int nb = n / QK_K;
    double acc = 0.0;
    for (int i = 0; i < nb; ++i) {
        const double d = (double)q8[i].d;
        for (int j = 0; j < QK_K; ++j)
            acc += (double)w_f32[i * QK_K + j] * d * (double)q8[i].qs[j];
    }
    return (float)acc;
}

int main(void)
{
    enum { NBLK = 64 };
    static uint8_t block_iq2[NBLK][WUBU_IQ2S_BLOCK_SIZE];
    static uint8_t block_iq3[NBLK][WUBU_IQ3S_BLOCK_SIZE];
    static float w_f32[NBLK][QK_K];
    static float x[QK_K];
    static block_q8_K q8[2];

    /* The block scale is fp16; keep it in a normal range for a meaningful test. */
    for (int b = 0; b < NBLK; ++b) {
        memset(block_iq2[b], 0, sizeof(block_iq2[b]));
        memset(block_iq3[b], 0, sizeof(block_iq3[b]));
        /* d = fp16(0.01 .. 0.05): exponent 0x2C.., mantissa low bits */
        const uint16_t d16 = (uint16_t)(0x2C00u + (rng_u32() & 0x3FFu));
        memcpy(&block_iq2[b][0], &d16, 2);
        memcpy(&block_iq3[b][0], &d16, 2);
        for (int i = 2; i < WUBU_IQ2S_BLOCK_SIZE; ++i) block_iq2[b][i] = (uint8_t)rng_u32();
        for (int i = 2; i < WUBU_IQ3S_BLOCK_SIZE; ++i) block_iq3[b][i] = (uint8_t)rng_u32();
    }

    float worst2 = 0.0f, worst3 = 0.0f;

    for (int b = 0; b < NBLK; ++b) {
        for (int j = 0; j < QK_K; ++j) x[j] = ((float)(int32_t)(rng_u32() & 0xFFFF) - 32768.0f) / 4096.0f;
        quantize_row_q8_K(x, q8, QK_K);

        float got = 0.0f;
        const int rc2 = wubu_vec_dot_iq2_s_q8_k(QK_K, &got, block_iq2[b], q8);
        CHECK(rc2 == 0);
        gguf_dequantize(block_iq2[b], GGML_TYPE_IQ2_S, QK_K, w_f32[b]);
        float want = ref_dot(w_f32[b], q8, QK_K);
        float rel = fabsf(got - want) / (1.0f + fabsf(want));
        if (rel > worst2) worst2 = rel;
        CHECK(close_enough(got, want));

        /* Vector path must agree with the scalar port of the same kernel. */
        float sc2 = 0.0f;
        CHECK(wubu_vec_dot_iq2_s_q8_k_scalar(QK_K, &sc2, block_iq2[b], q8) == 0);
        CHECK(close_enough(got, sc2));

        float got3 = 0.0f;
        const int rc3 = wubu_vec_dot_iq3_s_q8_k(QK_K, &got3, block_iq3[b], q8);
        CHECK(rc3 == 0);
        gguf_dequantize(block_iq3[b], GGML_TYPE_IQ3_S, QK_K, w_f32[b]);
        float want3 = ref_dot(w_f32[b], q8, QK_K);
        float rel3 = fabsf(got3 - want3) / (1.0f + fabsf(want3));
        if (rel3 > worst3) worst3 = rel3;
        CHECK(close_enough(got3, want3));

        float sc3 = 0.0f;
        CHECK(wubu_vec_dot_iq3_s_q8_k_scalar(QK_K, &sc3, block_iq3[b], q8) == 0);
        CHECK(close_enough(got3, sc3));
    }

    /* Two-block run: exercises the cross-block accumulate, not just block 0. */
    {
        static uint8_t two[2 * WUBU_IQ2S_BLOCK_SIZE];
        static float w2[2 * QK_K];
        memcpy(two, block_iq2[3], WUBU_IQ2S_BLOCK_SIZE);
        memcpy(two + WUBU_IQ2S_BLOCK_SIZE, block_iq2[7], WUBU_IQ2S_BLOCK_SIZE);
        for (int j = 0; j < QK_K; ++j) x[j] = ((float)(int32_t)(rng_u32() & 0xFFFF) - 32768.0f) / 2048.0f;
        quantize_row_q8_K(x, q8, 2 * QK_K);
        float got = 0.0f;
        const int rc_two = wubu_vec_dot_iq2_s_q8_k(2 * QK_K, &got, two, q8);
        CHECK(rc_two == 0);
        gguf_dequantize(two, GGML_TYPE_IQ2_S, 2 * QK_K, w2);
        float want = ref_dot(w2, q8, 2 * QK_K);
        CHECK(close_enough(got, want));
    }

    /* Fail-closed contract. */
    {
        float s = 1234.0f;
        CHECK(wubu_vec_dot_iq2_s_q8_k(QK_K, NULL, block_iq2[0], q8) == -1);
        CHECK(wubu_vec_dot_iq2_s_q8_k(QK_K, &s, NULL, q8) == -1);
        CHECK(wubu_vec_dot_iq2_s_q8_k(QK_K, &s, block_iq2[0], NULL) == -1);
        CHECK(wubu_vec_dot_iq2_s_q8_k((QK_K - 1), &s, block_iq2[0], q8) == -1);
        CHECK(wubu_vec_dot_iq3_s_q8_k(0, &s, block_iq3[0], q8) == -1);
        CHECK(s == 1234.0f);
    }

    printf("EG-111 iq_fused blocks=%d worst_rel iq2_s=%.3g iq3_s=%.3g\n", NBLK, worst2, worst3);
    if (failures) { fprintf(stderr, "test_eg111_iq_fused: FAIL (%d)\n", failures); return 1; }
    printf("test_eg111_iq_fused: PASS\n");
    return 0;
}
