/* test_eg117_iq2xs.c -- EG-117 gate: fused IQ2_XS x Q8_K dot. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gguf_reader.h"
#include "wubu_vecdot_iq2xs.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)
#define QK_K 256

static uint64_t st = 0x2A5B77ULL;
static uint32_t r32(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (uint32_t)(st >> 32); }
static int close_enough(float a, float b)
{
    const float d = fabsf(a - b);
    const float s = fabsf(a) > fabsf(b) ? fabsf(a) : fabsf(b);
    return d <= 1e-2f + 1e-3f * s;
}

int main(void)
{
    enum { NBLK = 64 };
    static uint8_t blk[NBLK][WUBU_IQ2XS_BLOCK_SIZE];
    static float wf[NBLK][QK_K];
    static float x[QK_K];
    static block_q8_K q8;

    for (int b = 0; b < NBLK; ++b) {
        const uint16_t d16 = (uint16_t)(0x2C00u + (r32() & 0x3FFu));
        memcpy(&blk[b][0], &d16, 2);
        for (int i = 2; i < WUBU_IQ2XS_BLOCK_SIZE; ++i) blk[b][i] = (uint8_t)r32();
    }

    float worst = 0.0f;
    for (int b = 0; b < NBLK; ++b) {
        for (int j = 0; j < QK_K; ++j)
            x[j] = ((float)(int32_t)(r32() & 0xFFFF) - 32768.0f) / 4096.0f;
        quantize_row_q8_K(x, &q8, QK_K);

        float got = 0.0f;
        CHECK(wubu_vec_dot_iq2_xs_q8_k(QK_K, &got, blk[b], &q8) == 0);

        gguf_dequantize(blk[b], GGML_TYPE_IQ2_XS, QK_K, wf[b]);
        double acc = 0.0;
        for (int j = 0; j < QK_K; ++j) acc += (double)wf[b][j] * (double)q8.d * (double)q8.qs[j];
        const float want = (float)acc;
        const float rel = fabsf(got - want) / (1.0f + fabsf(want));
        if (rel > worst) worst = rel;
        CHECK(close_enough(got, want));
    }

    {
        float s = 9.0f;
        CHECK(wubu_vec_dot_iq2_xs_q8_k(QK_K, NULL, blk[0], &q8) == -1);
        CHECK(wubu_vec_dot_iq2_xs_q8_k(QK_K, &s, NULL, &q8) == -1);
        CHECK(wubu_vec_dot_iq2_xs_q8_k(128, &s, blk[0], &q8) == -1);
        CHECK(s == 9.0f);
    }

    printf("EG-117 iq2_xs blocks=%d worst_rel=%.3g\n", NBLK, worst);
    if (failures) { fprintf(stderr, "test_eg117_iq2xs: FAIL (%d)\n", failures); return 1; }
    printf("test_eg117_iq2xs: PASS\n");
    return 0;
}
