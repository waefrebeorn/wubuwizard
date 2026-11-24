/* test_eg120_iq3_xxs.c -- EG-120 gate: fused IQ3_XXS x Q8_K dot.
 *
 * Uses REAL IQ3_XXS blocks taken from the 12B Gemma GGUF (not synthetic bytes, so the
 * grid indices and sign/scale bytes are the ones the model actually carries) and a
 * synthetic activation row quantised with the engine's own quantize_row_q8_K.
 *
 * Reference: gguf_dequantize() -> F32 weights dotted against the dequantised Q8_K row
 * (d * qs[j]). Agreement to float rounding proves the grid lookup, the 7-bit sign
 * masks, the per-32 scale (2*hi+1) and the block scale are wired the same way.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gguf_reader.h"
#include "wubu_vecdot_iq3_xxs.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define QK_K 256
#define BLK_BYTES 98

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1]
        : "C:/Users/eman5/models/gemma-12b/gemma-4-12b-it-UD-IQ2_M.gguf";

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { printf("test_eg120: SKIP (no %s)\n", path); return 0; }
    if (!ctx->data_blob && !gguf_buffer_data(ctx)) {
        fprintf(stderr, "test_eg120: cannot map data blob\n");
        return 1;
    }

    /* Find a real IQ3_XXS tensor with at least 64 blocks (16K elements). */
    const gguf_tensor_info *pick = NULL;
    for (int64_t i = 0; i < ctx->n_tensors; ++i) {
        const gguf_tensor_info *t = &ctx->tensors[i];
        if (t->ggml_type != GGML_TYPE_IQ3_XXS) continue;
        int64_t n = 1;
        for (int d = 0; d < t->n_dims; ++d) n *= t->dims[d];
        if (n >= 64 * QK_K) { pick = t; break; }
    }
    CHECK(pick != NULL);
    if (!pick) { gguf_close(ctx); return 1; }

    enum { NB = 64 };
    const uint8_t *raw = (const uint8_t *)ctx->data_blob + pick->data_offset;

    static float w_f32[NB][QK_K];
    gguf_dequantize(raw, GGML_TYPE_IQ3_XXS, (int64_t)NB * QK_K, &w_f32[0][0]);

    /* Activation row + its Q8_K blocks. */
    static float x[NB * QK_K];
    static block_q8_K q8[NB];
    unsigned s = 12345u;
    for (int i = 0; i < NB * QK_K; ++i) {
        s = s * 1103515245u + 12345u;
        x[i] = (float)((int)((s >> 16) & 0x3FF) - 512) / 180.0f;
    }
    quantize_row_q8_K(x, q8, (int64_t)NB * QK_K);

    /* Reference: dequantised weights against the dequantised activation row. */
    double acc = 0.0;
    for (int b = 0; b < NB; ++b) {
        const double d = (double)q8[b].d;
        for (int j = 0; j < QK_K; ++j)
            acc += (double)w_f32[b][j] * d * (double)q8[b].qs[j];
    }
    const double ref = acc;
    const double got = (double)wubu_vec_dot_iq3_xxs_q8_K(raw, q8, NB * QK_K);
    const double denom = fabs(ref) > 1e-12 ? fabs(ref) : 1.0;
    const double rel = fabs(got - ref) / denom;

    printf("EG-120 %s: ref=%.6f scalar=%.6f rel=%.2e\n", pick->name, ref, got, rel);
    CHECK(rel < 1e-5);

    /* The type is already wired to iq3_xxs_vec_dot (AVX2 when built with -mavx2).
     * That path had no independent gate; check it against the same reference and
     * against the scalar port so an AVX2-only bug cannot hide. */
    {
        extern void iq3_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx,
                                    const void *vy, size_t by, int nrc);
        static float x2[NB * QK_K];
        static block_q8_K q8b[NB];
        unsigned s2 = 12345u;
        for (int i = 0; i < NB * QK_K; ++i) {
            s2 = s2 * 1103515245u + 12345u;
            x2[i] = (float)((int)((s2 >> 16) & 0x3FF) - 512) / 180.0f;
        }
        quantize_row_q8_K(x2, q8b, (int64_t)NB * QK_K);
        float wired = 0.0f;
        iq3_xxs_vec_dot(NB * QK_K, &wired, sizeof(float), raw, 0, q8b, 0, 1);
        double acc2 = 0.0;
        for (int b = 0; b < NB; ++b) {
            const double d = (double)q8b[b].d;
            for (int j = 0; j < QK_K; ++j) acc2 += (double)w_f32[b][j] * d * (double)q8b[b].qs[j];
        }
        const double wired_rel = fabs(acc2) > 1e-12 ? fabs((double)wired - acc2) / fabs(acc2) : 1.0;
        printf("EG-120 wired(AVX2)=%.6f rel=%.2e\n", (double)wired, wired_rel);
        CHECK(wired_rel < 1e-5);
    }

    /* Zero activation must give exactly zero, not a rounding leftover. */
    memset(q8, 0, sizeof(q8));
    CHECK(wubu_vec_dot_iq3_xxs_q8_K(raw, q8, NB * QK_K) == 0.0f);

    gguf_close(ctx);
    if (failures) { fprintf(stderr, "test_eg120_iq3_xxs: FAIL (%d)\n", failures); return 1; }
    printf("test_eg120_iq3_xxs: PASS\n");
    return 0;
}
