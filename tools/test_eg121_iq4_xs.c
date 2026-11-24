/* test_eg121_iq4_xs.c -- EG-121 gate: fused IQ4_XS x Q8_K dot.
 *
 * Real IQ4_XS blocks from the 12B Gemma GGUF, activation row quantised with the
 * engine's quantize_row_q8_K, reference = gguf_dequantize -> F32 dotted against the
 * dequantised Q8_K row. The wired scalar path (iq4_xs_vec_dot) is checked against the
 * same reference so both stay honest.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gguf_reader.h"
#include "wubu_vecdot_iq4_xs.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define QK_K 256

static float ref_dot(const float *w_f32, const block_q8_K *q8, int nb)
{
    double acc = 0.0;
    for (int b = 0; b < nb; ++b) {
        const double d = (double)q8[b].d;
        for (int j = 0; j < QK_K; ++j) acc += (double)w_f32[b * QK_K + j] * d * (double)q8[b].qs[j];
    }
    return (float)acc;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1]
        : "C:/Users/eman5/models/gemma-12b/gemma-4-12b-it-UD-IQ2_M.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { printf("test_eg121: SKIP (no %s)\n", path); return 0; }
    if (!ctx->data_blob && !gguf_buffer_data(ctx)) { fprintf(stderr, "cannot map blob\n"); return 1; }

    const gguf_tensor_info *pick = NULL;
    for (int64_t i = 0; i < ctx->n_tensors; ++i) {
        const gguf_tensor_info *t = &ctx->tensors[i];
        if (t->ggml_type != GGML_TYPE_IQ4_XS) continue;
        int64_t n = 1;
        for (int d = 0; d < t->n_dims; ++d) n *= t->dims[d];
        if (n >= 32 * QK_K) { pick = t; break; }
    }
    CHECK(pick != NULL);
    if (!pick) { gguf_close(ctx); return 1; }

    enum { NB = 32 };
    const uint8_t *raw = (const uint8_t *)ctx->data_blob + pick->data_offset;
    static float w_f32[NB * QK_K];
    gguf_dequantize(raw, GGML_TYPE_IQ4_XS, (int64_t)NB * QK_K, w_f32);

    static float x[NB * QK_K];
    static block_q8_K q8[NB];
    unsigned s = 99991u;
    for (int i = 0; i < NB * QK_K; ++i) {
        s = s * 1103515245u + 12345u;
        x[i] = (float)((int)((s >> 16) & 0x3FF) - 512) / 210.0f;
    }
    quantize_row_q8_K(x, q8, (int64_t)NB * QK_K);

    const float ref = ref_dot(w_f32, q8, NB);
    const double got = (double)wubu_vec_dot_iq4_xs_q8_K(raw, q8, NB * QK_K);
    const double rel = fabs(ref) > 1e-12 ? fabs(got - (double)ref) / fabs((double)ref) : 1.0;
    printf("EG-121 %s: ref=%.6f fused=%.6f rel=%.2e\n", pick->name, ref, (float)got, rel);
    CHECK(rel < 1e-5);

    {
        extern void iq4_xs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx,
                                   const void *vy, size_t by, int nrc);
        float wired = 0.0f;
        iq4_xs_vec_dot(NB * QK_K, &wired, sizeof(float), raw, 0, q8, 0, 1);
        const double wrel = fabs(ref) > 1e-12 ? fabs((double)wired - (double)ref) / fabs((double)ref) : 1.0;
        printf("EG-121 wired=%.6f rel=%.2e\n", (double)wired, wrel);
        CHECK(wrel < 1e-5);
    }

    gguf_close(ctx);
    if (failures) { fprintf(stderr, "test_eg121_iq4_xs: FAIL (%d)\n", failures); return 1; }
    printf("test_eg121_iq4_xs: PASS\n");
    return 0;
}
