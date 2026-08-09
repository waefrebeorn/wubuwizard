/* lfm2_ffn.c -- LFM2.5 SwiGLU feed-forward (C11, self-contained).
 * SPDX-License-Identifier: WaefreBeorn-UMV3 */
#include "lfm2_ffn.h"
#include "lfm2_math.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* set by lfm2_forward each layer; used only for LFM2_DUMP_FFN naming */
int lfm2_dump_layer = -1;

void lfm2_ffn(const float *w1, const float *w2, const float *w3,
              int ff, int d, const float *x, int T, float *out) {
    lfm2_ffn_q(w1, w2, w3, ff, d, x, T, out, NULL, 0, NULL, 0, NULL, 0);
}

void lfm2_ffn_q(const float *w1, const float *w2, const float *w3,
                int ff, int d, const float *x, int T, float *out,
                const uint8_t *q_w1, int q_w1_t,
                const uint8_t *q_w2, int q_w2_t,
                const uint8_t *q_w3, int q_w3_t) {
    float *g = (float *)malloc((size_t)T * ff * sizeof(float)); /* w1(x) */
    float *u = (float *)malloc((size_t)T * ff * sizeof(float)); /* w3(x) */
    lfm2_qmatmul(x, w1, q_w1, q_w1_t, T, d, ff, g);
    lfm2_qmatmul(x, w3, q_w3, q_w3_t, T, d, ff, u);
    if (getenv("LFM2_DUMP_FFN")) {
        char fn[128];
        snprintf(fn, sizeof(fn), "/tmp/lfm2_g_%d.bin", lfm2_dump_layer);
        FILE *fp = fopen(fn, "wb");
        if (fp) { fwrite(g, sizeof(float), (size_t)T * ff, fp); fclose(fp); }
        snprintf(fn, sizeof(fn), "/tmp/lfm2_u_%d.bin", lfm2_dump_layer);
        fp = fopen(fn, "wb");
        if (fp) { fwrite(u, sizeof(float), (size_t)T * ff, fp); fclose(fp); }
    }
    for (size_t i = 0; i < (size_t)T * ff; i++) {
        float v = g[i];
        g[i] = v / (1.0f + expf(-v)) * u[i]; /* silu(g) * u */
    }
    lfm2_qmatmul(g, w2, q_w2, q_w2_t, T, ff, d, out);
    free(g); free(u);
}
