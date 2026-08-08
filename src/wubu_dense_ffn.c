/* wubu_dense_ffn.c — dense SwiGLU FFN for hybrid models (C11). */

#include "wubu_dense_ffn.h"
#include "gguf_reader.h" /* declares quantized_matmul */
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct wubu_dense_ffn {
    const uint8_t *gate_q;
    int gate_type;
    const uint8_t *up_q;
    int up_type;
    const uint8_t *down_q;
    int down_type;
    int d_model;
    int d_ff;
};

wubu_dense_ffn *wubu_dense_ffn_create(
    const uint8_t *gate_q, int gate_type,
    const uint8_t *up_q,   int up_type,
    const uint8_t *down_q, int down_type,
    int d_model, int d_ff)
{
    if (d_model <= 0 || d_ff <= 0) return NULL;
    wubu_dense_ffn *f = calloc(1, sizeof(*f));
    if (!f) return NULL;
    f->gate_q = gate_q; f->gate_type = gate_type;
    f->up_q = up_q;     f->up_type = up_type;
    f->down_q = down_q; f->down_type = down_type;
    f->d_model = d_model;
    f->d_ff = d_ff;
    return f;
}

void wubu_dense_ffn_free(wubu_dense_ffn *f)
{
    free(f); /* weights are zero-copy, not owned */
}

int wubu_dense_ffn_ready(const wubu_dense_ffn *f)
{
    return (f && f->gate_q && f->up_q && f->down_q) ? 1 : 0;
}

void wubu_dense_ffn_forward(wubu_dense_ffn *f,
                            const float *x, float *y)
{
    if (!wubu_dense_ffn_ready(f) || !x || !y) return;
    const int d = f->d_model;
    const int ff = f->d_ff;
    float *gate = malloc((size_t)ff * sizeof(float));
    float *up = malloc((size_t)ff * sizeof(float));
    if (!gate || !up) { free(gate); free(up); return; }

    /* gate = x @ gate_q^T, up = x @ up_q^T (both [ff, d] quantized).
     * col_stride_bytes=0 -> the callee uses n_rows (row-major). */
    quantized_matmul(x, f->gate_q, f->gate_type, d, ff, 0, gate);
    quantized_matmul(x, f->up_q, f->up_type, d, ff, 0, up);

    /* SiLU gate: g = g * sigmoid(g); then h = g * up */
    for (int i = 0; i < ff; i++) {
        float g = gate[i];
        g = g / (1.0f + expf(-g)); /* SiLU */
        gate[i] = g * up[i];
    }

    /* y = h @ down_q^T (down is [d, ff]: ff inputs, d outputs) */
    quantized_matmul(gate, f->down_q, f->down_type, ff, d, 0, y);

    free(gate);
    free(up);
}
