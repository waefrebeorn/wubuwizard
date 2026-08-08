#ifndef WUBU_DENSE_FFN_H
#define WUBU_DENSE_FFN_H

/* wubu_dense_ffn.h — dense SwiGLU FFN for hybrid models (C11).
 *
 * Qwen3.5-family GGUFs use DENSE ffn_gate/up/down (no MoE exps).
 * The engine's MoE-only forward has no dense path — this module
 * fills the gap: a zero-copy quantized dense FFN using
 * quantized_matmul on the mmap'd blob pointers (no dequant).
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_dense_ffn wubu_dense_ffn;

/* Create from quantized blob pointers (zero-copy, not owned). */
wubu_dense_ffn *wubu_dense_ffn_create(
    const uint8_t *gate_q, int gate_type,
    const uint8_t *up_q,   int up_type,
    const uint8_t *down_q, int down_type,
    int d_model, int d_ff);

void wubu_dense_ffn_free(wubu_dense_ffn *f);

/* Forward: y = SwiGLU(x @ gate^T) @ down^T, where
 * SwiGLU(a,b) = a * sigmoid(a) * b  (SiLU gate * up).
 * x and y are d_model floats each (single token). */
void wubu_dense_ffn_forward(wubu_dense_ffn *f,
                            const float *x, float *y);

/* 1 when all three weight pointers are present. */
int wubu_dense_ffn_ready(const wubu_dense_ffn *f);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_DENSE_FFN_H */
