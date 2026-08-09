#ifndef LFM2_FFN_H
#define LFM2_FFN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* LFM2.5 SwiGLU FFN: h = w2( silu(w1(x)) * w3(x) ).
 * Self-contained. x: [T, d_model]. Writes out: [T, d_model]. */
void lfm2_ffn(const float *w1, const float *w2, const float *w3,
              int ff_dim, int d_model, const float *x, int T, float *out);

void lfm2_ffn_q(const float *w1, const float *w2, const float *w3,
                int ff_dim, int d_model, const float *x, int T, float *out,
                const uint8_t *q_w1, int q_w1_t,
                const uint8_t *q_w2, int q_w2_t,
                const uint8_t *q_w3, int q_w3_t);

#ifdef __cplusplus
}
#endif

#endif /* LFM2_FFN_H */
