/* lfm2_conv.c -- LFM2.5 gated depthwise-causal-conv block (C11, self-contained).
 * SPDX-License-Identifier: WaefreBeorn-UMV3 */
#include "lfm2_conv.h"
#include "lfm2_math.h"
#include <stdlib.h>
#include <string.h>

static void depthwise_causal_conv(const float *x, const float *w,
                                  int T, int C, int k, float *y) {
    /* PyTorch Conv1d(hidden, hidden, k, padding=k-1, groups=hidden):
     *   out[t,c] = sum_{j=0..k-1} w[c,(c),k-1-j] * in[t-j, c]   (in padded 0 for t<0)
     * Depthwise: output channel c == input channel c.
     * w layout [C, k]: element (c, j) at c*k + j. wc[k-1] = current tap. */
    int last = k - 1;
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            float s = 0.0f;
            const float *wc = w + (size_t)c * k;
            for (int j = 0; j < k; j++) {
                int tt = t - j;
                if (tt >= 0) s += wc[last - j] * x[(size_t)tt * C + c];
            }
            y[(size_t)t * C + c] = s;
        }
    }
}

/* Incremental (T=1) causal conv using the saved (k-1) previous gated
 * inputs. state[j][c] = y at pos-1-j. The new y is pushed in after. */
static void depthwise_causal_conv_step(const float *y_new, const float *w,
                                       int C, int k, const float *state,
                                       float *z) {
    int last = k - 1;
    for (int c = 0; c < C; c++) {
        const float *wc = w + (size_t)c * k;
        float s = wc[last] * y_new[c];              /* current tap */
        for (int j = 1; j < k; j++)
            s += wc[last - j] * state[(size_t)(j - 1) * C + c];
        z[c] = s;
    }
}

void lfm2_conv(const float *in_proj, const float *conv_w, const float *out_proj,
               int conv_k, int conv_dim, int d_model,
               const float *x, int T, float *op_out) {
    lfm2_conv_q(in_proj, conv_w, out_proj, conv_k, conv_dim, d_model, x, T,
                op_out, NULL, 0, NULL, 0, NULL, 0);
}

void lfm2_conv_q(const float *in_proj, const float *conv_w, const float *out_proj,
                 int conv_k, int conv_dim, int d_model,
                 const float *x, int T, float *op_out,
                 const uint8_t *q_in_proj, int q_in_t,
                 const uint8_t *q_out_proj, int q_out_t,
                 float *conv_state, int state_pos) {
    int cd = conv_dim, d = d_model, k = conv_k;
    int k1 = (k > 1) ? (k - 1) : 1;
    float *proj = (float *)malloc((size_t)T * 3 * cd * sizeof(float));
    lfm2_qmatmul(x, in_proj, q_in_proj, q_in_t, T, d, 3 * cd, proj);

    /* split B, C, h_tilde -- each [T, cd] */
    const float *Bp = proj;
    const float *Cp = proj + (size_t)T * cd;
    const float *Hp = proj + (size_t)T * 2 * cd;

    float *y = (float *)malloc((size_t)T * cd * sizeof(float));
    for (size_t i = 0; i < (size_t)T * cd; i++) y[i] = Bp[i] * Hp[i]; /* input gate */

    float *z = (float *)malloc((size_t)T * cd * sizeof(float));
    if (conv_state && T == 1 && state_pos > 0) {
        /* incremental: causal conv from the saved (k-1) past gated inputs */
        depthwise_causal_conv_step(y, conv_w, cd, k, conv_state, z);
        /* shift the ring: state[0] = y_now; state[j] = state[j-1] */
        memmove(conv_state + (size_t)1 * cd, conv_state, (size_t)(k1 - 1) * cd * sizeof(float));
        memcpy(conv_state, y, (size_t)cd * sizeof(float));
    } else {
        depthwise_causal_conv(y, conv_w, T, cd, k, z);
        if (conv_state) {
            /* save the last k-1 gated inputs for the next incremental step;
             * positions before t=0 are causal padding (0) */
            memcpy(conv_state, y + (size_t)(T - 1) * cd, (size_t)cd * sizeof(float));
            for (int j = 1; j < k1; j++) {
                if (T - 1 - j >= 0)
                    memcpy(conv_state + (size_t)j * cd,
                           y + (size_t)(T - 1 - j) * cd, (size_t)cd * sizeof(float));
                else
                    memset(conv_state + (size_t)j * cd, 0, (size_t)cd * sizeof(float));
            }
        }
    }

    float *gated = (float *)malloc((size_t)T * cd * sizeof(float));
    for (size_t i = 0; i < (size_t)T * cd; i++) gated[i] = Cp[i] * z[i]; /* output gate */

    lfm2_qmatmul(gated, out_proj, q_out_proj, q_out_t, T, cd, d, op_out);

    free(proj); free(y); free(z); free(gated);
}
