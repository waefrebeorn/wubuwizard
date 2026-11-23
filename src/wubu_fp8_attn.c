/*
 * wubu_fp8_attn.c -- EG-30 FP8 attention, F32 accumulate. Pure C11.
 */
#include "wubu_fp8_attn.h"

#include "wubu_fp8.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int wubu_fp8_attn(const uint8_t *q, int nq, int d, float q_scale,
                  const uint8_t *k, int nk, float k_scale,
                  const uint8_t *v, float v_scale,
                  float *out) {
    float *scores, *probs;
    if (!q || !k || !v || !out || nq <= 0 || nk <= 0 || d <= 0 ||
        !(q_scale > 0.0f) || !(k_scale > 0.0f) || !(v_scale > 0.0f))
        return -1;

    scores = (float *)malloc((size_t)(nq * nk) * sizeof(float));
    probs = (float *)malloc((size_t)nk * sizeof(float));
    if (!scores || !probs) {
        free(scores); free(probs);
        return -1;
    }

    for (int i = 0; i < nq; i++) {
        float mx = -INFINITY;
        for (int t = 0; t < nk; t++) {
            float dot = 0.0f;
            for (int j = 0; j < d; j++) {
                float qj = wubu_fp8_e4m3_to_f32(q[i*d + j]);
                float kt = wubu_fp8_e4m3_to_f32(k[t*d + j]);
                dot += qj * kt;
            }
            scores[i*nk + t] = dot * q_scale * k_scale / sqrtf((float)d);
            if (scores[i*nk + t] > mx) mx = scores[i*nk + t];
        }
        float sum = 0.0f;
        for (int t = 0; t < nk; t++) {
            probs[t] = expf(scores[i*nk + t] - mx);
            sum += probs[t];
        }
        for (int t = 0; t < nk; t++) probs[t] /= sum;

        memset(out + i*d, 0, (size_t)d * sizeof(float));
        for (int t = 0; t < nk; t++) {
            float p = probs[t];
            for (int j = 0; j < d; j++)
                out[i*d + j] += p * wubu_fp8_e4m3_to_f32(v[t*d + j]) * v_scale;
        }
    }
    free(scores); free(probs);
    return 0;
}
