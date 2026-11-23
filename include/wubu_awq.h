/*
 * wubu_awq.h -- AWQ activation-aware weight quantization (doc B05).
 *
 * Source: Lin et al., "AWQ", MLSys 2024.
 *
 * AWQ scales the top 1% salient channels (by activation magnitude) UP
 * before weight quantization, and scales the corresponding activations
 * DOWN by the same factor. This preserves the matmul result while reducing
 * quantization error on the important channels.
 *
 * Self-contained C11, no third-party deps.
 */

#ifndef WUBU_AWQ_H
#define WUBU_AWQ_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Identify salient channels: top fraction% by activation magnitude.
 * out_salient[i] = true if channel i is in the top fraction. */
void wubu_awq_find_salient(const float *act_magnitudes, int n,
                            float fraction, bool *out_salient);

/* Compute per-channel scaling factors for AWQ.
 * alpha controls the trade-off (0 = no scaling, 1 = full AWQ).
 * Salient channels get scale > 1, non-salient get scale = 1. */
void wubu_awq_compute_scales(const float *act_magnitudes,
                               const float *weight_magnitudes,
                               int n, float alpha,
                               float *out_scales);

/* Apply AWQ scaling to weights: W[i] *= scale[i] */
void wubu_awq_apply_scale_weights(float *W, const float *scales, int n);

/* Apply AWQ inverse scaling to activations: x[i] /= scale[i] */
void wubu_awq_apply_scale_activations(float *x, const float *scales, int n);

/* EG-84: uint4 affine group quantization (per-group scale + zero point). */
#define WUBU_AWQ_U4_MAX_W 4096

int wubu_awq_u4_pack(const float *w, int n, int group,
                     unsigned char *codes, float *scales, int *zeros);
int wubu_awq_u4_unpack(const unsigned char *codes, const float *scales,
                       const int *zeros, int n, int group, float *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_AWQ_H */
