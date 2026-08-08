/**
 * rsgd.c — Riemannian SGD optimizer for Poincaré ball parameters
 *
 * RSGD allows gradient-based optimization of parameters constrained to
 * the Poincaré ball. The key steps:
 *
 *   1. g = Euclidean gradient dL/dw (computed by standard backprop)
 *   2. g_riem = g / λ_w²           (project to tangent space at w)
 *   3. v = -lr * g_riem            (step in tangent space at w)
 *   4. w_new = exp_map_w(v)        (exponential map at point w)
 *          = w ⊕ (tanh(λ_w·||v||/2R) · v · R / (λ_w·||v||))
 *
 * Reference: Ganea et al. "Hyperbolic Neural Networks" (NeurIPS 2018)
 *            Becigneul & Ganea "Riemannian Adaptive Optimization" (ICLR 2019)
 */
#include "gguf_reader.h"
#include "wubu_mobius.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Apply one RSGD step to a batch of vectors in the Poincaré ball.
 *
 * Uses the proper Riemannian exponential map at each point w:
 *   w_new = exp_map_w(-lr * ∇_R f(w))
 *
 * w:        [n_vecs, dim]  — current parameter vectors (in ball), UPDATED IN-PLACE
 * dw:       [n_vecs, dim]  — Euclidean gradients (computed by standard backprop)
 * n_vecs:   number of vectors
 * dim:      dimension of each vector
 * lr:       learning rate
 * R:        Poincaré ball radius
 * clip:     gradient clipping factor (1.0 = no clip)
 */
void rsgd_step(float *w, const float *dw, int n_vecs, int dim,
               float lr, float R, float clip) {
    float *tmp = (float *)malloc(dim * sizeof(float));   // u = tanh(...) * v
    float *v = (float *)malloc(dim * sizeof(float));     // tangent step
    if (!tmp || !v) {
        fprintf(stderr, "RSGD: allocation failed\n");
        free(tmp); free(v);
        return;
    }

    for (int vec = 0; vec < n_vecs; vec++) {
        float *w_v = w + vec * dim;
        const float *dw_v = dw + vec * dim;

        // Compute ||w_v||²
        double n2 = 0.0;
        for (int i = 0; i < dim; i++)
            n2 += (double)w_v[i] * (double)w_v[i];

        float R2 = R * R;
        if (n2 >= R2 * 0.9999) n2 = R2 * 0.9999;  // safety clamp

        // Conformal factor at w: λ_w = 2*R² / (R² - ||w||²)
        double lambda_w;

        if (n2 >= R2 * 0.9999) {
            // Point is at/near boundary or outside ball.
            // Fall back to exp_map_0 retraction (old behavior):
            // Step in ambient space, then project to ball at origin.
            for (int i = 0; i < dim; i++) {
                double step = (double)(-lr) * (double)dw_v[i] * (double)clip;
                v[i] = (float)((double)w_v[i] + step);
            }
            wubu_exp_map(v, dim, R, w_v);
            continue;
        }

        lambda_w = 2.0 * R2 / (R2 - n2);

        // Riemannian gradient: g_riem = dw / λ_w²  (apply inverse metric)
        // Step in tangent space at w: v = -lr * g_riem * clip
        double v_norm = 0.0;
        for (int i = 0; i < dim; i++) {
            double g_riem = (double)dw_v[i] / (lambda_w * lambda_w);
            double step = (double)(-lr) * g_riem * (double)clip;
            v[i] = (float)step;
            v_norm += step * step;
        }

        v_norm = sqrt(v_norm);
        if (v_norm < 1e-30) {
            // Tiny or zero step — identity
            // w_v unchanged
            continue;
        }

        // Exponential map at point w:
        // exp_map_w(v) = w ⊕ (tanh(λ_w * ||v|| / (2*R)) * v * R / (λ_w * ||v||))
        double exp_coeff = tanh(lambda_w * v_norm / (2.0 * R)) * R / (lambda_w * v_norm);
        for (int i = 0; i < dim; i++)
            tmp[i] = (float)((double)v[i] * exp_coeff);

        // Möbius addition: w_new = w ⊕ u
        wubu_mobius_add(w_v, tmp, dim, R, w_v);  // write result back in-place
    }

    free(tmp);
    free(v);
}
