#ifndef WUBU_MOBIUS_H
#define WUBU_MOBIUS_H

/**
 * Möbius gyrovector operations for Poincaré ball hyperbolic geometry.
 *
 * All operations work in the Poincaré ball of radius R (curvature = -1/R^2).
 * The ball is defined as { x in R^n : ||x|| < R }.
 *
 * Key formulas (for curvature c = -1/R^2, equivalently κ = 1/R):
 * - Möbius addition:  x ⊕ y = (1 + 2⟨x,y⟩/R² + ||y||²/R²) x + (1 - ||x||²/R²) y
 *                                  1 + 2⟨x,y⟩/R² + ||x||²||y||²/R⁴
 * - Möbius scalar multiplication: r ⊗ x = exp_map(r * log_map(x))
 * - Geodesic (Poincaré) distance: d(x,y) = R * artanh(||(-x) ⊕ y|| / R)
 * - exp_map(v): project tangent vector v onto the ball
 * - log_map(x): inverse of exp_map
 *
 * References:
 * - Ganea et al. "Hyperbolic Neural Networks" (NeurIPS 2018)
 * - Ungar "Analytic Hyperbolic Geometry" (2005)
 */

#include <math.h>
#include <stdint.h>

// ============================================================
// Möbius addition: z = x ⊕ y
// Performs gyrovector addition in the Poincaré ball.
// x, y are d-dimensional vectors, result in z.
// Uses the numerically stable formula from Ganea et al. (2018).
// ============================================================
void wubu_mobius_add(const float *x, const float *y, int d, float R, float *z);

// ============================================================
// Möbius addition backward (vector-Jacobian product)
// Computes dx, dy given dz for z = x ⊕ y.
// Pass NULL for dx or dy to skip that gradient.
// ============================================================
void wubu_mobius_add_backward(const float *x, const float *y, int d, float R,
                               const float *z, const float *dz,
                               float *dx, float *dy);

// ============================================================
// exp_map: v → z = tanh(||v||/R) * v/||v|| * R
// Projects tangent vector v onto the Poincaré ball.
// ============================================================
void wubu_exp_map(const float *v, int d, float R, float *z);

// ============================================================
// log_map: x → v = R * artanh(||x||/R) * x/||x||
// Inverse of exp_map. Maps ball point to tangent space.
// ============================================================
void wubu_log_map(const float *x, int d, float R, float *v);

// ============================================================
// exp_map backward: v → z = tanh(||v||/R) * v/||v|| * R
// Computes dv given dz.
// ============================================================
void wubu_exp_map_backward(const float *v, int d, float R,
                            const float *z, const float *dz,
                            float *dv);

// ============================================================
// log_map backward: x → v = R * artanh(||x||/R) * x/||x||
// Computes dx given dv.
// ============================================================
void wubu_log_map_backward(const float *x, int d, float R,
                            const float *v, const float *dv,
                            float *dx);

// ============================================================
// Möbius scalar multiplication backward: z = r ⊗ x
// Computes dx given dz.
// Pass NULL for dx to skip.
// ============================================================
void wubu_mobius_scalar_mul_backward(float r, const float *x, int d, float R,
                                      const float *z, const float *dz,
                                      float *dx);

// ============================================================
// Möbius scalar multiplication: z = r ⊗ x
// Scales a point in the Poincaré ball by scalar r.
// Implemented as: exp_map(r * log_map(x))
// ============================================================
void wubu_mobius_scalar_mul(float r, const float *x, int d, float R, float *z);

// ============================================================
// Poincaré geodesic distance: d(x, y)
// Distance along the hyperbolic geodesic between x and y.
// d(x,y) = R * artanh(||(-x) ⊕ y|| / R)
// ============================================================
float wubu_poincare_dist(const float *x, const float *y, int d, float R);

// ============================================================
// Tangent-space linear combination (Poincaré I):
// z = exp_map(sum_i w_i * log_map(x_i))
//
/// This avoids composing Möbius additions by doing the linear
// combination in the tangent space (where it's Euclidean) and
// projecting back. More stable than Möbius add chains.
// ============================================================
void wubu_poincare_linear_comb(const float **xi, const float *wi, int n, int d, float R, float *z);

// ============================================================
// Möbius gyration operator: gyr[x,y]z
// The gyration operator captures the non-associativity of
// Möbius addition. In the Poincaré ball:
// x ⊕ y = gyr[x,y](y ⊕ x)
// For most applications we use gyr[x,y] to adjust weight
// directions:
//   w_gyrated = gyr[-p, v](w)
// where p is the current point and v is the update direction.
//
// Not yet implemented — Phase 2.2 only needs Poincaré linear comb.
// ============================================================
void wubu_mobius_gyrate(const float *x, const float *y, const float *z, int d, float R, float *out);

// ============================================================
// Poincaré norm: ||x|| (induced norm from the Poincaré metric)
// Standard L2 norm in the ambient Euclidean space.
// Always in [0, R) for valid ball elements.
// ============================================================
static inline float wubu_poincare_norm(const float *x, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; i++) sum += x[i] * x[i];
    return sqrtf(sum);
}

// ============================================================
// Conformal (Lorentz) factor: λ_x = 2*R^2 / (R^2 - ||x||^2)
// Scales Euclidean geometry at point x to hyperbolic geometry.
// Approaches infinity as ||x|| → R (boundary).
// ============================================================
static inline float wubu_lorentz_factor(const float *x, int d, float R) {
    float nx = wubu_poincare_norm(x, d);
    float R2 = R * R;
    float denom = R2 - nx * nx;
    // Clamp to prevent division by tiny values
    if (denom < 1e-12f) denom = 1e-12f;
    return 2.0f * R2 / denom;
}

#endif // WUBU_MOBIUS_H
