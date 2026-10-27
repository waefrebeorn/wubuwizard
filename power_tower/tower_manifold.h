#ifndef TOWER_MANIFOLD_H
#define TOWER_MANIFOLD_H

#include "../../bytropix/ENCODERS/hash-mind/c/wubu_math.h"
#include "../include/wubu_poincare_gqa.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Power Tower Manifold: π^(π^(π^π)) -> Nested Poincaré Spheres
// ============================================================
// Right-associated tower: π^(π^(π^π)) = π^(π^(π^π))
// Map each exponentiation step to a nested Poincaré sphere
// Level 0: π                    -> S² with curvature c₀
// Level 1: π^π                  -> S² with curvature c₁  
// Level 2: π^π^π                -> S² with curvature c₂
// Level 3: π^π^π^π              -> S² with curvature c₃
//
// The exponential map: a^b = exp(b * ln(a))
// In log-space: ln(π), π*ln(π), π^π*ln(π), π^π^π*ln(π)
// Each level maps to a Poincaré sphere via exponential map

#define TOWER_HEIGHT 4
#define TOWER_MAX_DIM 64  // Poincaré ball dimension (matches GQA head dim)

// Tower configuration constants
static const float TOWER_PI = 3.14159265358979323846f;
static const float TOWER_LN_PI = 1.1447298858494002f;  // ln(π)

// Tower level parameters
typedef struct {
    int level;                    // 0, 1, 2, 3
    float curvature;              // Poincaré ball curvature c
    float log_val;                // Log-space value at this level
    float radius;                 // Poincaré ball radius = 1/sqrt(c)
    float exp_map_scale;          // Scale for exponential map
} tower_level_t;

// Full tower state
typedef struct {
    tower_level_t levels[TOWER_HEIGHT];
    int height;
    
    // Precomputed log-space values
    float log_pi;           // ln(π)
    float pi_log_pi;        // π * ln(π)
    float pi_pi_log_pi;     // π^π * ln(π)
    float pi_pi_pi_log_pi;  // π^π^π * ln(π)
    
    // Poincaré ball curvatures for each level
    float curvatures[TOWER_HEIGHT];
    
    // Hopf sampling parameters
    int hopf_dim;           // S³ -> S² dimension (must be 4 for quaternion)
    float hopf_curvature;   // Curvature for Hopf fibration sampling
} tower_manifold_t;

// Tower manifold operations
void tower_manifold_init(tower_manifold_t *tower);
void tower_manifold_free(tower_manifold_t *tower);

// Map tower level to Poincaré sphere
// Input: level index (0-3), log-space vector v
// Output: point y on Poincaré ball at that level
void tower_level_to_poincare(const tower_manifold_t *tower, int level, 
                             const float *v, float *y, int dim);

// Full tower to manifold mapping
// Maps the complete tower to nested Poincaré spheres
void tower_to_nested_manifold(const tower_manifold_t *tower,
                              float **sphere_points,  // [TOWER_HEIGHT][dim]
                              int dim);

// Log-space computation for tower values
// Computes ln(π), π*ln(π), π^π*ln(π), π^π^π*ln(π) without overflow
void tower_compute_log_space(tower_manifold_t *tower);

// Curvature assignment for each tower level
// Higher levels get tighter curvature (smaller radius)
void tower_assign_curvatures(tower_manifold_t *tower);

// Integer invariant: Intersection number of curvature forms
// I = ∫_M ω₀ ∧ ω₁ ∧ ω₂ ∧ ω₃
// where ω_i are curvature 2-forms on each sphere
float tower_intersection_invariant(const tower_manifold_t *tower);

// Fractional part estimation via Monte Carlo
// Estimates {log log log log N} by sampling point clouds
void tower_fractional_part_estimate(const tower_manifold_t *tower,
                                    int num_samples,
                                    float *mean_fractional,
                                    float *std_fractional);

// Analytical fractional part computation (for validation)
float tower_analytical_fractional(const tower_manifold_t *tower);

#ifdef __cplusplus
}
#endif

#endif // TOWER_MANIFOLD_H