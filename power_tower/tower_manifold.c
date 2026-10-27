#include "tower_manifold.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <string.h>

// ============================================================
// Tower Manifold Implementation
// ============================================================

void tower_manifold_init(tower_manifold_t *tower) {
    memset(tower, 0, sizeof(tower_manifold_t));
    tower->height = TOWER_HEIGHT;
    tower->hopf_dim = 4;  // S³ -> S² needs 4D quaternion
    tower->hopf_curvature = 1.0f;
    
    // Compute log-space values
    tower_compute_log_space(tower);
    
    // Assign curvatures
    tower_assign_curvatures(tower);
    
    printf("Tower manifold initialized:\n");
    printf("  Height: %d\n", tower->height);
    printf("  log(π): %.8f\n", tower->log_pi);
    printf("  π*log(π): %.8f\n", tower->pi_log_pi);
    printf("  π^π*log(π): %.8f\n", tower->pi_pi_log_pi);
    printf("  π^π^π*log(π): %.8f\n", tower->pi_pi_pi_log_pi);
    for (int i = 0; i < TOWER_HEIGHT; i++) {
        printf("  Level %d: c=%.6f, log_val=%.8f\n", i, tower->curvatures[i], tower->levels[i].log_val);
    }
}

void tower_manifold_free(tower_manifold_t *tower) {
    (void)tower;  // No dynamic allocation in current implementation
}

void tower_compute_log_space(tower_manifold_t *tower) {
    tower->log_pi = TOWER_LN_PI;
    
    // Level 1: π * ln(π)
    tower->pi_log_pi = TOWER_PI * tower->log_pi;
    tower->levels[0].log_val = tower->log_pi;
    tower->levels[1].log_val = tower->pi_log_pi;
    
    // Level 2: π^π * ln(π) = exp(π * ln(π)) * ln(π)
    tower->pi_pi_log_pi = expf(tower->pi_log_pi) * tower->log_pi;
    tower->levels[2].log_val = tower->pi_pi_log_pi;
    
    // Level 3: π^π^π * ln(π) = exp(π^π * ln(π)) * ln(π)
    tower->pi_pi_pi_log_pi = expf(tower->pi_pi_log_pi) * tower->log_pi;
    tower->levels[3].log_val = tower->pi_pi_pi_log_pi;
    
    // Store in levels array
    tower->levels[0].level = 0;
    tower->levels[1].level = 1;
    tower->levels[2].level = 2;
    tower->levels[3].level = 3;
}

void tower_assign_curvatures(tower_manifold_t *tower) {
    // Curvature assignment strategy:
    // Higher tower levels = larger values = tighter curvature (smaller radius)
    // c = k * exp(log_val) where k is a scaling factor
    // Use logarithmic scaling to keep curvatures reasonable
    
    float scale = 0.1f;  // Base curvature scale
    
    for (int i = 0; i < TOWER_HEIGHT; i++) {
        // Log-space curvature: c = scale * log(1 + log_val)
        float log_val = tower->levels[i].log_val;
        float c = scale * logf(1.0f + log_val);
        c = fmaxf(c, 1e-4f);  // Minimum curvature
        c = fminf(c, 10.0f);  // Maximum curvature
        
        tower->curvatures[i] = c;
        tower->levels[i].curvature = c;
        tower->levels[i].radius = 1.0f / sqrtf(c);
        tower->levels[i].exp_map_scale = 1.0f;
    }
    
    // Special: Level 0 (π) gets base curvature
    tower->levels[0].curvature = 1.0f;
    tower->levels[0].radius = 1.0f;
    tower->curvatures[0] = 1.0f;
}

void tower_level_to_poincare(const tower_manifold_t *tower, int level, 
                             const float *v, float *y, int dim) {
    if (level < 0 || level >= TOWER_HEIGHT) {
        fprintf(stderr, "Invalid tower level: %d\n", level);
        return;
    }
    
    float c = tower->curvatures[level];
    float scale = tower->levels[level].exp_map_scale;
    
    // Use exponential map from wubu_math.h
    wubu_exp_map(v, y, dim, c, scale);
}

void tower_to_nested_manifold(const tower_manifold_t *tower,
                              float **sphere_points, int dim) {
    for (int i = 0; i < TOWER_HEIGHT; i++) {
        // Create log-space vector for this level
        float *v = (float *)malloc(dim * sizeof(float));
        memset(v, 0, dim * sizeof(float));
        
        // The log-space value at this level determines the "angle" on the sphere
        // Map log_val to a direction to a point on the sphere via direction
        float log_val = tower->levels[i].log_val;
        
        // Create a direction vector from the log value
        // Use multiple dimensions by spreading the value
        for (int j = 0; j < dim; j++) {
            // Create structured direction based on level and dimension
            float angle = log_val * (j + 1) * 0.1f;
            v[j] = sinf(angle) * 0.1f;  // Small magnitude for ball interior
        }
        
        // Map to Poincaré ball
        tower_level_to_poincare(tower, i, v, sphere_points[i], dim);
        
        free(v);
    }
}

// Intersection invariant: ∫_M ω₀ ∧ ω₁ ∧ ω₂ ∧ ω₃
// For nested Poincaré spheres, this is the product of Chern numbers
// ω_i = (i/2π) * curvature_i * vol_form
// Chern number of S² = 1, so intersection number = product of normalized curvatures
float tower_intersection_invariant(const tower_manifold_t *tower) {
    float invariant = 1.0f;
    
    for (int i = 0; i < TOWER_HEIGHT; i++) {
        // Normalized curvature: c_i / c_0
        float normalized_c = tower->curvatures[i] / tower->curvatures[0];
        invariant *= normalized_c;
    }
    
    // Multiply by base Chern number (1 for S²)
    invariant *= 1.0f;
    
    return invariant;
}

// Fractional part estimation via Monte Carlo
// {log log log log N} where N = π^(π^(π^π))
void tower_fractional_part_estimate(const tower_manifold_t *tower,
                                    int num_samples,
                                    float *mean_fractional,
                                    float *std_fractional) {
    if (num_samples <= 0) {
        *mean_fractional = 0.0f;
        *std_fractional = 0.0f;
        return;
    }
    
    // The fractional part we want: {log log log log N}
    // N = π^(π^(π^π))
    // log N = π^(π^π) * log π
    // log log N = log(π^(π^π) * log π) = π^π * log π + log(log π)
    // log log log N = log(π^π * log π + log(log π))
    // log log log log N = log(log(π^π * log π + log(log π)))
    
    // This is what we need to estimate the fractional part of
    // We use Monte Carlo on the manifold to estimate this
    
    double sum = 0.0;
    double sum_sq = 0.0;
    
    for (int i = 0; i < num_samples; i++) {
        // Sample a point on the nested manifold
        // Use Hopf fibration sampling for structured randomness
        
        // For each level, sample a point on the Poincaré sphere
        float total_fractional = 0.0f;
        
        for (int level = 0; level < TOWER_HEIGHT; level++) {
            // Sample direction on sphere using Hopf fibration
            // Quaternion-based sampling
            float q[4];
            for (int j = 0; j < 4; j++) {
                // Use level-dependent seed for reproducibility
                unsigned int seed = (level + 1) * (i + 1) * 12345 + j;
                float r = ((seed * 1103515245 + 12345) % 2147483648) / 2147483648.0f;
                q[j] = (r - 0.5f) * 2.0f;
            }
            
            // Normalize quaternion
            float q_norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
            for (int j = 0; j < 4; j++) q[j] /= q_norm;
            
            // Project to S² via Hopf map
            float v[3];
            v[0] = 2.0f * (q[0]*q[2] + q[1]*q[3]);
            v[1] = 2.0f * (q[2]*q[3] - q[0]*q[1]);
            v[2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
            
            // Map to Poincaré ball
            float y[3];
            tower_level_to_poincare((tower_manifold_t*)tower, level, v, y, 3);
            
            // Compute contribution to fractional part
            float norm = wubu_norm(y, 3);
            float frac = fmodf(norm * 1000.0f, 1.0f);
            total_fractional += frac;
        }
        
        // The fractional part is the total mod 1
        float sample_fractional = fmodf(total_fractional, 1.0f);
        sum += sample_fractional;
        sum_sq += sample_fractional * sample_fractional;
    }
    
    double mean = sum / num_samples;
    double variance = sum_sq / num_samples - mean * mean;
    double std = sqrt(fmax(0.0, variance));
    
    *mean_fractional = (float)mean;
    *std_fractional = (float)std;
}

// Analytical fractional part computation (for validation)
// Computes {log log log log N} using high-precision log-space arithmetic
float tower_analytical_fractional(const tower_manifold_t *tower) {
    // N = π^(π^(π^π))
    // Compute in log space using double precision
    double ln_pi = log(M_PI);
    
    // log N = π^(π^π) * ln π
    double pi_pi = exp(M_PI * ln_pi);           // π^π
    double pi_pi_pi = exp(pi_pi * ln_pi);       // π^π^π
    double log_N = pi_pi_pi * ln_pi;            // log N
    
    double log_log_N = log(log_N);              // log log N
    double log_log_log_N = log(log_log_N);      // log log log N
    double log_log_log_log_N = log(log_log_log_N); // log log log log N
    
    // Fractional part
    double frac = log_log_log_log_N - floor(log_log_log_log_N);
    
    return (float)frac;
}

#ifdef __cplusplus
}
#endif
