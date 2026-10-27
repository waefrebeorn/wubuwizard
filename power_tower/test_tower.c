#include "tower_manifold.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>

int main() {
    printf("=== Power Tower Manifold Test ===\n\n");
    
    tower_manifold_t tower;
    tower_manifold_init(&tower);
    
    printf("\n=== Analytical Fractional Part ===\n");
    float anal_frac = tower_analytical_fractional(&tower);
    printf("Analytical {log log log log N}: %.10f\n", anal_frac);
    
    printf("\n=== Intersection Invariant ===\n");
    float invariant = tower_intersection_invariant(&tower);
    printf("Intersection invariant I = %.10f\n", invariant);
    
    printf("\n=== Monte Carlo Fractional Part Estimate ===\n");
    float mean_frac, std_frac;
    tower_fractional_part_estimate(&tower, 10000, &mean_frac, &std_frac);
    printf("Monte Carlo (10k samples): mean=%.8f, std=%.8f\n", mean_frac, std_frac);
    
    printf("\n=== Sphere Points ===\n");
    int dim = 8;
    float *sphere_points[TOWER_HEIGHT];
    for (int i = 0; i < TOWER_HEIGHT; i++) {
        sphere_points[i] = (float *)malloc(dim * sizeof(float));
    }
    
    tower_to_nested_manifold(&tower, sphere_points, dim);
    
    for (int i = 0; i < TOWER_HEIGHT; i++) {
        float norm = wubu_norm(sphere_points[i], dim);
        printf("Level %d sphere point (dim=%d): norm=%.6f\n", i, dim, norm);
        free(sphere_points[i]);
    }
    
    tower_manifold_free(&tower);
    
    printf("\n=== Test Complete ===\n");
    return 0;
}
