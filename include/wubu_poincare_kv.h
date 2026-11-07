/* wubu_poincare_kv.h — hyperbolic geometry for the KV namespace (AN22 #8)
 *
 * The KV cache is a file system whose paths form a *space*: a hyperbolic
 * (Poincare-ball) embedding in which structurally related files — those
 * sharing namespace prefixes, e.g. two files in the same research pond —
 * land closer together than unrelated files. This gives the KV-FS a
 * built-in notion of "these regions are neighbors," usable for semantic
 * routing, eviction locality, and gyro-attention (research/060 INDEX AN08
 * hyperbolic track; wubu1-improvement-plan module #8).
 *
 * Each /kv/in/<path> maps to a point in the unit ball via feature-hashed
 * path segments (deterministic, no learned weights). The Poincare distance
 * is the hyperbolic geodesic; curvature of a region is 1 - |p|^2.
 *
 * C11, opaque value type, no third-party deps.
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_POINCARE_KV_H
#define WUBU_POINCARE_KV_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_POINCARE_DIM  16

/* A point in the open unit ball of the Poincare disk (value type). */
typedef struct {
    double x[WUBU_POINCARE_DIM];
    double radius;  /* ||p|| (0 <= radius < 1), precomputed */
} poincare_pos_t;

/* Map a namespace path (e.g. "/kv/in/pond-a/attn.txt") to a ball point.
 * Segments are feature-hashed into WUBU_POINCARE_DIM and L2-normalized to
 * radius 0.8, so related paths (shared prefixes) land close. Returns 0 on
 * success, -1 on invalid input. */
int wubu_poincare_kv_map_path(const char *path, poincare_pos_t *out);

/* Hyperbolic geodesic distance between two ball points (Poincare metric).
 * Finite and >= 0; identical points give 0. */
double wubu_poincare_kv_distance(const poincare_pos_t *a,
                                 const poincare_pos_t *b);

/* Curvature proxy of a region: 1 - |p|^2 in (0, 1]. Higher = closer to the
 * ball's centre (more "flat"); near the boundary -> 0. */
double wubu_poincare_kv_curvature(const poincare_pos_t *p);

/* Convenience: Poincare distance between two namespace paths. */
double wubu_poincare_kv_path_distance(const char *pa, const char *pb);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_POINCARE_KV_H */
