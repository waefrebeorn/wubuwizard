/* wubu_poincare_kv.c — hyperbolic geometry for the KV namespace
 *
 * Maps /kv/ paths into the open unit Poincare ball so related regions are
 * neighbors. Feature-hashes the path segments into a deterministic bag,
 * L2-normalizes to radius 0.8, and exposes hyperbolic distance + curvature.
 *
 * Design: wubu1-improvement-plan.md module #8 (AN22); research/060.
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_poincare_kv.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

#define BALL_RADIUS 0.8

/* FNV-1a 32-bit */
static uint32_t fnv1a(const char *s) {
    uint32_t h = 0x811c9dc5u;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 0x01000193u; }
    return h;
}

/* Signed feature hash for a segment: contributes +1 or -1 to one dim. */
static void add_segment(double *v, const char *seg, size_t len) {
    char buf[128];
    size_t n = len < sizeof(buf) - 1 ? len : sizeof(buf) - 1;
    memcpy(buf, seg, n);
    buf[n] = '\0';
    uint32_t h = fnv1a(buf);
    int idx = (int)(h % WUBU_POINCARE_DIM);
    double sgn = (h & 0x80000000u) ? 1.0 : -1.0;
    v[idx] += sgn;
}

int wubu_poincare_kv_map_path(const char *path, poincare_pos_t *out) {
    if (!path || !out || path[0] == '\0') return -1;

    double v[WUBU_POINCARE_DIM] = { 0.0 };
    const char *p = path;
    while (*p) {
        const char *sep = strchr(p, '/');
        size_t seglen = sep ? (size_t)(sep - p) : strlen(p);
        if (seglen > 0) add_segment(v, p, seglen);
        if (!sep) break;
        p = sep + 1;
    }

    /* L2 normalize, then scale into the ball (radius BALL_RADIUS < 1). */
    double n2 = 0.0;
    for (int i = 0; i < WUBU_POINCARE_DIM; i++) n2 += v[i] * v[i];
    if (n2 <= 1e-12) return -1; /* degenerate path (no segments) */

    double norm = sqrt(n2);
    double scale = BALL_RADIUS / norm;
    for (int i = 0; i < WUBU_POINCARE_DIM; i++) out->x[i] = v[i] * scale;
    out->radius = BALL_RADIUS;
    return 0;
}

double wubu_poincare_kv_distance(const poincare_pos_t *a,
                                 const poincare_pos_t *b) {
    double sq = 0.0;
    for (int i = 0; i < WUBU_POINCARE_DIM; i++) {
        double d = a->x[i] - b->x[i];
        sq += d * d;
    }
    double denom = (1.0 - a->radius * a->radius) *
                   (1.0 - b->radius * b->radius);
    if (denom <= 1e-12) return 1e18;
    return acosh(1.0 + 2.0 * sq / denom);
}

double wubu_poincare_kv_curvature(const poincare_pos_t *p) {
    if (!p) return 0.0;
    double c = 1.0 - p->radius * p->radius;
    return c > 0.0 ? c : 0.0;
}

double wubu_poincare_kv_path_distance(const char *pa, const char *pb) {
    poincare_pos_t a, b;
    if (wubu_poincare_kv_map_path(pa, &a) != 0) return 1e18;
    if (wubu_poincare_kv_map_path(pb, &b) != 0) return 1e18;
    return wubu_poincare_kv_distance(&a, &b);
}
