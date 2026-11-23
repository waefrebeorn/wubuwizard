/*
 * wubu_tv_accept.c -- EG-42 TV acceptance identity. Pure C11.
 */
#include "wubu_tv_accept.h"

#include <math.h>

static int rows_valid(const float *p, const float *q, int n) {
    float sp = 0.0f, sq = 0.0f;
    if (!p || !q || n <= 0) return 0;
    for (int i = 0; i < n; i++) {
        if (!(p[i] >= 0.0f) || !(q[i] >= 0.0f)) return 0;
        sp += p[i]; sq += q[i];
    }
    return fabsf(sp - 1.0f) < 1e-3f && fabsf(sq - 1.0f) < 1e-3f;
}

float wubu_tv_accept_prob(const float *p, const float *q, int n) {
    float s = 0.0f;
    if (!rows_valid(p, q, n)) return -1.0f;
    for (int i = 0; i < n; i++) s += p[i] < q[i] ? p[i] : q[i];
    return s;
}

float wubu_tv_distance(const float *p, const float *q, int n) {
    float s = 0.0f;
    if (!rows_valid(p, q, n)) return -1.0f;
    for (int i = 0; i < n; i++) s += fabsf(p[i] - q[i]);
    return 0.5f * s;
}

int wubu_tv_calibrate(const float *p, const float *q, int n,
                      unsigned seed, int trials, float *empirical) {
    unsigned state;
    int accepts = 0;
    if (!rows_valid(p, q, n) || trials <= 0 || !empirical) return -1;

    state = seed ? seed : 1u;
    for (int t = 0; t < trials; t++) {
        float u, cum = 0.0f, ratio = 0.0f;
        state = state * 1664525u + 1013904223u;
        u = (float)((state >> 8) & 0xFFFFFFu) / 16777216.0f;
        for (int i = 0; i < n; i++) {
            cum += q[i];
            if (u <= cum) {
                ratio = q[i] > 0.0f ? p[i] / q[i] : 0.0f;
                if (ratio > 1.0f) ratio = 1.0f;
                break;
            }
        }
        state = state * 1664525u + 1013904223u;
        u = (float)((state >> 8) & 0xFFFFFFu) / 16777216.0f;
        if (u <= ratio) accepts++;
    }
    *empirical = (float)accepts / (float)trials;
    return 0;
}
