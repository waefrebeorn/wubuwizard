/*
 * wubu_bitnest.c -- EG-48 nested W4-in-W8 representation. Pure C11.
 */
#include "wubu_bitnest.h"

#include <math.h>
#include <stdlib.h>

int wubu_bitnest_init(wubu_bitnest_t *b, int n) {
    if (!b || n <= 0) return -1;
    b->data = (signed char *)malloc((size_t)n);
    if (!b->data) return -1;
    b->n = n;
    b->w8_scale = 1.0f;
    b->w4_scale = 0.5f;
    return 0;
}

int wubu_bitnest_quantize(wubu_bitnest_t *b, const float *src, int n) {
    float absmax = 0.0f;
    if (!b || !b->data || !src || n != b->n || n <= 0) return -1;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > absmax) absmax = a;
    }
    /* Signed 4-bit grid [-8,7]; W8 view uses the full scale, W4 half. */
    b->w8_scale = absmax > 0.0f ? absmax / 7.0f : 1.0f;
    b->w4_scale = b->w8_scale * 0.5f;
    for (int i = 0; i < n; i++) {
        float q = src[i] / b->w8_scale;
        int v = (int)(q >= 0.0f ? floorf(q + 0.5f) : ceilf(q - 0.5f));
        if (v > 7) v = 7;
        if (v < -8) v = -8;
        b->data[i] = (signed char)v;
    }
    return 0;
}

float wubu_bitnest_dot_w8(const wubu_bitnest_t *b, const float *x) {
    float sum = 0.0f;
    if (!b || !b->data || !x) return 0.0f;
    for (int i = 0; i < b->n; i++)
        sum += ((float)b->data[i] * b->w8_scale) * x[i];
    return sum;
}

float wubu_bitnest_dot_w4(const wubu_bitnest_t *b, const float *x) {
    float sum = 0.0f;
    if (!b || !b->data || !x) return 0.0f;
    for (int i = 0; i < b->n; i++)
        sum += ((float)b->data[i] * b->w4_scale) * x[i];
    return sum;
}

void wubu_bitnest_free(wubu_bitnest_t *b) {
    if (!b) return;
    free(b->data);
    b->data = NULL;
    b->n = 0;
}
