#ifndef WUBU_BITNEST_H
#define WUBU_BITNEST_H

/*
 * wubu_bitnest.h -- EG-48: nested W4 draft inside W8 target, one physical
 * weight representation (BitNest lineage), reimplemented in C11.
 *
 * A block of floats is quantized once onto a signed 4-bit grid and stored as
 * int8 bytes. The W8 verification path reads those bytes directly; the W4
 * draft path reads the same bytes with a half-scale. No second weight copy.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n;
    float w8_scale;
    float w4_scale;
    signed char *data; /* one physical representation */
} wubu_bitnest_t;

int wubu_bitnest_init(wubu_bitnest_t *b, int n);
int wubu_bitnest_quantize(wubu_bitnest_t *b, const float *src, int n);
void wubu_bitnest_free(wubu_bitnest_t *b);

/* Dot product with the target (W8 view) or draft (W4 view). */
float wubu_bitnest_dot_w8(const wubu_bitnest_t *b, const float *x);
float wubu_bitnest_dot_w4(const wubu_bitnest_t *b, const float *x);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_BITNEST_H */
