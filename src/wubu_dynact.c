/*
 * wubu_dynact.c -- EG-26 per-token activation quantization. Pure C11.
 */
#include "wubu_dynact.h"

#include <math.h>

int wubu_dynact_quant_row(const float *x, int n, int bits,
                          unsigned char *out, float *scale) {
    float mx = 0.0f, sc;
    int qmax;
    if (!x || !out || !scale || n <= 0 || bits < 2 || bits > 8) return -1;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > mx) mx = a;
    }
    qmax = (1 << (bits - 1)) - 1;
    sc = mx > 0.0f ? mx / (float)qmax : 1.0f;
    for (int i = 0; i < n; i++) {
        int q = (int)roundf(x[i] / sc);
        if (q > qmax) q = qmax;
        if (q < -qmax) q = -qmax;
        out[i] = (unsigned char)(q + qmax);
    }
    *scale = sc;
    return 0;
}

int wubu_dynact_dequant_row(const unsigned char *q, int n, int bits,
                            float scale, float *out) {
    int qmax, zero;
    if (!q || !out || n <= 0 || bits < 2 || bits > 8 || scale <= 0.0f) return -1;
    qmax = (1 << (bits - 1)) - 1;
    zero = qmax;
    for (int i = 0; i < n; i++)
        out[i] = (float)((int)q[i] - zero) * scale;
    return 0;
}

float wubu_dynact_maxerr(const float *a, const float *b, int n) {
    float e = 0.0f;
    if (!a || !b || n <= 0) return -1.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > e) e = d;
    }
    return e;
}
