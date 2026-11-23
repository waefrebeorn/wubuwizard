/* wubu_kv8.c -- EG-66d kv8 INT8 K/V codec. Strict C11. */
#include "wubu_kv8.h"

#include <math.h>

int wubu_kv8_pack(const float *row, int n, signed char *codes, float *scale)
{
    float mx = 0.0f, sc;
    if (!row || !codes || !scale || n < 1 || n > WUBU_KV8_MAX_DIM) return -1;
    for (int i = 0; i < n; ++i) {
        float a = fabsf(row[i]);
        if (a > mx) mx = a;
    }
    sc = mx > 0.0f ? mx / 127.0f : 1.0f;
    *scale = sc;
    for (int i = 0; i < n; ++i) {
        double q = (double)row[i] / sc;
        if (q > 127.0) q = 127.0;
        if (q < -127.0) q = -127.0;
        codes[i] = (signed char)(int)(q >= 0.0 ? q + 0.5 : q - 0.5);
    }
    return 0;
}

int wubu_kv8_unpack(const signed char *codes, float scale, int n, float *out)
{
    if (!codes || !out || n < 1 || n > WUBU_KV8_MAX_DIM ||
        !(scale > 0.0f)) return -1;
    for (int i = 0; i < n; ++i) out[i] = (float)codes[i] * scale;
    return 0;
}

size_t wubu_kv8_storage_bytes(int n)
{
    if (n < 1 || n > WUBU_KV8_MAX_DIM) return 0;
    return (size_t)n + sizeof(float);
}
