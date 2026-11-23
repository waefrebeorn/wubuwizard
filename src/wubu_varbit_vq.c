/* wubu_varbit_vq.c -- EG-45b variable-bit VQ codec. Strict C11. */
#include "wubu_varbit_vq.h"

#include <math.h>

static int widths_ok(const int *bits, int n)
{
    if (!bits || n < 1 || n > WUBU_VQ_MAX_ITEMS) return 0;
    for (int i = 0; i < n; ++i)
        if (bits[i] < 1 || bits[i] > 8) return 0;
    return 1;
}

static void put_bits(unsigned char *out, size_t *pos, int code, int width)
{
    for (int b = width - 1; b >= 0; --b) {
        size_t at = (*pos)++;
        if ((code >> b) & 1) out[at >> 3] |= (unsigned char)(1u << (7u - (at & 7u)));
    }
}

static int get_bits(const unsigned char *in, size_t *pos, int width)
{
    int code = 0;
    for (int b = width - 1; b >= 0; --b) {
        size_t at = (*pos)++;
        int bit = (in[at >> 3] >> (7u - (at & 7u))) & 1u;
        code = (code << 1) | bit;
    }
    return code;
}

size_t wubu_varbit_vq_encoded_bytes(const int *bits, int n)
{
    size_t total = 0;
    if (!widths_ok(bits, n)) return 0;
    for (int i = 0; i < n; ++i) total += (size_t)bits[i];
    return (total + 7u) / 8u;
}

int wubu_varbit_vq_encode(const float *values, const int *bits, int n,
                          unsigned char *out)
{
    size_t pos = 0;
    if (!widths_ok(bits, n) || !values || !out ||
        wubu_varbit_vq_encoded_bytes(bits, n) == 0) return -1;
    for (int i = 0; i < n; ++i) {
        int maxcode, q;
        if (!(values[i] >= -1.0f && values[i] <= 1.0f)) return -1;
        maxcode = (1 << bits[i]) - 1;
        q = (int)floor(((double)values[i] + 1.0) * 0.5 * maxcode + 0.5);
        if (q < 0) q = 0;
        if (q > maxcode) q = maxcode;
        put_bits(out, &pos, q, bits[i]);
    }
    return 0;
}

int wubu_varbit_vq_decode(const unsigned char *in, const int *bits, int n,
                          float *out)
{
    size_t pos = 0;
    if (!widths_ok(bits, n) || !in || !out ||
        wubu_varbit_vq_encoded_bytes(bits, n) == 0) return -1;
    for (int i = 0; i < n; ++i) {
        int maxcode = (1 << bits[i]) - 1;
        int code = get_bits(in, &pos, bits[i]);
        out[i] = (float)((2.0 * code / maxcode) - 1.0);
    }
    return 0;
}
