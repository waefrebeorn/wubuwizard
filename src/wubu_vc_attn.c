/* wubu_vc_attn.c -- EG-77 V-residual contract gate. Strict C11. */
#include "wubu_vc_attn.h"

#include <stdlib.h>
#include <math.h>

static int valid_n(int n)
{
    return n >= WUBU_VC_TILE && n <= WUBU_VC_MAX_V &&
           n % WUBU_VC_TILE == 0;
}

int wubu_vc_quant(const float *v, int n, unsigned char *codes,
                  float *tile_means, float *tile_scales)
{
    int tiles;
    if (!v || !codes || !tile_means || !tile_scales || !valid_n(n)) return -1;
    tiles = n / WUBU_VC_TILE;
    for (int t = 0; t < tiles; ++t) {
        int lo = t * WUBU_VC_TILE;
        double sum = 0.0, mx = 0.0;
        float mean, scale;
        for (int i = 0; i < WUBU_VC_TILE; ++i) sum += v[lo + i];
        mean = (float)(sum / WUBU_VC_TILE);
        for (int i = 0; i < WUBU_VC_TILE; ++i) {
            double r = fabs((double)v[lo + i] - mean);
            if (r > mx) mx = r;
        }
        scale = mx > 0.0 ? (float)(mx / 7.0) : 1.0f;
        tile_means[t] = mean;
        tile_scales[t] = scale;
        for (int i = 0; i < WUBU_VC_TILE; ++i) {
            double q = ((double)v[lo + i] - mean) / scale + 7.0;
            if (q < 0.0) q = 0.0;
            if (q > 14.0) q = 14.0;
            codes[lo + i] = (unsigned char)(int)(q + 0.5);
        }
    }
    return tiles;
}

int wubu_vc_dequant(const unsigned char *codes, const float *tile_means,
                    const float *tile_scales, int n, float *out)
{
    int tiles;
    if (!codes || !tile_means || !tile_scales || !out || !valid_n(n)) return -1;
    tiles = n / WUBU_VC_TILE;
    for (int t = 0; t < tiles; ++t) {
        int lo = t * WUBU_VC_TILE;
        for (int i = 0; i < WUBU_VC_TILE; ++i)
            out[lo + i] = tile_means[t] +
                          ((int)codes[lo + i] - 7) * tile_scales[t];
    }
    return 0;
}

int wubu_vc_attention(const float *logits, const float *v, int n, float *out)
{
    double total = 0.0, acc = 0.0, peak;
    if (!logits || !v || !out || n < 1 || n > WUBU_VC_MAX_V) return -1;
    peak = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > peak) peak = logits[i];
    for (int i = 0; i < n; ++i) total += exp((double)logits[i] - peak);
    if (!(total > 0.0)) return -1;
    for (int i = 0; i < n; ++i)
        acc += (exp((double)logits[i] - peak) / total) * (double)v[i];
    *out = (float)acc;
    return 0;
}
