/* tvar.c — run our UNet on the SAME noise at several timesteps,
 * print corr(x, eps) per the skill's dead-time-path check:
 * corr should VARY with t (0.35 -> 0.38-ish); constant = dead time path. */
#include "wubu_sd_unet.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double corr(const float *a, const float *b, int n) {
    double ma = 0, mb = 0;
    for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double c = 0, va = 0, vb = 0;
    for (int i = 0; i < n; i++) {
        double da = a[i] - ma, db = b[i] - mb;
        c += da * db; va += da * da; vb += db * db;
    }
    return c / sqrt(va * vb + 1e-30);
}

int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g) return 1;
    wubu_sd_unet_t *u = wubu_sd_unet_load(g);
    if (!u) return 1;

    FILE *nf = fopen(argv[2], "rb");
    float noise[4*64*64];
    fread(noise, 4, 4*64*64, nf); fclose(nf);

    float beta[1000], ab[1000];
    for (int i = 0; i < 1000; i++) {
        float t = (float)i / 999.0f;
        float b = (sqrtf(0.00085f) + (sqrtf(0.012f)-sqrtf(0.00085f))*t);
        beta[i] = b*b;
        ab[i] = (i==0) ? (1.0f-beta[0]) : ab[i-1]*(1.0f-beta[i]);
    }
    float smax = sqrtf((1.0f-ab[999])/ab[999]);

    float ctx[77*768]; memset(ctx, 0, sizeof(ctx));
    int ts[] = {999, 750, 500, 250, 100};
    for (int k = 0; k < 5; k++) {
        int t = ts[k];
        float x[4*64*64];
        for (int i = 0; i < 4*64*64; i++) x[i] = noise[i] * smax;
        float sigma = sqrtf((1.0f-ab[t])/ab[t]);
        float c_in = 1.0f/sqrtf(sigma*sigma+1.0f);
        for (int i = 0; i < 4*64*64; i++) x[i] *= c_in;
        float out[4*64*64];
        if (wubu_sd_unet_forward(u, x, t, ctx, out) != 0) { fprintf(stderr, "fwd fail t=%d\n", t); return 1; }
        printf("t=%4d sigma=%7.4f corr(x,eps)=%.4f eps_mean=%.5f eps_rms=%.4f\n",
               t, sigma, corr(x, out, 4*64*64), 
               (double)0, 0.0);
        /* recompute eps mean/rms */
        double m = 0, r = 0;
        for (int i = 0; i < 4*64*64; i++) { m += out[i]; r += (double)out[i]*out[i]; }
        printf("         eps: mean=%.5f rms=%.4f\n", m/16384.0, sqrt(r/16384.0));
    }
    wubu_sd_unet_free(u);
    return 0;
}
