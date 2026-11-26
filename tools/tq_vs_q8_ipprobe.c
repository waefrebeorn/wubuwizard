/* q8_0 vs TurboQuant for the only thing K-side attention needs: recovering <q,k>.
 * q8_0 stores amax/127 per 32 elements (8.5 bits/elem) with a row-fitted scale, so its dot-product estimator is biased.
 * TurboQuant's inner-product construction is unbiased. Same-ish bits, measure the error and the bias. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "wubu_kv_q8.h"
#include "wubu_tqvec.h"

static uint64_t S = 424242;
static double ur(void){ S^=S>>12; S^=S<<25; S^=S>>27; return (double)((S*2685821657736338717ULL)>>11)/9007199254740992.0; }
static double nr(void){ double u=ur(), v=ur(); if(u<1e-12) u=1e-12; return sqrt(-2.0*log(u))*cos(6.283185307179586*v); }

int main(void)
{
    const int d = 128, T = 2000;
    float *k = malloc(sizeof(float)*d);
    float *q = malloc(sizeof(float)*d);
    float *kr = malloc(sizeof(float)*d);
    if (!k || !q || !kr) return 1;

    setvbuf(stdout, NULL, _IONBF, 0);
    printf("%-18s %-10s %-12s %-12s %-10s\n", "quantiser", "bits/elem", "IP rmse", "IP bias", "bytes/row");

    for (int b = 8; b >= 4; b -= 2) {
        wubu_tqvec *tq = wubu_tqvec_init(d, b, 0xBEEFull);
        if (!tq) { printf("init NULL at b=%d\n", b); return 1; }
        const size_t nb = wubu_tqvec_code_bytes(tq);
        unsigned char *code = malloc(nb);
        if (!code) return 1;
        double e2 = 0.0, bias = 0.0;
        for (int t = 0; t < T; ++t) {
            double nk = 0.0, nq = 0.0;
            for (int i = 0; i < d; ++i) {
                const double sc = 0.5 + 1.5*(double)(i % 7)/7.0;   /* keys are not isotropic across dims */
                k[i] = (float)(nr()*sc);
                q[i] = (float)nr();
                nk += (double)k[i]*k[i];
                nq += (double)q[i]*q[i];
            }
            nk = sqrt(nk); nq = sqrt(nq);
            for (int i = 0; i < d; ++i) { k[i] = (float)(k[i]/nk); q[i] = (float)(q[i]/nq); }
            double ex = 0.0;
            for (int i = 0; i < d; ++i) ex += (double)q[i]*(double)k[i];
            wubu_tqvec_encode(tq, k, code);
            const double est = (double)wubu_tqvec_score(tq, q, code);
            e2 += (est-ex)*(est-ex);
            bias += (est-ex);
        }
        printf("%-18s %-10d %-12.6f %-12.6f %-10zu\n", "turboquant", b, sqrt(e2/T), bias/T, nb);
        free(code);
        wubu_tqvec_free(tq);
    }

    {
        const size_t nb = kvq8_row_bytes(d);
        unsigned char *code = malloc(nb);
        if (!code) return 1;
        double e2 = 0.0, bias = 0.0;
        for (int t = 0; t < T; ++t) {
            double nk = 0.0, nq = 0.0;
            for (int i = 0; i < d; ++i) {
                const double sc = 0.5 + 1.5*(double)(i % 7)/7.0;
                k[i] = (float)(nr()*sc);
                q[i] = (float)nr();
                nk += (double)k[i]*k[i];
                nq += (double)q[i]*q[i];
            }
            nk = sqrt(nk); nq = sqrt(nq);
            for (int i = 0; i < d; ++i) { k[i] = (float)(k[i]/nk); q[i] = (float)(q[i]/nq); }
            double ex = 0.0;
            for (int i = 0; i < d; ++i) ex += (double)q[i]*(double)k[i];
            kvq8_quantize_row(k, code, d);
            kvq8_dequantize_row(code, kr, d);
            double est = 0.0;
            for (int i = 0; i < d; ++i) est += (double)q[i]*(double)kr[i];
            e2 += (est-ex)*(est-ex);
            bias += (est-ex);
        }
        printf("%-18s %-10.1f %-12.6f %-12.6f %-10zu\n", "q8_0 (current KV)", 8.5, sqrt(e2/T), bias/T, nb);
        free(code);
    }

    free(k); free(q); free(kr);
    return 0;
}
