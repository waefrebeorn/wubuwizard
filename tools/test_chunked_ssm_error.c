/**
 * test_chunked_ssm_error.c — Compare chunked vs sequential SSM recurrence.
 *
 * Build: make test_chunked_ssm_error
 * Usage: ./test_chunked_ssm_error [T]
 */
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void wubu_ssm_chunked_recurrence(int B, int T,
    const float *q_norm, const float *k_norm, const float *v_conv,
    const float *beta_flat, const float *gate_flat,
    float *ssm_state, float *delta_out);

void wubu_ssm_sequential_recurrence(int B, int T,
    const float *q_norm, const float *k_norm, const float *v_conv,
    const float *beta_flat, const float *gate_flat,
    float *ssm_state, float *delta_out);

static float rand_float(float mag) {
    return (float)(rand() / (double)RAND_MAX) * mag - mag * 0.5f;
}

int main(int argc, char **argv) {
    int T = (argc > 1) ? atoi(argv[1]) : 64;
    const int d  = SSM_D_STATE;
    const int hk = SSM_K_HEADS;
    const int hv = SSM_V_HEADS;

    fprintf(stderr, "test_chunked_ssm_error T=%d d=%d hk=%d hv=%d\n", T, d, hk, hv);

    srand(42);
    size_t sz_q = (size_t)T * hk * d;
    size_t sz_v = (size_t)T * hv * d;

    float *q_norm    = (float *)calloc(sz_q, sizeof(float));
    float *k_norm    = (float *)calloc(sz_q, sizeof(float));
    float *v_conv    = (float *)calloc(sz_v, sizeof(float));
    float *beta_flat = (float *)calloc((size_t)T * hv, sizeof(float));
    float *gate_flat = (float *)calloc((size_t)T * hv, sizeof(float));

    if (!q_norm || !k_norm || !v_conv || !beta_flat || !gate_flat) {
        fprintf(stderr, "Alloc failed\n"); return 1;
    }

    // Inputs with small random values (same for both paths)
    for (int i = 0; i < T * hk * d; i++) q_norm[i] = rand_float(0.5f);
    for (int i = 0; i < T * hk * d; i++) k_norm[i] = rand_float(0.5f);
    for (int i = 0; i < T * hv * d; i++) v_conv[i] = rand_float(0.5f);
    for (int i = 0; i < T * hv; i++)  beta_flat[i] = rand_float(0.5f) + 0.5f;
    for (int i = 0; i < T * hv; i++)  gate_flat[i] = fabsf(rand_float(0.5f));

    fprintf(stderr, "Inputs populated, calling chunked...\n");

    // Chunked path (CS=2)
    float *state_c = (float *)calloc((size_t)hv * d * d, sizeof(float));
    float *delta_c = (float *)calloc((size_t)hv * T * d, sizeof(float));
    wubu_ssm_chunked_recurrence(1, T, q_norm, k_norm, v_conv,
                                 beta_flat, gate_flat, state_c, delta_c);
    fprintf(stderr, "Chunked done, calling sequential...\n");

    // Sequential path
    float *state_s = (float *)calloc((size_t)hv * d * d, sizeof(float));
    float *delta_s = (float *)calloc((size_t)hv * T * d, sizeof(float));
    wubu_ssm_sequential_recurrence(1, T, q_norm, k_norm, v_conv,
                                    beta_flat, gate_flat, state_s, delta_s);
    fprintf(stderr, "Sequential done, computing metrics...\n");

    size_t out_sz = (size_t)hv * T * d;
    double dot = 0, na = 0, nb = 0;
    float max_err = 0;
    for (int i = 0; i < out_sz; i++) {
        float a = delta_c[i], b = delta_s[i];
        dot += (double)a * b;
        na += (double)a * a;
        nb += (double)b * b;
        float e = fabsf(a - b);
        if (e > max_err) max_err = e;
    }
    double denom = sqrt(na) * sqrt(nb);
    float cs = (denom > 1e-30) ? (float)(dot / denom) : 1.0f;

    printf("T=%d | cos-sim=%8.6f max-err=%8.6f\n", T, cs, max_err);

    free(q_norm); free(k_norm); free(v_conv);
    free(beta_flat); free(gate_flat);
    free(state_c); free(delta_c);
    free(state_s); free(delta_s);
    return 0;
}
