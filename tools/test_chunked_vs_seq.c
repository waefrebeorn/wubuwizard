/**
 * test_chunked_vs_seq.c — Debug chunked vs sequential SSM discrepancy.
 *
 * Tests at CS=1 where chunked should match sequential EXACTLY.
 * If CS=1 differs, the input/output layout is wrong.
 * If CS=1 matches but CS=2 differs, the chunking formula has a bug.
 *
 * Build: make test_chunked_ssm_error
 * Usage: ./test_chunked_vs_seq
 */
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// The chunked function uses #define CS 2 hardcoded in wubu_ssm_chunked.c
// To test at CS=1 we'd need to recompile with -DCS=1
// For now: test at CS=2 and compare specific output positions

void wubu_ssm_chunked_recurrence(int B, int T,
    const float *q_norm, const float *k_norm, const float *v_conv,
    const float *beta_flat, const float *gate_flat,
    float *ssm_state, float *delta_out);

void wubu_ssm_sequential_recurrence(int B, int T,
    const float *q_norm, const float *k_norm, const float *v_conv,
    const float *beta_flat, const float *gate_flat,
    float *ssm_state, float *delta_out);

int main(void) {
    int T = 4;
    const int d  = SSM_D_STATE;
    const int hk = SSM_K_HEADS;
    const int hv = SSM_V_HEADS;

    // Simple synthetic test: q=1, k=1, v=1, beta=1, gate=0
    // For these inputs the recurrence should produce well-defined outputs
    size_t sz_q = (size_t)T * hk * d;
    size_t sz_v = (size_t)T * hv * d;

    float *q_norm    = (float *)calloc(sz_q, sizeof(float));
    float *k_norm    = (float *)calloc(sz_q, sizeof(float));
    float *v_conv    = (float *)calloc(sz_v, sizeof(float));
    float *beta_flat = (float *)calloc((size_t)T * hv, sizeof(float));
    float *gate_flat = (float *)calloc((size_t)T * hv, sizeof(float));

    // Constant inputs: q = 0.1, k = 0.1, v = 0.2, beta = 0.5, gate = 0.0
    for (int i = 0; i < T * hk * d; i++) { q_norm[i] = 0.1f; k_norm[i] = 0.1f; }
    for (int i = 0; i < T * hv * d; i++) v_conv[i] = 0.2f;
    for (int i = 0; i < T * hv; i++) { beta_flat[i] = 0.5f; gate_flat[i] = 0.0f; }

    printf("=== Constant input test ===\n");
    printf("T=%d d=%d hk=%d hv=%d\n", T, d, hk, hv);

    float *state_c = (float *)calloc((size_t)hv * d * d, sizeof(float));
    float *delta_c = (float *)calloc((size_t)hv * T * d, sizeof(float));
    float *state_s = (float *)calloc((size_t)hv * d * d, sizeof(float));
    float *delta_s = (float *)calloc((size_t)hv * T * d, sizeof(float));

    wubu_ssm_chunked_recurrence(1, T, q_norm, k_norm, v_conv,
                                 beta_flat, gate_flat, state_c, delta_c);
    wubu_ssm_sequential_recurrence(1, T, q_norm, k_norm, v_conv,
                                    beta_flat, gate_flat, state_s, delta_s);

    size_t out_sz = (size_t)hv * T * d;
    printf("First 20 delta values (chunked vs sequential):\n");
    for (int i = 0; i < 20 && i < (int)out_sz; i++) {
        printf("  [%d] chunked=%+.6f seq=%+.6f\n", i, delta_c[i], delta_s[i]);
    }
    printf("\nState[0] first 8 values:\n");
    for (int i = 0; i < 8 && i < hv * d * d; i++) {
        printf("  [%d] chunked_state=%+.6f seq_state=%+.6f\n", i, state_c[i], state_s[i]);
    }

    // Metrics
    double dot = 0, na = 0, nb = 0;
    float max_err = 0;
    for (int i = 0; i < (int)out_sz; i++) {
        float a = delta_c[i], b = delta_s[i];
        dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
        float e = fabsf(a - b);
        if (e > max_err) max_err = e;
    }
    float cs = (sqrt(na)*sqrt(nb) > 1e-30) ? (float)(dot / (sqrt(na)*sqrt(nb))) : 1.0f;
    printf("\nOverall: cos-sim=%8.6f max-err=%8.6f\n", cs, max_err);

    free(q_norm); free(k_norm); free(v_conv);
    free(beta_flat); free(gate_flat);
    free(state_c); free(delta_c);
    free(state_s); free(delta_s);
    return 0;
}
