/*
 * wubu_dspark.c -- EG-62 correction / confidence / budget. Pure C11.
 */
#include "wubu_dspark.h"

#include <math.h>

int wubu_dspark_correct(const float *base, const float *A, const float *B,
                        int vocab, int rank, float *draft) {
    float sum = 0.0f;
    if (!base || !A || !B || !draft || vocab <= 0 || rank < 0) return -1;
    for (int v = 0; v < vocab; v++) {
        float x = base[v];
        for (int k = 0; k < rank; k++) x += A[(size_t)v * 64 + k] * B[k];
        draft[v] = x > 0.0f ? x : 0.0f;
        sum += draft[v];
    }
    if (sum <= 0.0f) return -1;
    for (int v = 0; v < vocab; v++) draft[v] /= sum;
    return 0;
}

int wubu_dspark_confidence(const float *draft, const float *target,
                           int vocab, float *conf) {
    float tv = 0.0f;
    if (!draft || !target || !conf || vocab <= 0) return -1;
    for (int v = 0; v < vocab; v++) tv += fabsf(draft[v] - target[v]);
    *conf = 1.0f - 0.5f * tv;
    if (*conf < 0.0f) *conf = 0.0f;
    if (*conf > 1.0f) *conf = 1.0f;
    return 0;
}

int wubu_dspark_budget(const float *conf, int n, int total, int *budgets) {
    if (!conf || !budgets || n <= 0 || total < n) return -1;
    for (int i = 0; i < n; i++) budgets[i] = 1;
    for (int left = total - n; left > 0; left--) {
        int best = 0;
        for (int i = 1; i < n; i++)
            if (conf[i] > conf[best]) best = i;
        budgets[best]++;
    }
    return 0;
}

double wubu_dspark_accepted(const float *conf, const int *budgets, int n) {
    double sum = 0.0;
    if (!conf || !budgets || n <= 0) return -1.0;
    for (int i = 0; i < n; i++) sum += (double)budgets[i] * conf[i];
    return sum;
}
