/* test_eg62_dspark.c -- EG-62 gate: correction + confidence budget gain. */
#include <stdio.h>

#include "wubu_dspark.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define V 8
#define N 4

int main(void) {
    float base[V], target[V], A[V*64], B[64], draft0[V], draft2[V];
    float conf[V], confpos[N] = { 0.9f, 0.8f, 0.3f, 0.2f };
    int adapt[N], uniform[N];
    double acc_adapt, acc_uniform;

    for (int v = 0; v < V; v++) {
        base[v] = 1.0f / V;
        target[v] = (float)(V - v) / (float)(V * (V + 1) / 2);
        for (int k = 0; k < 64; k++) A[v*64+k] = 0.0f;
        A[v*64+0] = target[v] - base[v];
        A[v*64+1] = 0.02f * (float)((v % 3) - 1);
    }
    B[0] = 1.0f;
    B[1] = 0.5f;

    CHECK(wubu_dspark_correct(base, A, B, V, 0, draft0) == 0);
    CHECK(wubu_dspark_correct(base, A, B, V, 2, draft2) == 0);
    CHECK(wubu_dspark_confidence(draft0, target, V, &conf[0]) == 0);
    CHECK(wubu_dspark_confidence(draft2, target, V, &conf[1]) == 0);
    printf("EG-62 conf rank0=%.5f rank2=%.5f\n", conf[0], conf[1]);
    CHECK(conf[1] > conf[0] + 0.1);

    CHECK(wubu_dspark_budget(confpos, N, 12, adapt) == 0);
    for (int i = 0; i < N; i++) uniform[i] = 12 / N;
    acc_adapt = wubu_dspark_accepted(confpos, adapt, N);
    acc_uniform = wubu_dspark_accepted(confpos, uniform, N);
    printf("EG-62 budgets adapt=[%d %d %d %d] accepted=%.4f uniform=%.4f\n",
           adapt[0], adapt[1], adapt[2], adapt[3], acc_adapt, acc_uniform);
    CHECK(acc_adapt > acc_uniform + 1.0);

    CHECK(wubu_dspark_correct(base, A, B, V, -1, draft2) == -1);
    CHECK(wubu_dspark_confidence(draft2, target, 0, &conf[0]) == -1);
    CHECK(wubu_dspark_budget(confpos, N, N - 1, adapt) == -1);

    if (failures) {
        fprintf(stderr, "test_eg62_dspark: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg62_dspark: PASS\n");
    return 0;
}
