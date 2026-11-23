/* test_eg45_lowrank.c -- EG-45a gate: low-rank key approximation. */
#include <stdio.h>
#include <math.h>

#include "wubu_lowrank_kv.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define ROWS 8
#define DIM 6
#define N (ROWS*DIM)

int main(void) {
    float ut[ROWS*2] = { 1,0, 0,1, 1,1, 2,-1, -1,2, 0.5f,0.5f, 3,0, -2,-1 };
    float vt[2*DIM] = { 1,2,-1,0.5f,3,-2,  0.5f,-1,2,1,-0.5f,1 };
    float K[N], out[N], U[ROWS*2], V[2*DIM];
    float e1, e2, r1;
    int i, j;

    for (i = 0; i < ROWS; i++)
        for (j = 0; j < DIM; j++)
            K[i*DIM+j] = ut[i*2]*vt[j] + ut[i*2+1]*vt[DIM+j];

    CHECK(wubu_lowrank_keys(K, ROWS, DIM, 2, 40, U, V, out) == 0);
    e2 = wubu_lowrank_relerr(K, out, N);
    printf("EG-45 rank2 relerr=%.6f\n", e2);
    CHECK(e2 < 0.05f);

    {
        float u1[ROWS], v1[DIM], out1[N];
        CHECK(wubu_lowrank_keys(K, ROWS, DIM, 1, 40, u1, v1, out1) == 0);
        e1 = wubu_lowrank_relerr(K, out1, N);
        printf("EG-45 rank1 relerr=%.6f\n", e1);
        CHECK(e1 > e2);
    }
    r1 = (float)(ROWS*2 + 2*DIM) / (float)N;
    printf("EG-45 stored ratio=%.3f\n", r1);
    CHECK(r1 < 0.7f);

    CHECK(wubu_lowrank_keys(NULL, ROWS, DIM, 2, 4, U, V, out) == -1);
    CHECK(wubu_lowrank_keys(K, 0, DIM, 2, 4, U, V, out) == -1);
    CHECK(wubu_lowrank_keys(K, ROWS, DIM, 0, 4, U, V, out) == -1);
    CHECK(wubu_lowrank_relerr(K, out, 0) == -1.0f);

    if (failures) {
        fprintf(stderr, "test_eg45_lowrank: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg45_lowrank: PASS\n");
    return 0;
}
