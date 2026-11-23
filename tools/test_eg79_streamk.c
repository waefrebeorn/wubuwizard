/* test_eg79_streamk.c -- EG-79 gate: Stream-K partition + partial-sum fixup. */
#include <stdio.h>
#include <math.h>

#include "wubu_streamk.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    float A[2][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    float B[4][3] = {{1, 0, 2}, {0, 1, 1}, {2, 0, 0}, {0, 3, 1}};
    float out[2][3], want[2][3];
    int lo[2], hi[2];
    double maxerr = 0.0;
    CHECK(wubu_streamk_partition(2, 2, lo, hi) == 0);
    CHECK(lo[0] == 0 && hi[0] == 1 && lo[1] == 1 && hi[1] == 2);
    CHECK(wubu_streamk_matmul(&A[0][0], &B[0][0], 2, 3, 4, 2, &out[0][0]) == 0);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) {
            double v = 0.0;
            for (int k = 0; k < 4; ++k) v += A[i][k] * B[k][j];
            want[i][j] = (float)v;
            if (fabs(out[i][j] - want[i][j]) > maxerr)
                maxerr = fabs(out[i][j] - want[i][j]);
        }
    CHECK(maxerr < 1e-4);
    CHECK(wubu_streamk_fixups(&out[0][0], 2, 3, 2) == 2);
    printf("EG-79 blocks=2 partition=[0,1)+[1,2) fixups=2 maxerr=%.8f\n", maxerr);
    CHECK(wubu_streamk_partition(3, 2, lo, hi) == 0);
    CHECK(lo[0] == 0 && hi[0] == 1 && lo[1] == 1 && hi[1] == 3);
    CHECK(wubu_streamk_partition(3, 4, lo, hi) == -1);
    CHECK(wubu_streamk_matmul(NULL, &B[0][0], 2, 3, 4, 2, &out[0][0]) == -1);

    if (failures) { fprintf(stderr, "test_eg79_streamk: FAIL (%d)\n", failures); return 1; }
    printf("test_eg79_streamk: PASS\n");
    return 0;
}
